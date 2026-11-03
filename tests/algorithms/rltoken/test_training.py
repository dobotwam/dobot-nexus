import ast
import logging
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dobot_nexus.algorithms.rltoken.configuration import LearningConfig
from dobot_nexus.algorithms.rltoken.training import RLTokenTrainer, validate_training
from dobot_nexus.core.config import load_config

ROOT = Path(__file__).resolve().parents[3]


class Handle:
    def __init__(self, value):
        self.value = value

    def done(self):
        return True


class CoordinatorFixture:
    """Immediate scalar handles exercise the coordinator clocks, not Ray internals."""

    def __init__(self):
        self.step = self.chunks = self.applied_actor_update = 0
        self.barriers = []
        self.actor = SimpleNamespace(update=self.update, snapshot=self.snapshot)
        self.env = SimpleNamespace(collect_chunk=self.collect)
        self.rollout = SimpleNamespace(
            serve_slice=lambda: Handle({"applied_actor_update": self.applied_actor_update}),
            export_deployment=lambda path: Handle({"path": path}),
        )

    def start(self):
        return {
            "updates": 0,
            "critic_updates": 0,
            "actor_updates": 0,
            "admitted": 0,
            "budget": 0,
            "metrics": {},
        }

    def collect(self, actor_ready=False):
        self.chunks += 1
        return Handle({"chunks": self.chunks, "env_steps": self.chunks, "faulted": False})

    def update(self, max_updates):
        self.step += max_updates
        return Handle(
            {
                "updates": max_updates,
                "critic_updates": self.step,
                "actor_updates": self.step,
                "admitted": self.chunks,
                "budget": 0,
                "metrics": {},
            }
        )

    def update_rollout_weights(self, no_wait=False):
        self.applied_actor_update = self.step

    def finish_sync(self):
        pass

    def wait(self, handle):
        return handle.value

    def snapshot(self):
        self.barriers.append(self.step)
        return Handle(f"checkpoint-{self.step}")


class TrainingTest(unittest.TestCase):
    def test_terminal_recipe_requires_an_explicit_known_name(self):
        from dobot_nexus.algorithms.rltoken.configuration import recipe_identity

        config = load_config(ROOT / "examples/configs/rltoken/stage2_simulation.yaml")
        config.algorithm.recipe = "rlt_ac_terminal_v2"
        validate_training(config)
        self.assertNotEqual(recipe_identity(config.algorithm.recipe),
                            recipe_identity("rlt_ac_fixed_v1"))
        config.algorithm.recipe = "rlt_ac_terminal_typo"
        with self.assertRaisesRegex(ValueError, "Unknown RLToken recipe"):
            validate_training(config)

    def test_simulation_batch_size_is_independent_of_replay_and_robot_is_single(self):
        config = load_config(ROOT / "examples/configs/rltoken/stage2_simulation.yaml")
        validate_training(config)
        self.assertEqual(config.environment.batch_size, 16)
        self.assertEqual(config.algorithm.learning.batch_size, 256)
        for name, value in (
            ("kind", "robot"),
            ("batch_size", 0),
            ("batch_size", True),
            ("batch_size", 1.5),
            ("kind", "simulator"),
        ):
            invalid = load_config(ROOT / "examples/configs/rltoken/stage2_simulation.yaml")
            invalid.environment[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_training(invalid)
        config.environment.batch_size = 1
        config.runtime.resources.env_gpu = None
        validate_training(config)
        for index in (-1, True, 1.5):
            config.runtime.resources.env_gpu = index
            with self.assertRaisesRegex(ValueError, "env_gpu"):
                validate_training(config)

    def config(self):
        return load_config(ROOT / "examples/configs/rltoken/stage2.yaml")

    def test_preflight_rejects_conflicting_bc_flags(self):
        validate_training(self.config())
        cases = (
            ("runtime.launcher", "local"),
            ("runtime.fsdp.enabled", True),
            ("runtime.resources.learner_gpu", 0),
            ("train.ema_decay", 0.99),
            ("train.grad_accum_steps", 2),
            ("environment.min_actor_updates", 0),
        )
        for path, value in cases:
            config = self.config()
            target = config
            keys = path.split(".")
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_training(config)

    def test_single_driver_launcher_with_multiple_visible_gpus(self):
        result = subprocess.run(
            [
                str(ROOT / "examples/train.sh"),
                "--config",
                "examples/configs/rltoken/stage2.yaml",
                "--gpus",
                "0,1",
                "--dry-run",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", result.stdout)
        self.assertNotIn("torch.distributed.run", result.stdout)
        self.assertIn("processes=1", result.stderr)

    def test_coordinator_stops_exactly_at_update_limit_and_saves_at_barriers(self):
        trainer = RLTokenTrainer.__new__(RLTokenTrainer)
        trainer.config = self.config()
        trainer.config.train.max_steps = 3
        trainer.config.train.save_every = 2
        trainer.learning = LearningConfig(warmup=256)
        with tempfile.TemporaryDirectory() as directory:
            trainer.experiment = SimpleNamespace(
                run_dir=Path(directory), logger=logging.getLogger("test")
            )
            execution = CoordinatorFixture()
            result = trainer._coordinate(execution)
        self.assertEqual(result.global_step, 3)
        self.assertEqual(execution.barriers, [2, 3])
        self.assertEqual(result.checkpoint, "checkpoint-3")
        self.assertEqual(result.metrics["deployment_ready"], 0)

    def test_pretrained_actor_starts_immediately_and_drains_episode_budget(self):
        class Execution(CoordinatorFixture):
            def __init__(self):
                super().__init__()
                self.budget = 0
                self.ready = []
                self.update_chunks = []
                self.applied_actor_update = 1
                self.rollout.rng_state = lambda: Handle(None)
            def status(self, updates=0):
                return dict(updates=updates, critic_updates=self.step, actor_updates=self.step//2,
                    actor_revision=1+self.step//2, ready_revision=1, pretrained_ready=True,
                    admitted=2+self.chunks, replay_size=2+self.chunks, budget=self.budget, metrics={})
            def start(self):
                return self.status()
            def collect(self, actor_ready=False):
                self.ready.append(actor_ready)
                self.chunks += 1
                if self.chunks == 2:
                    self.budget += 4
                return Handle(dict(chunks=self.chunks, env_steps=self.chunks*2, faulted=False,
                                   episode_ended=self.chunks == 2))
            def update(self, max_updates):
                count = min(max_updates, self.budget)
                if count:
                    self.update_chunks.append(self.chunks)
                self.step += count
                self.budget -= count
                return Handle(self.status(count))
            def update_rollout_weights(self, no_wait=False):
                self.applied_actor_update = 1+self.step//2
            def snapshot(self, **kwargs):
                return super().snapshot()
        trainer = RLTokenTrainer.__new__(RLTokenTrainer)
        trainer.config = self.config()
        trainer.config.train.max_steps = 10
        trainer.config.train.save_every = 10
        trainer.config.environment.max_chunks = 2
        trainer.config.environment.update_schedule = "episode_end"
        trainer.config.environment.drain_on_stop = True
        trainer.learning = LearningConfig(warmup=1, critic_actor_ratio=2, td3={})
        with tempfile.TemporaryDirectory() as directory:
            trainer.experiment = SimpleNamespace(run_dir=Path(directory), logger=logging.getLogger("test"))
            execution = Execution()
            result = trainer._coordinate(execution)
        self.assertEqual(result.global_step, 4)
        self.assertEqual(execution.ready, [True, True])
        self.assertEqual(execution.update_chunks, [2]*4)
        self.assertEqual(execution.chunks, 2)
        self.assertEqual(result.metrics["budget"], 0)
        self.assertEqual(result.metrics["deployment_ready"], 1)

    def test_import_boundaries_are_lazy_and_algorithms_do_not_import_models(self):
        for path in (ROOT / "src/dobot_nexus/algorithms/rltoken").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                for name in names:
                    self.assertFalse(name.startswith("dobot_nexus.models.pi"), str(path))
                    if name.split(".")[0] in {"rlinf", "ray"}:
                        self.assertIn(path.name, {"execution_rlinf.py", "workers_rlinf.py"})
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import dobot_nexus.algorithms.rltoken.policy; "
                "import dobot_nexus.algorithms.rltoken.training; "
                "assert not any(n.split('.')[0] in {'rlinf','ray'} for n in sys.modules)",
            ],
            check=True,
            capture_output=True,
        )
