from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from dobot_nexus.core.artifacts import (
    _validate_state_dict,
    load_manifest,
    resolve_artifact,
    resume_fingerprint,
)
from dobot_nexus.core.config import Config
from dobot_nexus.core.experiment import Experiment
from dobot_nexus.training import BCTrainer
from dobot_nexus.training.ema import ParameterEMA
from dobot_nexus.training.runtime import LocalRuntime
from tests.fakes import DummyVLAModule


def _wrap(value):
    if isinstance(value, dict):
        return Config({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _config(
    root: str,
    *,
    name: str,
    max_steps: int,
    save_every: int = 1,
    resume_from=None,
    resume_limit=2,
    target="tests.fakes:DummyVLAModule",
):
    return _wrap(
        {
            "schema_version": 1,
            "task": "train",
            "algorithm": "bc",
            "exp": {"name": name, "output_dir": root, "seed": 7},
            "model": {
                "name": "dummy",
                "target": target,
                "checkpoint": "unused.pt",
                "state_dim": 2,
                "action_dim": 2,
                "action_horizon": 3,
            },
            "data": {"batch_size": 2, "num_workers": 0},
            "runtime": {"launcher": "local", "device": "cpu", "precision": "fp32"},
            "optim": {"name": "adamw", "lr": 0.1, "weight_decay": 0.0},
            "scheduler": {"name": "constant"},
            "train": {
                "max_steps": max_steps,
                "grad_accum_steps": 1,
                "grad_clip_norm": 1.0,
                "log_every": 1,
                "save_every": save_every,
                "resume_from": resume_from,
                "trust_resume_checkpoint": True,
                "resume_limit": resume_limit,
            },
        }
    )


class CheckpointResumeTest(unittest.TestCase):
    def test_resume_preserves_lazy_adam_state_and_later_branch_activation(self):
        def update(model, optimizer, scheduler, *, activate_unused):
            optimizer.zero_grad(set_to_none=True)
            loss = model["used"](torch.ones(2, 2)).square().sum()
            if activate_unused:
                loss = loss + model["unused"](torch.ones(2, 2)).square().sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

        def build():
            model = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2)
                                         for name in ("used", "unused")})
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
            return model, optimizer, scheduler

        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalRuntime(_config(directory, name="lazy", max_steps=3))
            model, optimizer, scheduler = build()
            update(model, optimizer, scheduler, activate_unused=False)
            runtime.save_training_state(
                model=model, optimizer=optimizer, scheduler=scheduler,
                resume_dir=Path(directory) / "resume", trainer_state={"step": 1},
            )
            restored, restored_optimizer, restored_scheduler = build()
            runtime.load_training_state(
                model=restored, optimizer=restored_optimizer, scheduler=restored_scheduler,
                checkpoint_dir=directory, trusted_checkpoint=True,
            )
            self.assertNotIn(restored["unused"].weight, restored_optimizer.state)
            for activate_unused in (False, True):
                update(model, optimizer, scheduler, activate_unused=activate_unused)
                update(restored, restored_optimizer, restored_scheduler,
                       activate_unused=activate_unused)
                torch.testing.assert_close(model.state_dict(), restored.state_dict(), rtol=0, atol=0)
                torch.testing.assert_close(optimizer.state_dict(), restored_optimizer.state_dict(),
                                           rtol=0, atol=0)
            self.assertEqual(restored_optimizer.state[restored["unused"].weight]["step"], 1)
            self.assertEqual(scheduler.state_dict(), restored_scheduler.state_dict())

    def test_lazy_optimizer_resume_still_rejects_missing_model_or_moment_tensors(self):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.api import CheckpointException
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

        for missing in ("model", "moment"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                runtime = LocalRuntime(_config(directory, name="strict", max_steps=2))
                model = torch.nn.Linear(2, 2)
                optimizer = torch.optim.AdamW(model.parameters())
                model(torch.ones(1, 2)).sum().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                runtime.save_training_state(
                    model=model, optimizer=optimizer, scheduler=None,
                    resume_dir=Path(directory) / "resume", trainer_state={"step": 1},
                )
                model_state, optimizer_state = get_state_dict(
                    model, optimizer, options=StateDictOptions(cpu_offload=True))
                if missing == "model":
                    model_state.pop("bias")
                    expected = "model.bias"
                else:
                    optimizer_state["state"]["weight"].pop("exp_avg_sq")
                    expected = "optimizer.state.weight.exp_avg_sq"
                dcp.save({"model": model_state, "optimizer": optimizer_state},
                         checkpoint_id=Path(directory) / "resume/dcp", no_dist=True)
                restored = torch.nn.Linear(2, 2)
                restored_optimizer = torch.optim.AdamW(restored.parameters())
                with self.assertRaisesRegex(CheckpointException, expected):
                    runtime.load_training_state(
                        model=restored, optimizer=restored_optimizer, scheduler=None,
                        checkpoint_dir=directory, trusted_checkpoint=True,
                    )

    def test_obsolete_jax_export_path_does_not_change_resume_identity(self):
        config = _config("unused", name="resume", max_steps=2)
        expected = resume_fingerprint(config)
        config.model.jax_export_base_checkpoint = "/missing/old-base"
        self.assertEqual(resume_fingerprint(config), expected)

    def test_autocast_default_preserves_resume_but_off_changes_identity(self):
        config = _config("unused", name="resume", max_steps=2)
        original = resume_fingerprint(config)
        config.runtime.autocast = "auto"
        self.assertEqual(resume_fingerprint(config), original)
        config.runtime.autocast = "off"
        disabled = resume_fingerprint(config)
        self.assertNotEqual(disabled, original)
        config.runtime.autocast = False
        self.assertEqual(resume_fingerprint(config), disabled)

    def test_deployment_requires_full_ordinary_tensors(self):
        class ShardLikeTensor(torch.Tensor):
            pass

        _validate_state_dict({"weight": torch.ones(2)}, label="test")
        with self.assertRaisesRegex(RuntimeError, "gather a full state dict"):
            _validate_state_dict({"weight": torch.ones(2).as_subclass(ShardLikeTensor)},
                                 label="test")

    def test_dynamic_training_class_keeps_configured_artifact_target(self):
        from dobot_nexus.inference import load_policy

        dynamic = type("FSDPDummyVLAModule", (DummyVLAModule,),
                       {"__module__": "torch.distributed.fsdp"})
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory, name="dynamic", max_steps=1)
            with mock.patch("dobot_nexus.training.bc_trainer.build_model",
                            return_value=dynamic(config, mode="train")):
                result = BCTrainer(config, Experiment.from_config(config)).run()
            manifest = load_manifest(Path(result.checkpoint))
            self.assertEqual(manifest["model"]["target"], config.model.target)
            self.assertEqual(manifest["model"]["action_dim"], 2)
            config.model.artifact = result.checkpoint
            policy = load_policy(config)
            self.assertIs(type(policy.model), DummyVLAModule)
            self.assertEqual(policy.warmup().actions.shape, (1, 3, 2))

    def test_seed_and_effective_schedule_are_in_resume_fingerprint(self):
        config = _config("unused", name="test", max_steps=5)
        fingerprint = resume_fingerprint(config)
        config.exp.seed += 1
        self.assertNotEqual(resume_fingerprint(config), fingerprint)
        config.scheduler = _wrap({"name": "warmup_cosine", "warmup_steps": 2})
        fingerprint = resume_fingerprint(config)
        config.train.max_steps = 7
        self.assertNotEqual(resume_fingerprint(config), fingerprint)
        config.scheduler.total_steps = 5
        self.assertEqual(resume_fingerprint(config), fingerprint)

    def test_random_training_resumes_across_epochs_and_persistent_workers(self):
        for workers, split in ((0, 5), (2, 5), (2, 6)):
            with (
                self.subTest(workers=workers, split=split),
                tempfile.TemporaryDirectory() as directory,
            ):

                def run(name, steps, resume=None, *, workers=workers):
                    cfg = _config(
                        directory,
                        name=name,
                        max_steps=steps,
                        save_every=steps,
                        resume_from=resume,
                        target="tests.fakes:StochasticDummyVLAModule",
                    )
                    cfg.data.num_workers = workers
                    cfg.train.grad_accum_steps = 4
                    cfg.scheduler = _wrap(
                        {"name": "warmup_cosine", "warmup_steps": 2, "total_steps": 9}
                    )
                    models = []
                    from dobot_nexus.core.import_utils import build_model

                    def build(*args, **kwargs):
                        model = build_model(*args, **kwargs)
                        models.append(model)
                        return model

                    with mock.patch(
                        "dobot_nexus.training.bc_trainer.build_model", side_effect=build
                    ):
                        result = BCTrainer(cfg, Experiment.from_config(cfg)).run()
                    return result, models[0].trace

                continuous, expected = run("continuous", 9)
                _, beginning = run("resumed", split)
                resumed, ending = run("resumed", 9, "auto")
                self.assertEqual(expected, beginning + ending)
                for name in ("deployment/pytorch/full_weights.pt", "resume/trainer_state.pt"):
                    left = torch.load(Path(continuous.checkpoint) / name, weights_only=True)
                    right = torch.load(Path(resumed.checkpoint) / name, weights_only=True)
                    if name.endswith("trainer_state.pt"):
                        self.assertEqual(left["scheduler"], right["scheduler"])
                        self.assertEqual(right["scheduler"]["last_epoch"], 9)
                        self.assertNotIn("data_state", right)
                    else:
                        torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_skipped_optimizer_does_not_advance_clock(self):
        class SkippingRuntime(LocalRuntime):
            def prepare(self, model, optimizer, data_loader, scheduler):
                prepared = super().prepare(model, optimizer, data_loader, scheduler)
                original_step = optimizer.step
                self.attempts = 0
                self.skipped = False

                def step(*args, **kwargs):
                    self.attempts += 1
                    self.skipped = self.attempts == 2
                    if not self.skipped:
                        return original_step(*args, **kwargs)

                optimizer.step = step
                return prepared

            @property
            def optimizer_step_was_skipped(self):
                return self.skipped

        with tempfile.TemporaryDirectory() as directory:
            cfg = _config(directory, name="skip", max_steps=3, save_every=1)
            cfg.train.grad_accum_steps = 2
            cfg.scheduler = _wrap({"name": "warmup_cosine", "warmup_steps": 2, "total_steps": 3})
            runtime = SkippingRuntime(cfg)
            with mock.patch("dobot_nexus.training.bc_trainer.build_runtime", return_value=runtime):
                result = BCTrainer(cfg, Experiment.from_config(cfg)).run()
            self.assertEqual(runtime.attempts, 4)
            self.assertEqual(result.global_step, 3)
            state = torch.load(
                Path(result.checkpoint) / "resume/trainer_state.pt", weights_only=True
            )
            self.assertEqual(state["scheduler"]["last_epoch"], 3)
            rank = torch.load(
                Path(result.checkpoint) / "resume/rng/rank_00000.pt", weights_only=True
            )
            data = rank["data_state"]
            self.assertEqual(data["epoch"] * data["loader_length"] + data["batches_in_epoch"], 8)

    def test_incompatible_state_is_rejected_before_dcp_load(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = _config(directory, name="incompatible", max_steps=1)
            result = BCTrainer(cfg, Experiment.from_config(cfg)).run()
            path = Path(result.checkpoint) / "resume/trainer_state.pt"
            original = torch.load(path, weights_only=True)
            cfg.train.max_steps = 2
            cfg.train.resume_from = "auto"
            for mutation, message in (
                ({"training_state_version": 1}, "predates"),
                ({"scheduler": {"last_epoch": 8}}, "scheduler clock"),
                ({"world_size": 2}, "same world size"),
                ({"config_fingerprint": "wrong"}, "incompatible"),
            ):
                with self.subTest(mutation=mutation):
                    torch.save({**original, **mutation}, path)
                    with mock.patch("torch.distributed.checkpoint.load") as load:
                        with self.assertRaisesRegex(ValueError, message):
                            BCTrainer(cfg, Experiment.from_config(cfg)).run()
                        load.assert_not_called()

    def test_local_resume_matches_uninterrupted_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            continuous_config = _config(directory, name="continuous", max_steps=3, save_every=3)
            continuous = BCTrainer(
                continuous_config, Experiment.from_config(continuous_config)
            ).run()

            first_config = _config(directory, name="resumed", max_steps=1)
            BCTrainer(first_config, Experiment.from_config(first_config)).run()
            resumed_config = _config(
                directory,
                name="resumed",
                max_steps=3,
                save_every=3,
                resume_from="auto",
            )
            resumed = BCTrainer(resumed_config, Experiment.from_config(resumed_config)).run()

            continuous_state = torch.load(
                Path(continuous.checkpoint) / "deployment/pytorch/full_weights.pt",
                map_location="cpu",
                weights_only=True,
            )
            resumed_state = torch.load(
                Path(resumed.checkpoint) / "deployment/pytorch/full_weights.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(resumed.global_step, 3)
            self.assertEqual(set(continuous_state), set(resumed_state))
            for key in continuous_state:
                torch.testing.assert_close(continuous_state[key], resumed_state[key])

    def test_resume_limit_prunes_only_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory, name="retention", max_steps=3, resume_limit=2)
            result = BCTrainer(config, Experiment.from_config(config)).run()
            checkpoint_root = Path(result.checkpoint).parent
            steps = [checkpoint_root / f"step_{step:08d}" for step in (1, 2, 3)]

            self.assertFalse((steps[0] / "resume").exists())
            self.assertEqual(load_manifest(steps[0])["components"]["resume"]["status"], "pruned")
            for checkpoint in steps[1:]:
                self.assertTrue((checkpoint / "resume/dcp/.metadata").is_file())
                self.assertEqual(
                    load_manifest(checkpoint)["components"]["resume"]["status"], "ready"
                )
            for checkpoint in steps:
                self.assertTrue((checkpoint / "deployment/pytorch/full_weights.pt").is_file())
            self.assertEqual(resolve_artifact(checkpoint_root.parent), steps[-1].resolve())
            self.assertEqual(
                (checkpoint_root / "latest_resume.txt").read_text(encoding="utf-8").strip(),
                steps[-1].name,
            )

    def test_shuffled_dataloader_resumes_at_the_exact_next_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = "tests.fakes:ShuffledDummyVLAModule"
            continuous_config = _config(
                directory,
                name="shuffled_continuous",
                max_steps=6,
                save_every=6,
                target=target,
            )
            continuous = BCTrainer(
                continuous_config, Experiment.from_config(continuous_config)
            ).run()

            first_config = _config(
                directory,
                name="shuffled_resumed",
                max_steps=3,
                target=target,
            )
            BCTrainer(first_config, Experiment.from_config(first_config)).run()
            resumed_config = _config(
                directory,
                name="shuffled_resumed",
                max_steps=6,
                save_every=6,
                resume_from="auto",
                target=target,
            )
            resumed = BCTrainer(resumed_config, Experiment.from_config(resumed_config)).run()

            continuous_state = torch.load(
                Path(continuous.checkpoint) / "deployment/pytorch/full_weights.pt",
                map_location="cpu",
                weights_only=True,
            )
            resumed_state = torch.load(
                Path(resumed.checkpoint) / "deployment/pytorch/full_weights.pt",
                map_location="cpu",
                weights_only=True,
            )
            for key in continuous_state:
                torch.testing.assert_close(continuous_state[key], resumed_state[key])

    def test_resume_limit_zero_writes_deployment_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory, name="deployment_only", max_steps=1, resume_limit=0)
            result = BCTrainer(config, Experiment.from_config(config)).run()
            checkpoint = Path(result.checkpoint)
            manifest = load_manifest(checkpoint)
            self.assertFalse((checkpoint / "resume").exists())
            self.assertEqual(manifest["components"]["resume"]["status"], "disabled")
            self.assertTrue((checkpoint / "deployment/pytorch/full_weights.pt").is_file())
            self.assertFalse((checkpoint.parent / "latest_resume.txt").exists())

    def test_failed_deployment_does_not_prune_previous_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory, name="failed_export", max_steps=3, resume_limit=1)
            calls = 0

            def export(_model, *, source_weights, artifact_dir):
                nonlocal calls
                del source_weights, artifact_dir
                calls += 1
                if calls == 3:
                    raise RuntimeError("injected deployment failure")
                return {}

            with mock.patch.object(DummyVLAModule, "export_deployment_artifacts", export):
                with self.assertRaisesRegex(RuntimeError, "injected deployment failure"):
                    BCTrainer(config, Experiment.from_config(config)).run()

            checkpoint_root = Path(directory) / "failed_export" / "checkpoints"
            previous = checkpoint_root / "step_00000002"
            self.assertTrue((previous / "resume/dcp/.metadata").is_file())
            self.assertEqual(
                (checkpoint_root / "latest_resume.txt").read_text(encoding="utf-8").strip(),
                previous.name,
            )
            self.assertFalse((checkpoint_root / "step_00000003").exists())


class EMATest(unittest.TestCase):
    def test_update_export_context_and_restore(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2)
        ema = ParameterEMA(model, 0.5)
        with torch.no_grad():
            model.weight.fill_(4)
        ema.update()
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with ema.apply():
                self.assertEqual(model.weight.item(), 3)
                raise RuntimeError("fixture")
        self.assertEqual(model.weight.item(), 4)
        with tempfile.TemporaryDirectory() as directory:
            ema.save(Path(directory) / "resume", 0)
            restored = ParameterEMA(model, 0.5)
            restored.load(directory, 0)
            self.assertEqual(restored.updates, 1)
            with restored.apply():
                self.assertEqual(model.weight.item(), 3)

    def test_trainer_ema_resume_matches_continuous_and_exports_averaged_weights(self):
        with tempfile.TemporaryDirectory() as directory:

            def run(name, steps, resume=None):
                cfg = _config(directory, name=name, max_steps=steps, resume_from=resume)
                cfg.train.ema_decay = 0.9
                return BCTrainer(cfg, Experiment.from_config(cfg)).run()

            continuous = run("continuous", 3)
            initial = run("resumed", 1)
            resumed = run("resumed", 3, "auto")
            for relative in ("deployment/pytorch/full_weights.pt", "resume/ema/rank_00000.pt"):
                a = torch.load(Path(continuous.checkpoint) / relative, weights_only=True)
                b = torch.load(Path(resumed.checkpoint) / relative, weights_only=True)
                if "ema/" in relative:
                    self.assertEqual(a["updates"], 3)
                    a, b = a["shadow"], b["shadow"]
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            deployed = torch.load(
                Path(continuous.checkpoint) / "deployment/pytorch/full_weights.pt",
                weights_only=True,
            )
            for key, value in a.items():
                torch.testing.assert_close(deployed[key], value, rtol=0, atol=0)
            manifest = load_manifest(initial.checkpoint)
            self.assertEqual(manifest["training"]["deployment_weights"], "ema")
            cfg = _config(directory, name="test", max_steps=3)
            before = resume_fingerprint(cfg)
            cfg.train.ema_decay = 0.9
            self.assertNotEqual(before, resume_fingerprint(cfg))

    def test_ema_updates_only_after_successful_accumulated_optimizer_steps(self):
        class SkippingRuntime(LocalRuntime):
            def prepare(self, model, optimizer, data_loader, scheduler):
                prepared = super().prepare(model, optimizer, data_loader, scheduler)
                original_step = optimizer.step
                self.attempts, self.skipped = 0, False
                self.parameters_after_update = []

                def step(*args, **kwargs):
                    self.attempts += 1
                    self.skipped = self.attempts == 2
                    if not self.skipped:
                        result = original_step(*args, **kwargs)
                        self.parameters_after_update.append(float(model.scale.detach()))
                        return result

                optimizer.step = step
                return prepared

            @property
            def optimizer_step_was_skipped(self):
                return self.skipped

        with tempfile.TemporaryDirectory() as directory:
            cfg = _config(directory, name="skipped", max_steps=3)
            cfg.train.ema_decay, cfg.train.grad_accum_steps = 0.9, 2
            runtime = SkippingRuntime(cfg)
            with mock.patch("dobot_nexus.training.bc_trainer.build_runtime", return_value=runtime):
                result = BCTrainer(cfg, Experiment.from_config(cfg)).run()
            self.assertEqual(runtime.attempts, 4)
            state = torch.load(
                Path(result.checkpoint) / "resume/ema/rank_00000.pt", weights_only=True
            )
            self.assertEqual(state["updates"], 3)
            values = runtime.parameters_after_update
            expected = 0.5 * 0.9**3 + 0.1 * (values[0] * 0.9**2 + values[1] * 0.9 + values[2])
            self.assertAlmostEqual(state["shadow"]["scale"].item(), expected, places=6)
            self.assertNotAlmostEqual(expected, values[-1], places=4)


if __name__ == "__main__":
    unittest.main()
