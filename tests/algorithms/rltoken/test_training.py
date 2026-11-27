from __future__ import annotations

import ast
import errno
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dobot_nexus.algorithms.rltoken.artifacts import (
    feature_metadata,
    file_digest,
    load_provider,
    retain_feature_bundle,
    write_model_bundle,
)
from dobot_nexus.algorithms.rltoken.configuration import HeadConfig, LearningConfig
from dobot_nexus.algorithms.rltoken.identity import content_digest
from dobot_nexus.algorithms.rltoken.learner import create_learner
from dobot_nexus.algorithms.rltoken.policy import RLTokenPolicy
from dobot_nexus.algorithms.rltoken.pretraining import save_prewarm
from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.roles import EnvRole, LearnerRole, RolloutRole
from dobot_nexus.algorithms.rltoken.training import RLTokenTrainer, validate_training
from dobot_nexus.algorithms.rltoken.transitions import Finalization
from dobot_nexus.core.config import _wrap, load_config
from dobot_nexus.core.contracts import SamplingOptions
from dobot_nexus.inference.policy import InferencePolicy
from dobot_nexus.runtime.local import LocalRuntime
from dobot_nexus.runtime.sync import SingleInFlightSyncCoordinator
from dobot_nexus.training.resume import capture_rng_state

from .fakes import observation, transition
from .test_env_rollout import ConfirmingAgent
from .test_roles import online_config

ROOT = Path(__file__).resolve().parents[3]


class EpisodeAgent(ConfirmingAgent):
    def __init__(self, config):
        super().__init__()

    def execute(self, command_id, commands, *, timeout):
        feedback = super().execute(command_id, commands, timeout=timeout)
        return replace(feedback, terminated=self.executions == 2)

    def finalize_episode(self, *, pending, **kwargs):
        return [Finalization(row.key, torch.ones(2), True, True) for row in pending]


class TrainingTest(unittest.TestCase):
    def test_all_templates_load_and_online_recipes_validate(self):
        from dobot_nexus.algorithms.rltoken.pretraining import RLTokenPrewarmTrainer
        from dobot_nexus.algorithms.rltoken.token import TokenConfig

        for path in sorted((ROOT / "examples/configs/rltoken").glob("*.yaml")):
            with self.subTest(path=path.name):
                config = load_config(path)
                if config.runtime.get("launcher") == "rlinf":
                    validate_training(config)
                elif path.stem == "td3_prewarm":
                    RLTokenPrewarmTrainer(config, None)
                elif path.stem == "td3_token":
                    self.assertEqual(
                        TokenConfig(**config.model.rltoken.token).mode, "position_bottleneck"
                    )
        config = load_config(ROOT / "examples/configs/rltoken/td3_online.yaml")
        for scale in (1.0, [1.0] * 14, [1.0] * 84):
            config.model.rltoken.residual.scale = scale
            validate_training(config)
        config.model.rltoken.residual.scale = [1.0, 2.0]
        with self.assertRaisesRegex(ValueError, "scale"):
            validate_training(config)

    def test_terminal_recipe_requires_an_explicit_known_name(self):
        from dobot_nexus.algorithms.rltoken.configuration import recipe_identity

        config = load_config(ROOT / "examples/configs/rltoken/stage2_simulation.yaml")
        config.algorithm.recipe = "rlt_ac_terminal_v2"
        validate_training(config)
        self.assertNotEqual(
            recipe_identity(config.algorithm.recipe), recipe_identity("rlt_ac_fixed_v1")
        )
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

    def online(self, td3=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.experiment = SimpleNamespace(run_dir=self.root, logger=logging.getLogger("test"))
        return online_config(self.root, td3=td3)

    def trainer(self, config):
        return RLTokenTrainer(config, self.experiment, runtime_factory=LocalRuntime)

    def test_non_residual_resume_binds_explicit_reference_steps(self):
        cfg = self.online()
        cfg.environment.target = "tests.algorithms.rltoken.test_roles:scalar_agent"
        cfg.train.max_steps = 1
        self.assertNotIn("reference_num_steps", cfg.algorithm)
        self.assertEqual(self.trainer(cfg).task_fingerprint,
                         "dfd1e486ee2169622605bfa790415da4e2a95551bc0d819f6751b48f89bfd93f")

        cfg.algorithm.reference_num_steps = 4
        trainer = self.trainer(cfg)
        trainer.run()
        cfg.train.resume_from = "auto"
        self.assertEqual(self.trainer(cfg).task_fingerprint, trainer.task_fingerprint)
        cfg.algorithm.reference_num_steps = 9
        with self.assertRaisesRegex(ValueError, "Resume task/reward fingerprint mismatch"):
            self.trainer(cfg)

    def test_workers_use_pinned_source_after_latest_moves(self):
        config = self.online()
        config.train.max_steps = 1
        source = config.model.features.artifact
        provider, provider_cfg, spec = load_provider(source)
        with torch.no_grad():
            provider.scale.add_(1)
        other = write_model_bundle(self.root / "other", provider, provider_cfg)
        self.assertNotEqual(spec, feature_metadata(other)[1])
        pointer = self.root / "latest.txt"
        pointer.write_text(source)
        config.model.features.artifact = str(pointer)
        trainer = self.trainer(config)
        pointer.write_text(str(other))
        self.assertEqual(trainer.run().global_step, 1)
        self.assertEqual(feature_metadata(self.root / "rltoken/features")[1], spec)

    def test_worker_rejects_changed_source_manifest_at_pinned_root(self):
        config = self.online()
        trainer = self.trainer(config)
        manifest = Path(config.model.features.artifact) / "manifest.yaml"
        manifest.write_text(manifest.read_text() + "\n# changed after selection\n")
        with self.assertRaisesRegex(ValueError, "source manifest checksum"):
            trainer.run()

    def historical_run(self):
        """Write the historical deep v1 state, journal and writable provider bundle directly."""
        cfg = self.online()
        cfg.train.max_steps = 1
        original = self.trainer(cfg)
        run = self.root / "rltoken"
        provider, config, spec = load_provider(cfg.model.features.artifact)
        retained = write_model_bundle(run / "features", provider, config)
        for file in retained.rglob("*"):
            if file.is_file():
                file.chmod(0o644)
        cfg.model.features.artifact = str(retained)
        role = LearnerRole(cfg, original.run_id, original.epoch, original.task_fingerprint, None)
        role.initialize(SimpleNamespace(channels=None, device="cpu"))
        legacy = run / "checkpoints/step-000000001-0123abcd"
        legacy.mkdir(parents=True)
        try:
            learner = role.learner
            learner.update(1, incoming=[transition(spec=spec)])
            state = learner.state_dict()
            state.update(
                rng=capture_rng_state(),
                task_fingerprint=original.task_fingerprint,
                run_id=original.run_id,
            )
            torch.save(state, legacy / "state.pt")
            connection = sqlite3.connect(legacy / "journal.sqlite")
            try:
                learner.replay.connection.backup(connection)
            finally:
                connection.close()
            manifest = dict(
                format="rltoken_resume_v1",
                state_digest=content_digest(state),
                run_id=original.run_id,
                feature_manifest="../../features/manifest.yaml",
                watermark=learner.replay.watermark,
                task_fingerprint=original.task_fingerprint,
                critic_updates=learner.critic_updates,
                actor_updates=learner.actor_updates,
            )
            (legacy / "manifest.json").write_text(json.dumps(manifest))
        finally:
            role.close()
        (run / "journal.sqlite").unlink()
        (legacy.parent / "latest_resume").write_text(legacy.name)
        self.assertNotIn("retained_files_sha256", (retained / "manifest.yaml").read_text())
        self.assertFalse(retained.with_suffix(".sha256").exists())
        policy = RLTokenPolicy(cfg, mode="infer")
        policy.actor.load_state_dict(state["actor"])
        options = SamplingOptions(seed=42, deterministic=True)
        actions = policy.predict_action_chunk(observation(), options).actions
        cfg.train.trust_resume_checkpoint = True
        cfg.train.resume_from = "auto"
        return cfg, retained, actions

    def check_legacy_upgrade(self, interrupt_at=None):
        cfg, retained, expected_actions = self.historical_run()
        seal = retained.with_suffix(".sha256")
        manifest = retained / "manifest.yaml"
        old_digest = file_digest(manifest)
        if interrupt_at:
            replace_file = os.replace

            def interrupt(src, dst):
                if interrupt_at == "A" and dst == manifest:
                    raise OSError(errno.ENOSPC, "before manifest replacement")
                if interrupt_at == "B" and dst == seal and len(src.read_text().splitlines()) == 1:
                    raise OSError(errno.ENOSPC, "before seal narrowing")
                return replace_file(src, dst)

            with patch("os.replace", interrupt), self.assertRaises(OSError) as failure:
                retain_feature_bundle(retained, retained)
            self.assertEqual(failure.exception.errno, errno.ENOSPC)
            new_digest, prior = seal.read_text().splitlines()
            self.assertEqual(seal.stat().st_mode & 0o777, 0o400)
            self.assertEqual(prior, old_digest)
            self.assertNotEqual(new_digest, prior)
            self.assertEqual(file_digest(manifest), prior if interrupt_at == "A" else new_digest)
            RLTokenPolicy(cfg, mode="train")  # A learner can initialize at either boundary.
            retain_feature_bundle(retained, retained)
            self.assertEqual(seal.read_text(), new_digest + "\n")

        result = self.trainer(cfg).run()
        self.assertEqual(result.global_step, 1)
        final = file_digest(manifest)
        self.assertEqual(seal.read_text(), final + "\n")
        self.assertEqual(seal.stat().st_mode & 0o777, 0o400)
        self.assertNotEqual(old_digest, final)
        self.assertIn("retained_files_sha256", manifest.read_text())
        saved = json.loads((Path(result.checkpoint) / "manifest.json").read_text())
        self.assertEqual(saved["feature_manifest_sha256"], final)
        deployed = retained.parent / (retained.parent / "latest_deployment.txt").read_text()
        for file in retained.rglob("*"):
            if file.is_file():
                self.assertEqual(file.stat().st_mode & 0o222, 0)
                self.assertTrue(
                    file.samefile(deployed / "deployment/features" / file.relative_to(retained))
                )
        shutil.rmtree(retained)
        infer = _wrap(
            dict(
                schema_version=1,
                task="infer",
                model=dict(artifact=str(deployed)),
                runtime=dict(device="cpu"),
            )
        )
        actual = InferencePolicy(infer).predict_chunk(
            observation(), SamplingOptions(seed=42, deterministic=True)
        )
        self.assertTrue((actual.actions == expected_actions).all())

    def test_historical_checkpoint_and_writable_features_upgrade_through_local_runtime(self):
        self.check_legacy_upgrade()

    def test_legacy_upgrade_retries_after_transition_seal_publication(self):
        self.check_legacy_upgrade("A")

    def test_legacy_upgrade_retries_after_manifest_replacement(self):
        self.check_legacy_upgrade("B")

    def test_v1_update_limit_snapshots_after_last_durable_ack_and_auto_resume(self):
        config = self.online()
        config.environment.min_actor_updates = 256
        completed, barriers = [], []
        collect, snapshot = EnvRole.collect_chunk, LearnerRole.snapshot

        def collect_chunk(role, **kwargs):
            result = collect(role, **kwargs)
            time.sleep(0.04)  # The learner can hit a save boundary while this RPC is in flight.
            completed.append(result["chunks"])
            return result

        def snapshot_without_rng(role, *, feature_manifest_sha256):
            self.assertEqual(role.learner.replay.admitted, len(completed))
            barriers.append(role.learner.critic_updates)
            self.assertEqual(len(feature_manifest_sha256), 64)
            return snapshot(role, feature_manifest_sha256=feature_manifest_sha256)

        trainer = self.trainer(config)
        with (
            patch.object(EnvRole, "collect_chunk", collect_chunk),
            patch.object(LearnerRole, "snapshot", snapshot_without_rng),
        ):
            result = trainer.run()
        self.assertEqual(result.global_step, 3)
        self.assertEqual(barriers, [2, 3])
        self.assertEqual(result.metrics["deployment_ready"], 0)
        manifest = json.loads((Path(result.checkpoint) / "manifest.json").read_text())
        self.assertEqual(manifest["format"], "rltoken_resume_v1")
        config.train.resume_from = "auto"
        config.train.max_steps = 4
        shutil.rmtree(config.model.features.artifact)
        resumed = self.trainer(config)
        self.assertEqual(resumed.task_fingerprint, trainer.task_fingerprint)
        self.assertEqual(resumed.run_id, trainer.run_id)
        self.assertNotEqual(resumed.epoch, trainer.epoch)
        self.assertEqual(resumed.run().global_step, 4)

    def test_online_retention_across_resume_keeps_both_artifacts_loadable(self):
        from dobot_nexus.core.config import _wrap
        from dobot_nexus.inference.policy import InferencePolicy

        from .fakes import observation

        config = self.online()
        config.train.update(max_steps=1, save_every=1)
        trainer = self.trainer(config)
        trainer.run()
        config.train.update(resume_from="auto", max_steps=4, keep_last_checkpoints=2,
                            keep_last_deployments=2)
        resumed = self.trainer(config)
        self.assertEqual(resumed.task_fingerprint, trainer.task_fingerprint)
        resumed.run()
        root = self.root / "rltoken"
        checkpoints = list((root / "checkpoints").glob("step-*"))
        deployments = list((root / "deployments").glob("actor-*"))
        self.assertEqual((len(checkpoints), len(deployments)), (2, 2))
        self.assertIn(root / (root / "latest_deployment.txt").read_text(), deployments)
        active = (root / "checkpoints/latest_resume").read_text()
        self.assertIn(active, [p.name for p in checkpoints])
        for path in checkpoints:
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertTrue((path / manifest["feature_manifest"]).is_file())
            role = LearnerRole(
                config, resumed.run_id, resumed.epoch, resumed.task_fingerprint, path
            )
            try:
                status = role.initialize(SimpleNamespace(device="cpu", channels=None))
                self.assertEqual(status["critic_updates"], manifest["critic_updates"])
            finally:
                role.close()
        for path in deployments:
            infer = _wrap(dict(model=dict(artifact=str(path)), runtime=dict(device="cpu")))
            self.assertEqual(InferencePolicy(infer).predict_chunk(observation()).actions.shape,
                             (1, 2, 2))

    def test_pretrained_actor_starts_immediately_and_drains_episode_budget(self):
        config = self.online(td3=True)
        config.environment.update(
            target=f"{__name__}:EpisodeAgent",
            reward_mode="episode",
            update_schedule="episode_end",
            max_chunks=2,
            drain_on_stop=True,
        )
        config.train.update(max_steps=10, save_every=10)
        config.algorithm.learning.updates_per_new_transition = 2
        spec = feature_metadata(config.model.features.artifact)[1]
        replay = Replay(self.root / "offline.sqlite", spec, 2, updates_per_new_transition=2)
        self.addCleanup(replay.close)
        offline = create_learner(
            spec,
            HeadConfig(**config.model.rltoken),
            LearningConfig(**config.algorithm.learning),
            replay,
        )
        for _ in range(2):
            offline.train_rows([transition(spec=spec, episode_success=True)])
        config.algorithm.initialize_from = str(self.root / "prewarm.pt")
        save_prewarm(
            config.algorithm.initialize_from,
            offline,
            task_identity=config.algorithm.task_identity,
            cache_id="test",
            feature_manifest="manifest.yaml",
        )

        from dobot_nexus.algorithms.rltoken.serialization import load_payload
        from dobot_nexus.core.config import clone_config

        torch.save(load_payload(config.algorithm.initialize_from), config.algorithm.initialize_from)
        config.train.trust_resume_checkpoint = True
        isolated = clone_config(config)
        isolated.exp.run_dir = str(self.root / "untrusted-init")
        role = LearnerRole(isolated, "run", "epoch", "task", None)
        try:
            with self.assertRaisesRegex(ValueError, "trusted_checkpoint"):
                role.initialize(SimpleNamespace(channels=None, device="cpu"))
        finally:
            role.close()
        config.train.trust_legacy_prewarm = True
        ready, updates = [], []
        collect, update = EnvRole.collect_chunk, LearnerRole.update

        def collect_chunk(role, actor_ready=False):
            ready.append(actor_ready)
            return collect(role, actor_ready=actor_ready)

        def update_learner(role, max_updates):
            status = update(role, max_updates)
            if status["updates"]:
                updates.append(len(ready))
            return status

        with (
            patch.object(EnvRole, "collect_chunk", collect_chunk),
            patch.object(LearnerRole, "update", update_learner),
        ):
            result = self.trainer(config).run()
        self.assertEqual(ready, [True, True])
        self.assertEqual(updates, [2] * 4)
        self.assertEqual(result.global_step, 4)
        self.assertEqual(result.metrics["budget"], 0)
        self.assertEqual(result.metrics["deployment_ready"], 1)
        root = self.root / "rltoken"
        self.assertTrue(
            (root / (root / "latest_deployment.txt").read_text() / "manifest.yaml").is_file()
        )

    def test_td3_resume_restores_rollout_rng_and_pending_before_collection(self):
        config = self.online(td3=True)
        config.environment.update(
            target=f"{__name__}:EpisodeAgent",
            reward_mode="episode",
            max_chunks=1,
        )
        trainer = self.trainer(config)
        first = trainer.run()
        checkpoint = Path(first.checkpoint)
        state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
        self.assertEqual(
            json.loads((checkpoint / "manifest.json").read_text())["format"],
            "rltoken_td3_resume_v3",
        )
        self.assertIsInstance(state["rollout_rng"], torch.Tensor)
        config.train.resume_from = "auto"
        sequence = []
        restore_rng, restore_pending = RolloutRole.restore_rng, EnvRole.restore_pending
        synchronize, collect = LearnerRole.sync_model_to_rollout, EnvRole.collect_chunk

        def rng(role, saved):
            torch.testing.assert_close(saved, state["rollout_rng"])
            sequence.append("rng")
            return restore_rng(role, saved)

        def pending(role, rows):
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].observation_id.session_epoch, trainer.epoch)
            sequence.append("pending")
            return restore_pending(role, rows)

        def sync(role):
            version = synchronize(role)
            sequence.append("sync")
            return version

        def collect_chunk(role, **kwargs):
            sequence.append("collect")
            return collect(role, **kwargs)

        with (
            patch.object(RolloutRole, "restore_rng", rng),
            patch.object(EnvRole, "restore_pending", pending),
            patch.object(LearnerRole, "sync_model_to_rollout", sync),
            patch.object(EnvRole, "collect_chunk", collect_chunk),
        ):
            self.trainer(config).run()
        self.assertEqual(sequence[:4], ["rng", "pending", "sync", "collect"])

    def test_stale_epoch_on_background_sync_permanently_blocks_deployment(self):
        config = self.online()
        trainer = self.trainer(config)
        sync = SingleInFlightSyncCoordinator()
        receive = RolloutRole.request_actor_sync_model
        seen = []

        def stale_reply(role):
            applied = receive(role)
            seen.append(applied)
            # First call is the successful blocking startup handshake.
            return replace(applied, learner_epoch="stale") if len(seen) == 2 else applied

        with (
            patch(
                "dobot_nexus.algorithms.rltoken.training.SingleInFlightSyncCoordinator",
                return_value=sync,
            ),
            patch.object(RolloutRole, "request_actor_sync_model", stale_reply),
            patch.object(RolloutRole, "export_deployment") as export,
        ):
            with self.assertRaisesRegex(ValueError, "Synchronized Actor version mismatch"):
                trainer.run()
            export.assert_not_called()
        self.assertEqual(len(seen), 2)
        self.assertEqual(sync.last_result, (seen[0], seen[0]))
        self.assertTrue(sync.failed)
        with self.assertRaisesRegex(RuntimeError, "previous synchronization failed"):
            sync.sync_blocking(self.fail, self.fail)
        self.assertFalse((self.root / "rltoken/latest_deployment.txt").exists())

    def test_mismatched_export_or_failed_final_sync_preserves_published_pointer(self):
        config = self.online()
        config.train.max_steps = 1
        self.trainer(config).run()
        pointer = self.root / "rltoken/latest_deployment.txt"
        original = pointer.read_bytes()
        config.train.resume_from = "auto"
        export = RolloutRole.export_deployment
        for field, bad in (("run_id", "foreign"), ("learner_epoch", "stale"), ("actor_update", 0)):

            def wrong_version(role, path, key=field, wrong=bad):
                return {**export(role, path), key: wrong}

            with (
                self.subTest(field=field),
                patch.object(RolloutRole, "export_deployment", wrong_version),
            ):
                with self.assertRaisesRegex(ValueError, "final Actor version"):
                    self.trainer(config).run()
                self.assertEqual(pointer.read_bytes(), original)

        sync = LearnerRole.sync_model_to_rollout
        calls = []

        def fail_final_send(role):
            ack = sync(role)
            calls.append(ack)
            if len(calls) == 2:
                raise TimeoutError("lost applied ACK")
            return ack

        with patch.object(LearnerRole, "sync_model_to_rollout", fail_final_send):
            with self.assertRaisesRegex(TimeoutError, "lost applied ACK"):
                self.trainer(config).run()
        self.assertEqual(pointer.read_bytes(), original)

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
                    self.assertNotIn(name.split(".")[0], {"rlinf", "ray"}, str(path))
