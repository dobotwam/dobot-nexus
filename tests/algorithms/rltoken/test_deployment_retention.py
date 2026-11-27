import errno
import json
import logging
import os
import shutil
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from dobot_nexus.algorithms.rltoken.artifacts import (
    _link_feature_bundle,
    _verify_retained_feature_bundle,
    feature_metadata,
    file_digest,
    load_provider,
    restore_policy_config,
    retain_feature_bundle,
    write_model_bundle,
)
from dobot_nexus.algorithms.rltoken.checkpoint import (
    _publication_lock,
    _sync_dir,
    prune_deployments,
)
from dobot_nexus.algorithms.rltoken.policy import RLTokenPolicy
from dobot_nexus.algorithms.rltoken.pretraining import RLTokenCacheTrainer, RLTokenPrewarmTrainer
from dobot_nexus.algorithms.rltoken.roles import LearnerRole, RolloutRole
from dobot_nexus.algorithms.rltoken.training import RLTokenTrainer
from dobot_nexus.algorithms.rltoken.weight_sync import ActorSnapshot, ActorVersion
from dobot_nexus.core.artifacts import (
    get_verified_artifact_snapshot,
    load_manifest,
    resolve_artifact,
    verified_artifact_snapshot,
)
from dobot_nexus.core.config import _wrap, clone_config, load_config, load_config_bytes, save_config
from dobot_nexus.core.contracts import SamplingOptions
from dobot_nexus.inference.policy import InferencePolicy
from dobot_nexus.runtime.local import LocalRuntime

from . import test_policy
from .fakes import observation
from .test_roles import online_config


class DeploymentTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.cfg = online_config(self.root)
        factory = test_policy.PolicyTest()
        factory.root = self.root
        self.source = Path(factory.make_config().model.features.artifact)
        self.cfg.model.features.artifact = str(self.source)
        experiment = SimpleNamespace(run_dir=self.root, logger=logging.getLogger(__name__))
        self.trainer = RLTokenTrainer(self.cfg, experiment)
        self.role = RolloutRole(self.cfg, self.trainer.run_id, self.trainer.epoch)
        self.role.initialize(SimpleNamespace(channels=None, device="cpu"))
        self.policy = self.role.engine.policy
        self.version = ActorVersion(self.trainer.run_id, self.trainer.epoch, 1)
        self.role.engine.weights.apply(
            ActorSnapshot.capture(
                self.policy.actor, self.version, self.policy.feature_spec.feature_id
            )
        )
        self.retained = self.root / "rltoken/features"
        self.pointer = self.retained.parent / "latest_deployment.txt"
        self.history = self.retained.parent / "deployments/.published.json"
        self.options = SamplingOptions(seed=42, deterministic=True)

    def publish(self):
        # Only the transport is synchronous here. Both the rollout export and the
        # coordinator's version check, pointer replacement and pruning run unmodified.
        def call(method, path):
            return SimpleNamespace(wait=lambda: getattr(self.role, method)(path))

        result = self.trainer._publish_deployment(SimpleNamespace(call=call), self.version)
        self.assertEqual(result, self.pointer.parent / self.pointer.read_text())
        return result

    def actions(self, path):
        cfg = _wrap(
            dict(
                schema_version=1,
                task="infer",
                model=dict(artifact=str(path)),
                runtime=dict(device="cpu"),
            )
        )
        return InferencePolicy(cfg).predict_chunk(observation(), self.options).actions

    def test_actor_only_live_export_and_independent_hardlinked_deployments(self):
        expected = self.policy.predict_action_chunk(observation(), self.options).actions
        # A full outer state_dict or a clone of any provider tensor would regress peak memory.
        provider_storage = {
            t.untyped_storage().data_ptr() for t in self.policy.provider.parameters()
        }
        provider_storage.update(
            t.untyped_storage().data_ptr() for t in self.policy.provider.buffers()
        )
        original_clone = torch.Tensor.clone

        def clone(tensor, *args, **kwargs):
            self.assertNotIn(tensor.untyped_storage().data_ptr(), provider_storage)
            return original_clone(tensor, *args, **kwargs)

        self.policy.provider.register_buffer("scratch", torch.ones(2), persistent=False)
        with (
            patch.object(
                self.policy.provider, "state_dict", side_effect=AssertionError("provider")
            ),
            patch.object(self.policy, "state_dict", side_effect=AssertionError("full export")),
            patch.object(torch.Tensor, "clone", clone),
        ):
            first = self.publish()
            second = self.publish()
        self.assertNotEqual(first, second)
        self.assertFalse(
            os.path.samefile(
                first / "deployment/pytorch/full_weights.pt",
                second / "deployment/pytorch/full_weights.pt",
            )
        )
        state = torch.load(second / "deployment/pytorch/full_weights.pt", weights_only=True)
        self.assertEqual(set(state), set(self.policy.actor.state_dict()))
        for key, value in self.policy.actor.state_dict().items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
        for src in self.retained.rglob("*"):
            if src.is_file():
                self.assertFalse(src.stat().st_mode & 0o222)
                for bundle in (first, second):
                    dst = bundle / "deployment/features" / src.relative_to(self.retained)
                    self.assertTrue(os.path.samefile(src, dst))
        shutil.rmtree(self.source)
        (self.root / "asset-2.bin").unlink()
        shutil.rmtree(first)
        shutil.rmtree(self.retained)
        moved = second.rename(self.root / "moved")
        self.assertTrue((self.actions(moved) == expected).all())

    def test_link_capability_errors_copy_and_other_errors_abort_before_pointer(self):
        for number in {errno.EXDEV, errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP}:
            with self.subTest(number=number), patch("os.link", side_effect=OSError(number, "link")):
                deployed = self.publish()
                for src in self.retained.rglob("*"):
                    if src.is_file():
                        dst = deployed / "deployment/features" / src.relative_to(self.retained)
                        self.assertFalse(os.path.samefile(src, dst))
                        self.assertEqual(dst.read_bytes(), src.read_bytes())
                        self.assertEqual(
                            stat.S_IMODE(src.stat().st_mode), stat.S_IMODE(dst.stat().st_mode)
                        )
        self.cfg.train.keep_last_deployments = 1
        before = set(deployed.parent.iterdir())
        pointer = self.pointer.read_bytes()
        for number in (errno.ENOSPC, errno.EEXIST, errno.EIO):
            with self.subTest(number=number), patch("os.link", side_effect=OSError(number, "link")):
                with self.assertRaises(OSError) as error:
                    self.publish()
                self.assertEqual(error.exception.errno, number)
                self.assertEqual(set(deployed.parent.iterdir()), before)
                self.assertEqual(self.pointer.read_bytes(), pointer)

    def test_live_parameter_persistent_buffer_and_spec_drift_abort_publication(self):
        def load_with_buffers(root, **kwargs):
            provider, config, spec = load_provider(root, **kwargs)
            provider.register_buffer("strided", torch.arange(4.0)[::2][:1])
            provider.register_buffer("empty", torch.empty(0))
            return provider, config, spec

        with patch("dobot_nexus.algorithms.rltoken.policy.load_provider", load_with_buffers):
            self.policy = RLTokenPolicy(self.cfg, mode="infer", retained_feature_root=self.retained)
        self.role.engine.policy = self.policy
        active = self.publish()
        self.cfg.train.keep_last_deployments = 1
        expected = self.actions(active)
        for tensor in (
            self.policy.provider.scale,
            self.policy.provider.running,
            self.policy.provider.strided,
        ):
            saved = tensor.detach().clone()
            tensor.data.add_(1)  # Does not bump the original tensor's Torch version counter.
            with self.assertRaisesRegex(RuntimeError, "drift"):
                self.publish()
            tensor.data.copy_(saved)
        self.policy.provider.empty.resize_(1).zero_()
        with self.assertRaisesRegex(RuntimeError, "drift"):
            self.publish()
        self.policy.provider.empty.resize_(0)
        self.policy.feature_spec = replace(self.policy.feature_spec, feature_id="changed")
        with self.assertRaisesRegex(RuntimeError, "drift"):
            self.publish()
        self.assertEqual(self.pointer.parent / self.pointer.read_text(), active)
        self.assertTrue((self.actions(active) == expected).all())

    def test_retained_file_checksums_and_unlisted_symlinks_abort_publication(self):
        old = self.publish()
        expected = self.actions(old)
        files = load_manifest(self.retained)["files"]
        for name in (
            files["weights"],
            files["model_assets"]["model.asset"],
            files["nexus_weights"],
            files["resolved_config"],
        ):
            path = self.retained / name
            original = path.read_bytes()
            replacement = path.with_suffix(".new")
            replacement.write_bytes(original + b"changed")
            replacement.replace(path)
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.publish()
            with self.assertRaisesRegex(ValueError, "checksum"):
                retain_feature_bundle(self.source, self.retained)
            with self.assertRaisesRegex(ValueError, "checksum"):
                retain_feature_bundle(self.retained, self.root / "fresh")
            replacement.write_bytes(original)
            replacement.chmod(0o444)
            replacement.replace(path)

        symlink = self.retained / "unlisted"
        symlink.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.publish()
        self.assertEqual(self.pointer.parent / self.pointer.read_text(), old)
        self.assertTrue((self.actions(old) == expected).all())

    def test_replaced_source_between_verification_and_link_aborts(self):
        active = self.publish()
        link = os.link

        def swap_source(src, dst, **kwargs):
            replacement = src.with_name(".replacement")
            replacement.write_bytes(b"changed after verification")
            replacement.replace(src)
            return link(src, dst, **kwargs)

        with patch("os.link", swap_source), self.assertRaisesRegex(ValueError, "changed"):
            self.publish()
        self.assertEqual(self.pointer.parent / self.pointer.read_text(), active)
        self.assertEqual(self.actions(active).shape, (1, 2, 2))

    def test_retention_only_strips_write_bits_and_rejects_configured_redirection(self):
        # Sealing honors the regular file's existing mode, including executable bits.
        asset = self.source / load_manifest(self.source)["files"]["model_assets"]["model.asset"]
        asset.chmod(0o710)
        fresh = self.root / "another-retained"
        retain_feature_bundle(self.source, fresh)
        self.assertEqual(
            stat.S_IMODE((fresh / "deployment/assets/model_asset.bin").stat().st_mode), 0o510
        )
        alias = self.root / "linked-source"
        alias.symlink_to(fresh, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "redirect"):
            retain_feature_bundle(alias, fresh)
        (fresh / "latest.txt").write_text(str(self.source))
        self.cfg.model.features.artifact = str(fresh)
        with self.assertRaisesRegex(ValueError, "redirect"):
            RLTokenPolicy(self.cfg, mode="infer")

    def test_manifest_paths_and_redirects_are_confined(self):
        manifest = load_manifest(self.retained)
        spec = feature_metadata(self.retained)[1]
        manifest_file = self.retained / "manifest.yaml"
        manifest_file.chmod(0o600)
        for field in ("weights", "resolved_config", "model_assets", "nexus_weights"):
            for value in ("../features/manifest.yaml", str(manifest_file), "deployment", "absent"):
                with self.subTest(field=field, value=value):
                    bad = dict(manifest["files"])
                    bad[field] = {"model.asset": value} if field == "model_assets" else value
                    save_config({**manifest, "files": bad}, manifest_file)
                    with self.assertRaisesRegex(ValueError, "path"):
                        _verify_retained_feature_bundle(self.retained, spec)
        save_config(manifest, manifest_file)
        (self.retained / "latest.txt").write_text(str(self.source))
        with self.assertRaisesRegex(ValueError, "redirect"):
            _verify_retained_feature_bundle(self.retained, spec)
        (self.retained / "latest.txt").unlink()
        manifest_file.chmod(0o444)
        deployed = self.publish()
        directory = deployed / "deployment"
        directory.rename(self.root / "outside")
        directory.symlink_to(self.root / "outside", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.actions(deployed)

    def test_failed_pointer_replace_preserves_the_previous_runner_up(self):
        first = self.publish()
        second = self.publish()
        self.cfg.train.keep_last_deployments = 2
        original = Path.replace

        def replace_pointer(path, target):
            if target == self.pointer:
                raise OSError("pointer replacement")
            return original(path, target)

        with (
            patch.object(Path, "replace", replace_pointer),
            self.assertRaisesRegex(OSError, "pointer replacement"),
        ):
            self.publish()
        self.assertEqual(set(first.parent.iterdir()), {first, second, self.history})
        self.assertEqual(self.pointer.parent / self.pointer.read_text(), second)
        third = self.publish()
        self.assertEqual(set(first.parent.iterdir()), {second, third, self.history})

    def test_legacy_retained_config_and_ordinary_stage_one_at_same_path(self):
        manifest = load_manifest(self.retained)
        self.retained.with_suffix(".sha256").unlink()
        del manifest["retained_files_sha256"]
        file = self.retained / "manifest.yaml"
        file.chmod(0o600)
        save_config(manifest, file)
        self.cfg.model.features.artifact = str(self.retained)
        legacy = RLTokenPolicy(self.cfg, mode="infer")
        self.assertIsNone(legacy._retained_root)
        manifest["components"]["resume"] = {"status": "ready", "path": "resume/dcp"}
        manifest["training"] = {"step": 3}
        save_config(manifest, file)
        ordinary = RLTokenPolicy(self.cfg, mode="infer")
        self.assertIsNone(ordinary._retained_root)

    def test_migrate_writable_legacy_files_without_changing_external_hardlinks(self):
        manifest = load_manifest(self.retained)
        manifest.pop("retained_files_sha256")
        manifest.pop("source_deployment_sha256")
        seal = self.retained.with_suffix(".sha256")
        seal.unlink()
        for file in self.retained.rglob("*"):
            if file.is_file():
                file.chmod(0o644)
        save_config(manifest, self.retained / "manifest.yaml")
        asset = self.retained / manifest["files"]["model_assets"]["model.asset"]
        external = self.root / "other-run-asset"
        os.link(asset, external)
        external_stat = external.stat()
        old_directory = self.retained.stat().st_ino
        self.cfg.model.features.artifact = str(self.retained)
        direct = RLTokenPolicy(self.cfg, mode="infer")
        self.assertIsNone(direct._retained_root)
        with self.assertRaisesRegex(ValueError, "sealed"):
            RLTokenPolicy(self.cfg, mode="infer", retained_feature_root=self.retained)
        self.role.engine.policy = direct
        with patch("os.link", side_effect=AssertionError("shared writable file")):
            self.assertEqual(self.actions(self.publish()).shape, (1, 2, 2))

        with self.assertRaisesRegex(ValueError, "configured feature source"):
            retain_feature_bundle(self.source, self.retained)
        retain_feature_bundle(self.retained, self.retained)
        self.assertEqual(old_directory, self.retained.stat().st_ino)
        self.assertEqual(external.stat().st_ino, external_stat.st_ino)
        self.assertEqual(stat.S_IMODE(external.stat().st_mode), 0o644)
        self.assertFalse(asset.samefile(external))
        self.assertEqual(asset.read_bytes(), external.read_bytes())
        self.assertEqual(seal.read_text().strip(), file_digest(self.retained / "manifest.yaml"))
        self.assertEqual(stat.S_IMODE(seal.stat().st_mode), 0o400)
        self.assertIn("retained_files_sha256", load_manifest(self.retained))
        migrated = RLTokenPolicy(self.cfg, mode="infer", retained_feature_root=self.retained)
        self.assertEqual(migrated._retained_root, self.retained.resolve())

        relative = asset.relative_to(self.retained)
        verified = {relative: migrated._retained_files[relative]}
        asset.chmod(0o644)
        with patch("os.link") as link, self.assertRaisesRegex(ValueError, "unsealed"):
            _link_feature_bundle(self.retained, self.root / "race", verified)
        link.assert_not_called()

    def test_stage_one_validates_declared_inputs_before_loading(self):
        path = self.source / "manifest.yaml"
        valid = load_manifest(self.source)
        files = valid["files"]
        assets = self.source / "deployment/assets"
        (assets / "link").symlink_to(self.root / "asset-2.bin")
        (assets / "dirlink").symlink_to(self.root, target_is_directory=True)
        cases = [
            {"files": {**files, "model_assets": {"model.asset": "../asset-2.bin"}}},
            {"files": {**files, "model_assets": {"model.asset": "deployment/assets/link"}}},
            {
                "files": {
                    **files,
                    "model_assets": {"model.asset": "deployment/assets/dirlink/asset-2.bin"},
                }
            },
            {"files": {**files, "nexus_weights": str(self.source / files["nexus_weights"])}},
            {"files": {**files, "norm_stats": "../norm.json"}},
            {"files": {**files, "tokenizer": "deployment/assets/link"}},
            {"files": {**files, "custom_asset": "deployment/assets"}},
            {"weights_sha256": "0" * 64},
        ]
        destination = self.root / "fresh"
        with patch("dobot_nexus.algorithms.rltoken.artifacts.load_provider") as load:
            for case in cases:
                save_config({**valid, **case}, path)
                for target in (destination, self.retained):
                    with self.subTest(case=case, target=target), self.assertRaises(ValueError):
                        retain_feature_bundle(self.source, target)
            save_config(valid, self.root / "other-manifest.yaml")
            path.unlink()
            path.symlink_to(self.root / "other-manifest.yaml")
            with self.assertRaisesRegex(ValueError, "regular"):
                retain_feature_bundle(self.source, destination)
            load.assert_not_called()
        self.assertFalse(destination.exists())

    def test_step_a_retry_preserves_planned_digest_after_asset_tampering(self):
        manifest = load_manifest(self.retained)
        manifest.pop("retained_files_sha256")
        manifest.pop("source_deployment_sha256")
        path = self.retained / "manifest.yaml"
        seal = self.retained.with_suffix(".sha256")
        seal.unlink()
        path.chmod(0o644)
        save_config(manifest, path)
        replace = os.replace

        def fail_after_seal(src, dst):
            if dst == path:
                raise OSError(errno.ENOSPC, "interrupted legacy migration")
            return replace(src, dst)

        with patch("os.replace", fail_after_seal), self.assertRaises(OSError):
            retain_feature_bundle(self.retained, self.retained)
        planned = seal.read_bytes()
        self.assertEqual(len(planned.splitlines()), 2)
        asset = self.retained / manifest["files"]["model_assets"]["model.asset"]
        asset.chmod(0o600)
        asset.write_bytes(b"modified during interruption")
        before = {
            p: (p.stat().st_ino, file_digest(p)) for p in self.retained.rglob("*") if p.is_file()
        }
        with self.assertRaisesRegex(ValueError, "changed during interrupted migration"):
            retain_feature_bundle(self.retained, self.retained)
        self.assertEqual(seal.read_bytes(), planned)
        self.assertEqual({p: (p.stat().st_ino, file_digest(p)) for p in before}, before)

    def test_stage_one_ignores_training_only_files_and_directories(self):
        (self.source / "resume").mkdir()
        (self.source / "resume/trainer_state.pt").write_bytes(b"training state")
        (self.source / "resume/other").symlink_to(self.root / "missing")
        (self.source / "token_validation.json").write_text("{}")
        manifest = load_manifest(self.source)
        manifest["files"]["trainer_state"] = "resume/trainer_state.pt"
        save_config(manifest, self.source / "manifest.yaml")
        destination = retain_feature_bundle(self.source, self.root / "fresh")
        self.assertTrue((destination / "manifest.yaml").is_file())
        self.assertFalse((destination / "resume").exists())
        self.assertFalse((destination / "token_validation.json").exists())

    def test_stage_one_reuse_fingerprints_deployment_file_contents(self):
        manifest = load_manifest(self.source)
        asset = self.source / manifest["files"]["model_assets"]["model.asset"]
        config = self.source / "config_resolved.yaml"
        original_manifest = (self.source / "manifest.yaml").read_bytes()
        original_config = config.read_bytes()
        cfg = load_config(config)
        cfg.model.num_steps = cfg.model.get("num_steps", 10) + 10
        save_config(cfg, config)
        edited_config = config.read_bytes()
        config.write_bytes(original_config)
        for path, changed in ((asset, b"different calibration"), (config, edited_config)):
            with self.subTest(file=path.name):
                original = path.read_bytes()
                path.write_bytes(changed)
                with self.assertRaisesRegex(ValueError, "configured feature source"):
                    retain_feature_bundle(self.source, self.retained)
                self.assertEqual((self.source / "manifest.yaml").read_bytes(), original_manifest)
                path.write_bytes(original)
                self.assertEqual(retain_feature_bundle(self.source, self.retained), self.retained)

    def test_stage_one_reuse_accepts_pruned_training_state(self):
        manifest = load_manifest(self.source)
        trainer_state = self.source / "resume/trainer_state.pt"
        trainer_state.parent.mkdir()
        trainer_state.write_bytes(b"optimizer state")
        manifest["files"]["trainer_state"] = str(trainer_state.relative_to(self.source))
        manifest["components"]["resume"] = {"status": "ready", "path": "resume"}
        index = self.source / "manifest.yaml"
        save_config(manifest, index)
        source_before = file_digest(index)
        destination = retain_feature_bundle(self.source, self.root / "before-pruning")
        retained_manifest = destination / "manifest.yaml"
        original_inode = retained_manifest.stat().st_ino
        original_digest = file_digest(retained_manifest)
        manifest["files"].pop("trainer_state")
        manifest["components"]["resume"]["status"] = "pruned"
        trainer_state.unlink()
        save_config(manifest, index)
        self.assertNotEqual(source_before, file_digest(index))
        self.assertEqual(retain_feature_bundle(self.source, destination), destination)
        self.assertEqual(retained_manifest.stat().st_ino, original_inode)
        self.assertEqual(file_digest(retained_manifest), original_digest)

    def test_stage_one_decoder_is_excluded_from_retention_and_reuse(self):
        manifest = load_manifest(self.source)
        weights = self.source / manifest["files"]["weights"]
        state = torch.load(weights, map_location="cpu", weights_only=True)
        state["token_decoder.weight"] = torch.ones(3, 4)
        torch.save(state, weights)
        manifest["weights_sha256"] = file_digest(weights)
        trainer_state = self.source / "trainer_state.pt"
        trainer_state.touch()
        manifest["files"]["trainer_state"] = trainer_state.name
        manifest["components"]["resume"] = {"status": "ready"}
        index = self.source / "manifest.yaml"
        save_config(manifest, index)

        # Model the Stage 1 provider's inference loader dropping its training decoder.
        model_type = type(self.policy.provider)
        load = model_type.load_state_dict

        def load_infer(model, weights, **kwargs):
            self.assertEqual(model.mode, "infer")
            return load(model, {k: v for k, v in weights.items()
                                if not k.startswith("token_decoder.")}, **kwargs)

        with (
            patch.object(model_type, "load_state_dict", load_infer),
            patch.object(model_type, "export_backend_state_dict", autospec=True,
                         side_effect=model_type.export_backend_state_dict) as export_backend,
        ):
            retained = retain_feature_bundle(self.source, self.root / "inference-only")
        export_backend.assert_called_once()
        self.assertEqual(export_backend.call_args.args[0].mode, "infer")
        files = load_manifest(retained)["files"]
        for key in ("weights", "nexus_weights"):
            exported = torch.load(retained / files[key], map_location="cpu", weights_only=True)
            self.assertEqual(set(exported), set(state) - {"token_decoder.weight"})
        self.assertNotEqual(file_digest(retained / files["weights"]), file_digest(weights))
        inode = (retained / "manifest.yaml").stat().st_ino
        self.assertEqual(retain_feature_bundle(self.source, retained), retained)
        manifest["files"].pop("trainer_state")
        manifest["components"]["resume"]["status"] = "pruned"
        trainer_state.unlink()
        save_config(manifest, index)
        self.assertEqual(retain_feature_bundle(self.source, retained), retained)
        self.assertEqual((retained / "manifest.yaml").stat().st_ino, inode)
        for path in (self.source / manifest["files"]["model_assets"]["model.asset"],
                     self.source / "config_resolved.yaml"):
            original = path.read_bytes()
            path.write_bytes(original + b"\n# changed deployment input\n")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "feature source"):
                retain_feature_bundle(self.source, retained)
            path.write_bytes(original)

    def test_stage_one_source_changes_during_live_export_abort_retention(self):
        asset = self.source / load_manifest(self.source)["files"]["model_assets"]["model.asset"]
        destination = self.root / "interrupted-retention"

        def change_source(*, source_weights, artifact_dir):
            asset.write_bytes(b"changed after the provider loaded")
            return {}

        with (
            patch.object(type(self.policy.provider), "export_deployment_artifacts",
                         staticmethod(change_source)),
            self.assertRaisesRegex(ValueError, "deployment inputs changed during retention"),
        ):
            retain_feature_bundle(self.source, destination)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_suffix(".sha256").exists())

    def test_stage_one_config_must_match_fingerprint_during_provider_load(self):
        from dobot_nexus.algorithms.rltoken import artifacts

        relative = Path("config_resolved.yaml")
        path = self.source / relative
        config = load_config(path)
        config.model.num_steps = 10
        save_config(config, path)
        initial = path.read_bytes()
        digest = file_digest(path)

        def read_swapped_config(root, **kwargs):
            self.assertEqual(kwargs["_verified_files"][1][relative].sha256, digest)
            config.model.num_steps = 1
            save_config(config, path)
            try:
                return load_provider(root, **kwargs)
            finally:
                path.write_bytes(initial)

        destination = self.root / "fresh"
        with (
            patch.object(artifacts, "load_provider", read_swapped_config),
            patch("dobot_nexus.inference.policy.InferencePolicy", wraps=InferencePolicy) as ctor,
            self.assertRaisesRegex(ValueError, "Source config_resolved.yaml changed during load"),
        ):
            retain_feature_bundle(self.source, destination)
        ctor.assert_not_called()
        self.assertEqual(file_digest(path), digest)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_suffix(".sha256").exists())
        restored = retain_feature_bundle(self.source, destination)
        self.assertEqual(load_config(restored / relative).model.num_steps, 10)

    def test_stage_one_fingerprint_normalizes_legacy_deployment_assets(self):
        manifest = load_manifest(self.source)
        files = manifest["files"]
        files.update(norm_stats="stats.json", tokenizer="tokenizer.model")
        (self.source / "stats.json").write_text("{}")
        (self.source / "tokenizer.model").write_bytes(b"bundled tokenizer")
        index = self.source / "manifest.yaml"
        save_config(manifest, index)
        model_type = type(self.policy.provider)
        provider_assets = model_type.artifact_assets

        def with_tokenizer(model):
            assets = provider_assets(model)
            assets["data.tokenizer_path"] = model.config.data.tokenizer_path
            return assets

        with patch.object(model_type, "artifact_assets", with_tokenizer):
            destination = retain_feature_bundle(self.source, self.root / "aliases")
        files["model_assets"]["data.norm_stats_path"] = files.pop("norm_stats")
        files["model_assets"]["data.tokenizer_path"] = files.pop("tokenizer")
        save_config(manifest, index)
        self.assertEqual(retain_feature_bundle(self.source, destination), destination)
        (self.source / "tokenizer.model").write_bytes(b"changed tokenizer")
        with self.assertRaisesRegex(ValueError, "configured feature source"):
            retain_feature_bundle(self.source, destination)

    def test_cross_source_reuse_respects_each_sources_permission_ceiling(self):
        copy = retain_feature_bundle(self.retained, self.root / "copy")
        for path in self.retained.rglob("*"):
            if path.is_file():
                duplicate = copy / path.relative_to(self.retained)
                self.assertEqual(file_digest(path), file_digest(duplicate))
                path.chmod(0o400)
                duplicate.chmod(0o444)
        pin = file_digest(self.retained / "manifest.yaml")
        with self.assertRaisesRegex(ValueError, "broader permissions than source"):
            retain_feature_bundle(self.retained, copy, expected_manifest_sha256=pin)
        manifest = load_manifest(self.source)
        asset = manifest["files"]["model_assets"]["model.asset"]
        (self.source / asset).chmod(0o400)
        with self.assertRaisesRegex(ValueError, "broader permissions than source"):
            retain_feature_bundle(self.source, copy)
        self.assertEqual(stat.S_IMODE((copy / asset).stat().st_mode), 0o444)

    def test_cross_run_copy_aborts_if_source_mode_tightens_after_copy(self):
        from dobot_nexus.algorithms.rltoken import artifacts

        weights = self.retained / load_manifest(self.retained)["files"]["weights"]
        weights.chmod(0o444)
        destination = self.root / "copy"
        original = artifacts._copy_preserving_ceiling

        def tighten_after_copy(src, dst, **kwargs):
            original(src, dst, **kwargs)
            if src == weights:
                self.assertEqual(stat.S_IMODE(dst.stat().st_mode), 0o444)
                weights.chmod(0o400)

        with (patch.object(artifacts, "_copy_preserving_ceiling", tighten_after_copy),
              self.assertRaisesRegex(ValueError, "changed during copy")):
            retain_feature_bundle(self.retained, destination)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_suffix(".sha256").exists())
        copy = retain_feature_bundle(self.retained, destination)
        copied = copy / weights.relative_to(self.retained)
        self.assertEqual(stat.S_IMODE(copied.stat().st_mode), 0o400)

    def legacy_copy(self, run, mode=0o644):
        root = shutil.copytree(self.retained, self.root / run / "rltoken/features")
        manifest = load_manifest(root)
        manifest.pop("retained_files_sha256")
        manifest.pop("source_deployment_sha256")
        (root / "manifest.yaml").chmod(0o600)
        save_config(manifest, root / "manifest.yaml")
        for path in root.rglob("*"):
            if path.is_file():
                path.chmod(mode)
        return root

    def test_legacy_cross_run_reuse_rejects_swapped_calibration_binding(self):
        source, destination = self.legacy_copy("a"), self.legacy_copy("b")
        for root in (source, destination):
            (root / "alternate.bin").write_bytes(b"different calibration")
        manifest = load_manifest(destination)
        manifest["files"]["model_assets"]["model.asset"] = "alternate.bin"
        save_config(manifest, destination / "manifest.yaml")
        with self.assertRaisesRegex(ValueError, "configured feature source"):
            retain_feature_bundle(source, destination)
        self.assertFalse(destination.with_suffix(".sha256").exists())

    def test_cross_run_ceiling_follows_logical_key_through_swaps_and_aliases(self):
        source = self.legacy_copy("source")
        left, right = "first.bin", "second.bin"

        def install(root, first, second):
            # Identical bytes exclude content mismatches as a cause of rejection.
            for rel, mode in ((left, 0o400), (right, 0o444)):
                (root / rel).write_bytes(b"calibration")
                (root / rel).chmod(mode)
            manifest = load_manifest(root)
            manifest["files"].update(calibration_a=first, calibration_b=second)
            save_config(manifest, root / "manifest.yaml")

        install(source, left, right)
        for second in (left, right):
            with self.subTest(second=second):
                destination = self.legacy_copy(second)
                install(destination, right, second)
                with self.assertRaisesRegex(ValueError, "broader permissions"):
                    retain_feature_bundle(source, destination)
                (destination / right).chmod(0o400)
                self.assertEqual(retain_feature_bundle(source, destination), destination)

    def test_cross_run_legacy_modes_are_checked_before_and_after_migration(self):
        source, destination = self.legacy_copy("a"), self.legacy_copy("b")
        self.assertEqual(retain_feature_bundle(source, destination), destination)
        self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o444
                            for p in destination.rglob("*") if p.is_file()))
        self.assertEqual(destination.with_suffix(".sha256").read_text().strip(),
                         file_digest(destination / "manifest.yaml"))
        for src_mode in (0o600, 0o400):
            private = self.legacy_copy(f"private-{src_mode}", src_mode)
            for dst_mode in (0o644, 0o444):
                public = self.legacy_copy(f"public-{src_mode}-{dst_mode}", dst_mode)
                with (self.subTest(src_mode=src_mode, dst_mode=dst_mode),
                      self.assertRaisesRegex(ValueError, "broader permissions")):
                    retain_feature_bundle(private, public)
                self.assertFalse(public.with_suffix(".sha256").exists())

        from dobot_nexus.algorithms.rltoken import artifacts

        private = self.legacy_copy("private", 0o600)
        destination = self.legacy_copy("failed-migration", 0o600)
        migrate = artifacts._migrate_retained_bundle

        def widen_after_migration(*args):
            migrate(*args)
            (destination / "manifest.yaml").chmod(0o444)

        with (
            patch.object(artifacts, "_migrate_retained_bundle", widen_after_migration),
            self.assertRaisesRegex(ValueError, "broader permissions"),
        ):
            retain_feature_bundle(private, destination)

    def assert_constructor_rejects_unopened(self, external, error):
        with (
            patch("dobot_nexus.inference.policy.InferencePolicy",
                  wraps=InferencePolicy) as construct,
            patch("builtins.open", wraps=open) as builtin_open,
            patch.object(Path, "open", autospec=True, side_effect=Path.open) as path_open,
            self.assertRaisesRegex(ValueError, error),
        ):
            load_provider(self.source)
        construct.assert_not_called()
        opened = {Path(call.args[0]) for call in (*builtin_open.call_args_list,
                                                   *path_open.call_args_list)}
        self.assertIn(self.source / "config_resolved.yaml", opened)
        self.assertNotIn(external, opened)

    def test_external_constructor_asset_is_rejected_before_any_model_or_file_open(self):
        cfg_path = self.source / "config_resolved.yaml"
        cfg = load_config(cfg_path)
        external = self.root / "norm_stats.json"
        external.write_text("{}")
        cfg.data.norm_stats_path = str(external)
        save_config(cfg, cfg_path)
        self.assert_constructor_rejects_unopened(external, "Undeclared.*data.norm_stats_path")

    def test_load_and_live_export_require_every_declared_asset(self):
        self.cfg.model.features.artifact = str(self.source)
        policy = RLTokenPolicy(self.cfg, mode="infer")
        self.assertIsNone(policy._retained_root)
        with patch.object(type(policy.provider), "artifact_assets", return_value={}):
            with self.assertRaisesRegex(ValueError, "Export omitted.*model.asset"):
                load_provider(self.source)
            with self.assertRaisesRegex(ValueError, "Export omitted.*model.asset"):
                policy.export_deployment_artifacts(source_weights=None, artifact_dir=self.root)

    def test_model_calibration_path_is_checked_before_constructor_reads_it(self):
        external = self.root / "calibration.json"
        external.write_bytes(b"external calibration")
        config_path = self.source / "config_resolved.yaml"
        config = load_config(config_path)
        config.model.calibration_path = str(external)
        save_config(config, config_path)
        model_type = type(self.policy.provider)
        initialize = model_type.__init__
        assets = model_type.artifact_assets

        def read_calibration(model, cfg, *, mode):
            initialize(model, cfg, mode=mode)
            saved = load_config(Path(cfg.model.artifact) / "config_resolved.yaml",
                                expected_task="train")
            self.assertEqual(saved.model.calibration_path, cfg.model.calibration_path)
            model.calibration = Path(saved.model.calibration_path).read_bytes()

        with patch.object(model_type, "__init__", read_calibration):
            self.assert_constructor_rejects_unopened(
                external, "Undeclared external asset path at model.calibration_path"
            )

            bundled = self.source / "deployment/calibration.json"
            bundled.write_bytes(b"bundled calibration")
            config.model.calibration_path = str(bundled)
            save_config(config, config_path)
            manifest = load_manifest(self.source)
            manifest["files"]["calibration"] = str(bundled.relative_to(self.source))
            save_config(manifest, self.source / "manifest.yaml")
            self.assert_constructor_rejects_unopened(bundled, "Undeclared.*model.calibration_path")
            manifest["files"]["model_assets"]["model.calibration_path"] = (
                manifest["files"].pop("calibration")
            )
            save_config(manifest, self.source / "manifest.yaml")
            config.model.calibration_path = str(external)
            save_config(config, config_path)
            external.unlink()

            def report_assets(model):
                return {**assets(model),
                        "model.calibration_path": model.config.model.calibration_path}

            with patch.object(model_type, "artifact_assets", report_assets):
                retained = retain_feature_bundle(self.source, self.root / "relocated-features")
                shutil.rmtree(self.source)
                provider, restored, _ = load_provider(retained)
            asset = load_manifest(retained)["files"]["model_assets"]["model.calibration_path"]
            self.assertEqual(Path(restored.model.calibration_path), retained / asset)
            self.assertEqual(provider.calibration, b"bundled calibration")

    def test_units_with_slashes_are_valid_model_metadata(self):
        cfg_path = self.source / "config_resolved.yaml"
        cfg = load_config(cfg_path)
        for units in ("rad/s", "m/s"):
            with self.subTest(units=units):
                cfg.model["rltoken"] = {"action_codec": {"units": units}}
                save_config(cfg, cfg_path)
                _, loaded, _ = load_provider(self.source)
                self.assertEqual(loaded.model.rltoken.action_codec.units, units)

    def test_existing_undeclared_files_in_model_and_data_are_never_opened(self):
        cfg_path = self.source / "config_resolved.yaml"
        cfg = load_config(cfg_path)
        external = self.root / "calibration.json"
        external.write_text("{}")
        relative = self.source / "calibration.json"
        relative.write_text("{}")
        dangling = self.source / "missing-link"
        dangling.symlink_to(self.root / "does-not-exist")
        for section, key, value, file in (
            ("data", "calibration_path", str(external), external),
            ("model", "calibration", relative.name, relative),
            ("model", "blocks", [{"lookup": dangling.name}], dangling),
        ):
            with self.subTest(key=key):
                edited = clone_config(cfg)
                edited[section][key] = value
                save_config(edited, cfg_path)
                self.assert_constructor_rejects_unopened(file, f"Undeclared.*{section}.{key}")

    def test_prompt_and_metadata_can_name_existing_paths(self):
        self.assertTrue(Path("README.md").is_file())
        config_path = self.source / "config_resolved.yaml"
        config = load_config(config_path)
        config.data.default_prompt = "README.md"
        config.data.field_mapping = {"nested": ["README.md"]}
        config.data.transforms = [{"name": "README.md"}]
        config.model.io = {"cameras": {"base": "README.md"}}
        config.model.rltoken = {"action_codec": {"units": "README.md"},
                                "token": {"name": "README.md"}}
        save_config(config, config_path)
        _, restored, _ = load_provider(self.source)
        self.assertEqual(restored.data.default_prompt, "README.md")

    def test_pi05_training_directories_are_accepted_during_inference(self):
        config_path = self.source / "config_resolved.yaml"
        config = load_config(config_path)
        for directory in ("openpi", "stats-cache"):
            (self.root / directory).mkdir()
        config.model.openpi_root = str(self.root / "openpi")
        config.data.norm_stats_cache_dir = str(self.root / "stats-cache")
        save_config(config, config_path)
        _, restored, _ = load_provider(self.source)
        self.assertEqual(restored.task, "infer")

    def test_custom_io_and_rltoken_calibration_paths_require_a_binding(self):
        path = self.root / "calibration.bin"
        path.write_bytes(b"external calibration")
        config_path = self.source / "config_resolved.yaml"
        original = load_config(config_path)
        for subtree in ("io", "rltoken"):
            with self.subTest(subtree=subtree):
                config = clone_config(original)
                config.model[subtree] = {"calibration_path": str(path)}
                save_config(config, config_path)
                self.assert_constructor_rejects_unopened(
                    path, f"Undeclared external asset path at model.{subtree}.calibration_path"
                )

    def test_undeclared_calibration_directory_is_rejected_before_construction(self):
        directory = self.root / "calibration"
        directory.mkdir()
        scale = directory / "scale.json"
        scale.write_text("{}")
        config_path = self.source / "config_resolved.yaml"
        config = load_config(config_path)
        config.model.calibration_dir = str(directory)
        save_config(config, config_path)

        def read_scale(model, cfg, *, mode):
            (Path(cfg.model.calibration_dir) / "scale.json").read_bytes()

        with patch.object(type(self.policy.provider), "__init__", read_scale):
            self.assert_constructor_rejects_unopened(
                scale, "Undeclared external asset path.*calibration_dir"
            )

    def test_retained_provider_parses_only_the_bytes_authenticated_by_its_seal(self):
        from dobot_nexus.algorithms.rltoken import artifacts

        config_path = self.retained / "config_resolved.yaml"
        original = load_config(config_path)
        swapped = clone_config(original)
        swapped.model.num_steps = 317
        replacement = self.root / "swapped.yaml"
        save_config(swapped, replacement)

        def swap_after_verification(*args, **kwargs):
            verified = _verify_retained_feature_bundle(*args, **kwargs)
            replacement.replace(config_path)
            return verified

        constructed_with = []

        def construct(*args, **kwargs):
            policy = InferencePolicy(*args, **kwargs)
            constructed_with.append(policy.config.model.get("num_steps"))
            return policy

        with (
            patch.object(artifacts, "_verify_retained_feature_bundle", swap_after_verification),
            patch("dobot_nexus.inference.policy.InferencePolicy", side_effect=construct),
            patch("builtins.open", wraps=open) as builtin_open,
            patch.object(Path, "open", autospec=True, side_effect=Path.open) as path_open,
            self.assertRaisesRegex(ValueError, "file changed during provider load"),
        ):
            load_provider(self.retained)
        self.assertEqual(constructed_with, [original.model.get("num_steps")])
        calls = builtin_open.call_args_list + path_open.call_args_list
        opened = [call.args[0] for call in calls]
        self.assertEqual(opened.count(config_path), 1)
        self.assertEqual(load_config(config_path).model.num_steps, 317)
        with self.assertRaisesRegex(ValueError, "Retained config_resolved.yaml changed"):
            load_provider(self.retained)

    def test_provider_constructor_uses_the_validated_config_snapshot(self):
        config_path = self.source / "config_resolved.yaml"
        external = self.root / "unverified-norm_stats.json"
        external.write_text("{}")
        swapped = load_config(config_path)
        swapped.data.norm_stats_path = str(external)
        replacement = self.root / "swapped.yaml"
        save_config(swapped, replacement)

        def swap_then_construct(*args, **kwargs):
            replacement.replace(config_path)
            (self.source / "manifest.yaml").rename(self.root / "original-manifest.yaml")
            return InferencePolicy(*args, **kwargs)

        model_type = type(self.policy.provider)
        initialize = model_type.__init__

        observed = []

        def construct(model, cfg, *, mode):
            # Stage 1 models can also restore settings from a model.artifact locator.
            root = resolve_artifact(cfg.model.artifact)
            trained = load_config(root / "config_resolved.yaml")
            observed.append(trained.data.get("norm_stats_path"))
            if trained.data.get("norm_stats_path"):
                Path(trained.data.norm_stats_path).read_bytes()
            initialize(model, cfg, mode=mode)

        with (
            patch("dobot_nexus.inference.policy.InferencePolicy", side_effect=swap_then_construct),
            patch.object(model_type, "__init__", construct),
            patch("builtins.open", wraps=open) as builtin_open,
            patch.object(Path, "open", autospec=True, side_effect=Path.open) as path_open,
            self.assertRaisesRegex(ValueError, "file changed during provider load"),
        ):
            load_provider(self.source)
        self.assertEqual(observed, [None])
        opened = [Path(call.args[0]) for call in (*builtin_open.call_args_list,
                                                  *path_open.call_args_list)]
        self.assertEqual(opened.count(config_path), 1)
        self.assertNotIn(external, opened)
        self.assertFalse(replacement.exists())

    def test_retained_asset_change_during_construction_discards_the_loaded_provider(self):
        from dobot_nexus.algorithms.rltoken import artifacts

        rel = load_manifest(self.retained)["files"]["model_assets"]["model.asset"]
        asset = self.retained / rel
        original = asset.read_bytes()
        changed = b"x" * len(original)
        model_type = type(self.policy.provider)
        initialize = model_type.__init__
        consumed = []

        def consume_asset(model, cfg, *, mode):
            initialize(model, cfg, mode=mode)
            consumed.append(Path(cfg.model.asset).read_bytes())

        timestamp = 1_700_000_000_000_000_000
        for replace_inode in (True, False):
            asset.chmod(0o644)
            asset.write_bytes(original)
            asset.chmod(0o444)
            os.utime(asset, ns=(timestamp, timestamp))
            consumed.clear()

            def swap_after_verification(*args, replace_inode=replace_inode, **kwargs):
                verified = _verify_retained_feature_bundle(*args, **kwargs)
                if replace_inode:
                    replacement = self.root / "replacement.bin"
                    replacement.write_bytes(changed)
                    replacement.replace(asset)
                else:
                    asset.chmod(0o644)
                    asset.write_bytes(changed)
                asset.chmod(0o444)
                os.utime(asset, ns=(timestamp, timestamp))
                same_identity = artifacts._file_identity(asset) == verified[Path(rel)].identity
                self.assertEqual(same_identity, not replace_inode)
                return verified

            with (
                self.subTest(replace_inode=replace_inode),
                patch.object(artifacts, "_verify_retained_feature_bundle", swap_after_verification),
                patch.object(model_type, "__init__", consume_asset),
                self.assertRaisesRegex(ValueError, "file changed during provider load"),
            ):
                load_provider(self.retained)
            self.assertEqual(consumed, [changed])

    def test_provider_rechecks_both_weight_identities_without_rehashing(self):
        from dobot_nexus.algorithms.rltoken import artifacts

        files = load_manifest(self.source)["files"]
        for field in ("weights", "nexus_weights"):
            weight = self.source / files[field]

            def change_after_load(*args, weight=weight, **kwargs):
                policy = InferencePolicy(*args, **kwargs)
                weight.chmod(stat.S_IMODE(weight.stat().st_mode) ^ 0o100)
                return policy

            with (
                self.subTest(field=field),
                patch("dobot_nexus.inference.policy.InferencePolicy", change_after_load),
                patch.object(artifacts, "file_digest", wraps=file_digest) as hashed,
                self.assertRaisesRegex(ValueError, "file changed during provider load"),
            ):
                load_provider(self.source)
            hashed_paths = [call.args[0] for call in hashed.call_args_list]
            for key in ("weights", "nexus_weights"):
                self.assertEqual(hashed_paths.count(self.source / files[key]), 1)

    def test_constructor_loads_declared_sidecar_through_artifact_locator(self):
        manifest = load_manifest(self.source)
        sidecar = "deployment/provider_info.json"
        (self.source / sidecar).write_text('{"scale": 17}')
        manifest["files"]["provider_info"] = sidecar
        save_config(manifest, self.source / "manifest.yaml")
        config = load_config(self.source / "config_resolved.yaml")
        config.model.artifact = str(self.source)
        config.task = "infer"
        model_type = type(self.policy.provider)
        initialize = model_type.__init__

        def read_sidecar(model, cfg, *, mode):
            initialize(model, cfg, mode=mode)
            root = resolve_artifact(cfg.model.artifact)
            rel = load_manifest(root)["files"].pop("provider_info")
            model.sidecar = json.loads((root / rel).read_bytes())

        with patch.object(model_type, "__init__", read_sidecar):
            ordinary = InferencePolicy(config)
            provider, restored, _ = load_provider(self.source)
        self.assertEqual(restored.model.artifact, str(self.source))
        self.assertEqual(provider.sidecar, {"scale": 17})
        self.assertEqual(provider.sidecar, ordinary.model.sidecar)

    def test_snapshot_preserves_pi05_operator_numerics_check_with_one_config_read(self):
        from dobot_nexus.models.pi05.configuration_pi05 import resolve_core_settings

        config_path = self.source / "config_resolved.yaml"
        config = load_config(config_path)
        config.model.operator_numerics = "legacy"
        save_config(config, config_path)
        manifest = load_manifest(self.source)
        manifest["model"]["operator_numerics"] = "openpi"
        save_config(manifest, self.source / "manifest.yaml")
        config.model.artifact = str(self.source)
        config.task = "infer"

        def check_settings(model, cfg, *, mode):
            resolve_core_settings(cfg, mode=mode)
            self.fail("Conflicting Pi0.5 operator_numerics was accepted")

        with patch.object(type(self.policy.provider), "__init__", check_settings):
            with self.assertRaisesRegex(ValueError, "Artifact operator_numerics disagrees"):
                InferencePolicy(config)
            with (
                patch.object(Path, "open", autospec=True, side_effect=Path.open) as path_open,
                self.assertRaisesRegex(ValueError, "Artifact operator_numerics disagrees"),
            ):
                load_provider(self.source)
        self.assertEqual(sum(c.args[0] == config_path for c in path_open.call_args_list), 1)
        self.assertIsNone(get_verified_artifact_snapshot(self.source))

    def test_verified_snapshot_is_scoped_isolated_and_resolves_pointers(self):
        name = "config_resolved.yaml"
        manifest = load_manifest(self.source)
        config = load_config(self.source / name)
        pointer = self.root / "latest.txt"
        pointer.write_text(str(self.source))
        with verified_artifact_snapshot(self.source, manifest, config):
            (self.source / "manifest.yaml").unlink()
            loaded, saved = get_verified_artifact_snapshot(pointer)
            loaded["files"].clear()
            saved.model.target = "changed"
            self.assertEqual(load_manifest(pointer), manifest)
            self.assertEqual(load_config(self.source / name), config)
            self.assertIsNone(get_verified_artifact_snapshot(self.retained))
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertIsNone(pool.submit(get_verified_artifact_snapshot, self.source).result())
        self.assertIsNone(get_verified_artifact_snapshot(pointer))
        with self.assertRaises(FileNotFoundError):
            load_manifest(pointer)

    def test_load_config_bytes_shares_yaml_inheritance_validation_and_environment(self):
        path = self.root / "recipe.yaml"
        base = self.root / "base.yaml"
        save_config(load_config(self.source / "config_resolved.yaml"), base)
        raw = b"extends: base.yaml\nmodel:\n  note: '${env:RLTOKEN_CONFIG_NOTE}'\n"
        path.write_bytes(raw)
        with patch.dict(os.environ, RLTOKEN_CONFIG_NOTE="from environment"):
            expected = load_config(path, expected_task="train")
            path.unlink()
            self.assertEqual(load_config_bytes(raw, path, expected_task="train"), expected)
            with self.assertRaisesRegex(ValueError, "Expected task="):
                load_config_bytes(raw, path, expected_task="infer")
        with (patch.object(Path, "open", side_effect=AssertionError("unexpected config read")),
              self.assertRaisesRegex(ValueError, "Resolved config cannot use extends")):
            load_config_bytes(raw, path, allow_extends=False)

    def test_bundled_tokenizer_takes_precedence_over_legacy_config(self):
        external = self.root / "old-tokenizer.model"
        external.write_bytes(b"training tokenizer")
        config_path = self.source / "config_resolved.yaml"
        cfg = load_config(config_path)
        cfg.model["backend"] = {"paligemma_tokenizer": str(external)}
        save_config(cfg, config_path)
        bundled = self.source / "deployment/tokenizer.model"
        bundled.write_bytes(b"bundled tokenizer")
        manifest = load_manifest(self.source)
        assets = manifest["files"]["model_assets"]
        assets["data.tokenizer_path"] = str(bundled.relative_to(self.source))
        index = self.source / "manifest.yaml"
        save_config(manifest, index)
        model_type = type(self.policy.provider)
        provider_assets = model_type.artifact_assets

        def report_tokenizer(model):
            return {
                **provider_assets(model),
                "data.tokenizer_path": model.config.data.tokenizer_path,
            }

        with patch.object(model_type, "artifact_assets", report_tokenizer):
            _, loaded, _ = load_provider(self.source)
        self.assertEqual(Path(loaded.data.tokenizer_path), bundled)
        assets.pop("data.tokenizer_path")
        save_config(manifest, index)
        self.assert_constructor_rejects_unopened(external, "Undeclared.*data.tokenizer_path")

    def test_export_rejects_disagreement_between_model_and_config_norm_stats(self):
        provider, cfg, _ = load_provider(self.source)
        reported = self.root / "norm_stats_a.json"
        reported.write_text('{"mean": 1}')
        configured = self.root / "stats" / "norm_stats.json"
        configured.parent.mkdir()
        configured.write_text('{"mean": 2}')
        cfg.data.norm_stats_path = str(configured.parent)
        assets = provider.artifact_assets()
        allowed = {key: Path(path) for key, path in assets.items()}
        allowed["data.norm_stats_path"] = configured
        for permitted, error in ((None, "disagree"), (allowed, "Undeclared")):
            dest = self.root / ("unbound" if permitted is None else "bound")
            with (
                self.subTest(dest=dest),
                patch.object(provider, "artifact_assets",
                             return_value={**assets, "data.norm_stats_path": str(reported)}),
                self.assertRaisesRegex(ValueError, error),
            ):
                write_model_bundle(dest, provider, cfg, allowed_asset_files=permitted)
            self.assertFalse((dest / "manifest.yaml").exists())
            self.assertEqual(dest.stat().st_mode & 0o777, 0o700)
        with patch.object(provider, "artifact_assets",
                          return_value={**assets, "data.norm_stats_path": str(configured)}):
            dest = write_model_bundle(
                self.root / "matching", provider, cfg, allowed_asset_files=allowed
            )
        rel = load_manifest(dest)["files"]["model_assets"]["data.norm_stats_path"]
        self.assertEqual((dest / rel).read_bytes(), configured.read_bytes())

    def test_model_deployment_files_participate_in_reuse_and_live_export_ceilings(self):
        relative = "deployment/policy_info.json"
        output_relative = relative

        def source_info(*, source_weights, artifact_dir):
            self.assertEqual(artifact_dir.stat().st_mode & 0o777, 0o700)
            path = artifact_dir / output_relative
            path.write_text('{"calibration": 1}')
            path.chmod(0o644)
            return {"policy_info": output_relative}

        source = self.root / "stage-one-with-info"
        provider, cfg, _ = load_provider(self.source)
        path = source / relative
        with patch.object(type(provider), "export_deployment_artifacts", staticmethod(source_info)):
            write_model_bundle(source, provider, cfg)
            path.chmod(0o600)
            output_relative = "deployment/retained_policy_info.json"
            destination = retain_feature_bundle(source, self.root / "extra")
        retained_info = destination / output_relative
        self.assertEqual(stat.S_IMODE(retained_info.stat().st_mode), 0o400)
        self.assertEqual(retain_feature_bundle(source, destination), destination)
        retained_info.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "broader permissions"):
            retain_feature_bundle(source, destination)
        retained_info.chmod(0o400)
        path.write_text('{"calibration": 2}')
        with self.assertRaisesRegex(ValueError, "configured feature source"):
            retain_feature_bundle(source, destination)

        # A model can choose a new relative path during live export. The logical
        # key carries the ceiling from the original deployment.
        new_relative = "deployment/nested/policy_info.json"

        def export(*, source_weights, artifact_dir):
            self.assertEqual(artifact_dir.stat().st_mode & 0o777, 0o700)
            dst = artifact_dir / new_relative
            dst.parent.mkdir(parents=True)
            dst.write_bytes(path.read_bytes())
            dst.chmod(0o644)
            return {"policy_info": new_relative}

        self.cfg.model.features.artifact = str(source)
        policy = RLTokenPolicy(self.cfg, mode="infer")
        with patch.object(policy.provider, "export_deployment_artifacts", side_effect=export):
            policy.export_deployment_artifacts(source_weights=None, artifact_dir=self.root / "live")
        child = self.root / "live/deployment/features"
        self.assertEqual(load_manifest(child)["files"]["policy_info"], new_relative)
        self.assertEqual(stat.S_IMODE((child / new_relative).stat().st_mode), 0o600)
        default_dir = self.root / "default-mode"
        default_dir.mkdir()
        for finished in (child, destination):
            self.assertEqual(stat.S_IMODE(finished.stat().st_mode),
                             stat.S_IMODE(default_dir.stat().st_mode))
            self.assertFalse((finished / ".mode-probe").exists())

    def test_export_hook_cannot_claim_or_silently_overwrite_a_copied_asset(self):
        provider, config, _ = load_provider(self.source)
        rel = "deployment/assets/model_asset.bin"
        for reported in (True, False):
            def overwrite(*, source_weights, artifact_dir, reported=reported):
                (artifact_dir / rel).write_bytes(b"sidecar")
                return {"sidecar": rel} if reported else {}

            with (
                self.subTest(reported=reported),
                patch.object(type(provider), "export_deployment_artifacts",
                             staticmethod(overwrite)),
            ):
                error = "output collision" if reported else "changed generated file"
                direct = self.root / f"direct-{reported}"
                with self.assertRaisesRegex(ValueError, error):
                    write_model_bundle(direct, provider, config)
                self.assertFalse((direct / "manifest.yaml").exists())
                retained = self.root / f"retained-{reported}"
                with self.assertRaisesRegex(ValueError, error):
                    retain_feature_bundle(self.source, retained)
                self.assertFalse(retained.exists())
                self.assertFalse(retained.with_suffix(".sha256").exists())

    def test_deployment_outputs_require_distinct_unreserved_paths(self):
        provider, config, _ = load_provider(self.source)
        for rel in ("config_resolved.yaml", "deployment/pytorch/full_weights.pt",
                    "deployment/nexus/full_weights.pt", "deployment/sidecar"):
            def export(*, source_weights, artifact_dir, rel=rel):
                if rel == "deployment/sidecar":
                    (artifact_dir / rel).write_text("sidecar")
                    return {"first": rel, "second": rel}
                return {"sidecar": rel}

            with (self.subTest(rel=rel),
                  patch.object(provider, "export_deployment_artifacts", side_effect=export),
                  self.assertRaisesRegex(ValueError, "output collision")):
                direct = self.root / ("export-" + rel.replace("/", "-"))
                write_model_bundle(direct, provider, config)

    def test_direct_export_rejects_broadened_copied_asset_permissions(self):
        provider, config, _ = load_provider(self.source)
        source = Path(provider.artifact_assets()["model.asset"])
        source.chmod(0o400)
        relative = "deployment/assets/model_asset.bin"

        def broaden(*, source_weights, artifact_dir):
            copied = artifact_dir / relative
            self.assertEqual(stat.S_IMODE(copied.stat().st_mode), 0o400)
            copied.chmod(0o644)
            return {}

        direct = self.root / "direct"
        error = f"Export hook broadened asset permissions: {relative}"
        with (
            patch.object(provider, "export_deployment_artifacts", side_effect=broaden),
            self.assertRaisesRegex(ValueError, error),
        ):
            write_model_bundle(direct, provider, config)
        self.assertFalse((direct / "manifest.yaml").exists())
        self.assertEqual(direct.stat().st_mode & 0o777, 0o700)
        self.assertEqual(stat.S_IMODE(source.stat().st_mode), 0o400)

    def test_silent_edits_to_config_or_weights_abort_direct_export_and_retention(self):
        config_path = self.source / "config_resolved.yaml"
        config = load_config(config_path)
        config.model.num_steps = 10
        save_config(config, config_path)
        provider, config, _ = load_provider(self.source)
        files = load_manifest(self.source)["files"]
        for key in ("resolved_config", "weights", "nexus_weights"):
            relative = files[key]

            def overwrite(*, source_weights, artifact_dir, rel=relative):
                target = artifact_dir / rel
                if target.name == "config_resolved.yaml":
                    edited = load_config(target)
                    self.assertEqual(edited.model.num_steps, 10)
                    edited.model.num_steps = 1
                    save_config(edited, target)
                else:
                    target.write_bytes(b"changed weights")
                return {}

            with (self.subTest(key=key),
                  patch.object(type(provider), "export_deployment_artifacts",
                               staticmethod(overwrite))):
                direct = self.root / f"direct-{key}"
                with self.assertRaisesRegex(ValueError, f"changed generated file: {relative}"):
                    write_model_bundle(direct, provider, config)
                self.assertFalse((direct / "manifest.yaml").exists())
                retained = self.root / f"retained-{key}"
                with self.assertRaisesRegex(ValueError, f"changed generated file: {relative}"):
                    retain_feature_bundle(self.source, retained)
                self.assertFalse(retained.exists())
                self.assertFalse(retained.with_suffix(".sha256").exists())

    def test_fresh_retention_rechecks_live_source_modes_and_output_keys(self):
        rel = "deployment/policy_info.json"
        src_file = self.source / rel
        src_file.write_text("metadata")
        src_file.chmod(0o644)
        manifest = load_manifest(self.source)
        manifest["files"]["policy_info"] = rel
        save_config(manifest, self.source / "manifest.yaml")
        for tighten in (False, True):
            def export(*, source_weights, artifact_dir, tighten=tighten):
                output = artifact_dir / rel
                output.write_bytes(src_file.read_bytes())
                output.chmod(0o644)
                if tighten:
                    src_file.chmod(0o600)
                    return {"policy_info": rel}
                extra = "deployment/inference-only.json"
                (artifact_dir / extra).write_text("{}")
                return {"policy_info": rel, "inference_only": extra}

            destination = self.root / f"fresh-{tighten}"
            error = "broader permissions" if tighten else "no configured feature source"
            with (self.subTest(tighten=tighten),
                  patch.object(type(self.policy.provider), "export_deployment_artifacts",
                               staticmethod(export)),
                  self.assertRaisesRegex(ValueError, error)):
                retain_feature_bundle(self.source, destination)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".sha256").exists())

    def test_hook_hardlink_is_detached_before_applying_source_ceiling(self):
        external = self.root / "external-policy-info.json"
        external.write_text("{}")
        external.chmod(0o644)
        relative = "deployment/policy_info.json"
        source_file = self.source / relative
        source_file.write_bytes(external.read_bytes())
        source_file.chmod(0o600)
        manifest = load_manifest(self.source)
        manifest["files"]["policy_info"] = relative
        save_config(manifest, self.source / "manifest.yaml")

        def link_sidecar(*, source_weights, artifact_dir):
            os.link(external, artifact_dir / relative)
            return {"policy_info": relative}

        with patch.object(type(self.policy.provider), "export_deployment_artifacts",
                          staticmethod(link_sidecar)):
            retained = retain_feature_bundle(self.source, self.root / "fresh")
        private = retained / relative
        self.assertFalse(os.path.samefile(external, private))
        self.assertEqual(stat.S_IMODE(external.stat().st_mode), 0o644)
        self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o400)
        self.assertEqual(private.stat().st_nlink, 1)

    def test_sealing_detaches_shared_manifest_and_asset_inodes(self):
        from dobot_nexus.algorithms.rltoken import artifacts

        staging = self.root / "staging"
        shutil.copytree(self.source, staging)
        links = {}
        for rel in ("manifest.yaml", "deployment/assets/model_asset.bin"):
            src, dst = self.source / rel, staging / rel
            src.chmod(0o644)
            dst.unlink()
            os.link(src, dst)
            links[src] = (dst, src.read_bytes())
        verified = _verify_retained_feature_bundle(staging, self.policy.feature_spec)
        artifacts._seal_staging(staging, verified)
        for src, (dst, original) in links.items():
            self.assertEqual(src.read_bytes(), original)
            self.assertEqual(stat.S_IMODE(src.stat().st_mode), 0o644)
            self.assertFalse(os.path.samefile(src, dst))
            self.assertEqual(dst.stat().st_nlink, 1)
            self.assertEqual(stat.S_IMODE(dst.stat().st_mode), 0o444)

    def test_bundle_root_preserves_setgid_through_private_export(self):
        provider, config, _ = load_provider(self.source)
        root = self.root / "group-bundle"
        mkdir, chmod, lstat = os.mkdir, os.chmod, Path.lstat
        observed_mode = []

        def inherit_setgid(path, mode=0o777, **kwargs):
            mkdir(path, mode, **kwargs)
            if path == root:
                self.assertEqual(mode, 0o700)
                chmod(path, stat.S_IMODE(path.stat().st_mode) | stat.S_ISGID)
                observed_mode.append(stat.S_IMODE(path.stat().st_mode))

        def directory_stat(path):
            st = lstat(path)
            # Some filesystems clear setgid even on an explicit chmod. Supply the
            # inherited bit for this directory and an ordinary mode for the probe.
            if path == root:
                return SimpleNamespace(st_mode=st.st_mode | stat.S_ISGID)
            if path == root / ".mode-probe":
                return SimpleNamespace(st_mode=st.st_mode & ~stat.S_ISGID)
            return st

        def observe(*, source_weights, artifact_dir):
            self.assertEqual(stat.S_IMODE(artifact_dir.stat().st_mode), observed_mode[0])
            return {}

        with (patch("os.mkdir", side_effect=inherit_setgid),
              patch.object(Path, "lstat", directory_stat),
              patch("os.chmod", wraps=chmod) as changed,
              patch.object(provider, "export_deployment_artifacts", side_effect=observe)):
            write_model_bundle(root, provider, config)
        modes = [call.args[1] for call in changed.call_args_list if call.args[0] == root]
        self.assertEqual(modes, [stat.S_IMODE(root.stat().st_mode) | stat.S_ISGID])
        if observed_mode[0] & stat.S_ISGID:
            self.assertTrue(root.stat().st_mode & stat.S_ISGID)

    def test_all_provider_entry_points_validate_declared_and_effective_inputs(self):
        self.cfg.model.features.artifact = str(self.source)
        self.cfg.data.require_token_acceptance = False
        path = self.source / "manifest.yaml"
        valid = load_manifest(self.source)
        files = valid["files"]
        asset = self.source / files["model_assets"]["model.asset"]
        link = asset.with_name("symlink")
        link.symlink_to(asset)
        destination = self.root / "fresh"
        cases = [
            ({"files": {**files, "model_assets": {"model.asset": "../asset-2.bin"}}}, "file path"),
            (
                {
                    "files": {
                        **files,
                        "model_assets": {"model.asset": str(link.relative_to(self.source))},
                    }
                },
                "file path",
            ),
            ({"weights_sha256": "0" * 64}, "weights checksum"),
            ({"files": {**files, "model_assets": {}}}, "Undeclared"),
        ]
        for changes, error in cases:
            save_config({**valid, **changes}, path)
            for name, load in (
                ("provider", lambda: load_provider(self.source)),
                ("policy", lambda: RLTokenPolicy(self.cfg, mode="infer")),
                ("cache", lambda: RLTokenCacheTrainer(self.cfg, self.trainer.experiment).run()),
                ("retain", lambda: retain_feature_bundle(self.source, destination)),
            ):
                with (
                    self.subTest(entry=name, error=error),
                    self.assertRaisesRegex(ValueError, error),
                ):
                    load()
                self.assertFalse(destination.exists())

    def test_retain_checks_effective_assets_again_at_export(self):
        model_type = type(self.policy.provider)
        real = model_type.artifact_assets
        calls = []

        def assets(model):
            calls.append(model)
            if len(calls) == 1:
                return real(model)
            return {"model.asset": self.root / "asset-2.bin"}

        destination = self.root / "fresh"
        with patch.object(model_type, "artifact_assets", assets):
            with self.assertRaisesRegex(ValueError, "Undeclared"):
                retain_feature_bundle(self.source, destination)
        self.assertEqual(len(calls), 2)
        self.assertFalse(destination.exists())

    def test_logical_asset_cannot_be_redirected_to_another_declared_input(self):
        weights = self.source / load_manifest(self.source)["files"]["weights"]
        wrong = {"model.asset": weights}
        with (
            patch.object(type(self.policy.provider), "artifact_assets", return_value=wrong),
            self.assertRaisesRegex(ValueError, "Undeclared.*model.asset"),
        ):
            load_provider(self.source)
        self.cfg.model.features.artifact = str(self.source)
        policy = RLTokenPolicy(self.cfg, mode="infer")
        self.assertIsNone(policy._retained_root)
        with (
            patch.object(policy.provider, "artifact_assets", return_value=wrong),
            self.assertRaisesRegex(ValueError, "Undeclared.*model.asset"),
        ):
            policy.export_deployment_artifacts(
                source_weights=None, artifact_dir=self.root / "direct"
            )

    def test_provider_validation_reuses_only_the_verified_source_weight_digest(self):
        for root in (self.source, self.retained):
            weights = root / load_manifest(root)["files"]["weights"]
            with patch(
                "dobot_nexus.algorithms.rltoken.artifacts.file_digest", wraps=file_digest
            ) as hashed:
                if root == self.source:
                    retain_feature_bundle(root, self.root / "fresh")
                else:
                    RLTokenPolicy(self.cfg, mode="infer", retained_feature_root=root)
            self.assertEqual(sum(call.args == (weights,) for call in hashed.call_args_list), 1)

    def test_new_nexus_weights_inherit_backend_permission_ceiling(self):
        manifest = load_manifest(self.source)
        nexus = manifest["files"].pop("nexus_weights")
        (self.source / nexus).unlink()
        (self.source / manifest["files"]["weights"]).chmod(0o600)
        save_config(manifest, self.source / "manifest.yaml")
        retained = retain_feature_bundle(self.source, self.root / "fresh")
        self.assertEqual(stat.S_IMODE((retained / nexus).stat().st_mode), 0o400)
        self.assertEqual(retain_feature_bundle(self.source, retained), retained)
        (retained / nexus).chmod(0o444)
        with self.assertRaisesRegex(ValueError, "broader permissions"):
            retain_feature_bundle(self.source, retained)

    def test_run_seal_requires_one_or_two_complete_read_only_digests(self):
        path = self.retained.with_suffix(".sha256")
        digest = file_digest(self.retained / "manifest.yaml")
        for text, mode in (
            ("", 0o400),
            (digest + "\n\n", 0o400),
            ("g" * 64, 0o400),
            ((digest + "\n") * 3, 0o400),
            (digest, 0o600),
        ):
            path.chmod(0o600)
            path.write_text(text)
            path.chmod(mode)
            with self.subTest(text=text, mode=mode), self.assertRaisesRegex(ValueError, "seal"):
                retain_feature_bundle(self.source, self.retained)

    def test_manifest_mapping_is_bound_to_run_seal_and_enclosing_deployment(self):
        deployed = self.publish()
        manifest_path = self.retained / "manifest.yaml"
        child = deployed / "deployment/features/manifest.yaml"
        outer_path = deployed / "manifest.yaml"
        outer = load_manifest(deployed)
        digest = file_digest(child)
        self.assertEqual(outer["feature_manifest_sha256"], digest)
        self.assertTrue(child.samefile(manifest_path))
        modified = load_manifest(self.retained)
        del modified["files"]["model_assets"]["model.asset"]
        replacement = self.root / "changed.yaml"
        save_config(modified, replacement)
        replacement.replace(manifest_path)
        self.cfg.model.features.artifact = str(self.retained)
        with self.assertRaisesRegex(ValueError, "manifest.*checksum"):
            retain_feature_bundle(self.source, self.retained)
        with self.assertRaisesRegex(ValueError, "manifest.*checksum"):
            RLTokenPolicy(self.cfg, mode="infer")
        with self.assertRaisesRegex(ValueError, "manifest.*checksum"):
            retain_feature_bundle(self.retained, self.root / "new-run")

        save_config(modified, replacement)
        replacement.replace(child)
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            self.actions(deployed)
        del outer["feature_manifest_sha256"]
        save_config(outer, outer_path)
        with self.assertRaisesRegex(ValueError, "Missing independent"):
            self.actions(deployed)

    def test_missing_or_symlinked_run_seal_rejected(self):
        seal = self.retained.with_suffix(".sha256")
        backup = seal.rename(self.root / "real-seal")
        self.cfg.model.features.artifact = str(self.retained)
        for state in ("missing", "symlink"):
            if state == "symlink":
                seal.symlink_to(backup)
            with self.subTest(state=state), self.assertRaises(ValueError):
                retain_feature_bundle(self.source, self.retained)
            with self.subTest(state=state), self.assertRaises(ValueError):
                RLTokenPolicy(self.cfg, mode="infer")

    def test_direct_deployment_feature_source_uses_its_enclosing_manifest(self):
        deployed = self.publish()
        child = deployed / "deployment/features"
        self.cfg.model.features.artifact = str(child)
        direct = RLTokenPolicy(self.cfg, mode="infer")
        self.assertEqual(direct._retained_root, child.resolve())
        new_run = retain_feature_bundle(child, self.root / "next-run/features")
        self.assertEqual(
            file_digest(new_run / "manifest.yaml"),
            new_run.with_suffix(".sha256").read_text().strip(),
        )

        outer = load_manifest(deployed)
        del outer["feature_manifest_sha256"]
        save_config(outer, deployed / "manifest.yaml")
        with self.assertRaisesRegex(ValueError, "Missing independent"):
            RLTokenPolicy(self.cfg, mode="infer")
        with self.assertRaisesRegex(ValueError, "Missing independent"):
            retain_feature_bundle(child, new_run)

    def test_direct_child_cannot_downgrade_to_an_unpinned_legacy_manifest(self):
        deployed = self.publish()
        child = deployed / "deployment/features"
        manifest = load_manifest(child)
        del manifest["retained_files_sha256"]
        replacement = self.root / "old.yaml"
        save_config(manifest, replacement)
        replacement.replace(child / "manifest.yaml")
        self.cfg.model.features.artifact = str(child)
        for load in (
            lambda: load_provider(child),
            lambda: RLTokenPolicy(self.cfg, mode="infer"),
            lambda: retain_feature_bundle(child, self.root / "fresh"),
        ):
            with self.assertRaisesRegex(ValueError, "Deployment feature manifest checksum"):
                load()

        directory = child.parent.rename(deployed / "moved")
        child.parent.symlink_to(directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "deployment feature directory"):
            load_provider(child)

    def test_cross_run_resume_authenticates_source_and_destination(self):
        role = LearnerRole(
            self.cfg, self.trainer.run_id, self.trainer.epoch, self.trainer.task_fingerprint, None
        )
        try:
            role.initialize(SimpleNamespace(device="cpu", channels=None))
            checkpoint = role.snapshot()
        finally:
            role.close()
        pin = file_digest(self.retained / "manifest.yaml")
        copy = retain_feature_bundle(
            self.retained, self.root / "copy", expected_manifest_sha256=pin
        )
        self.assertEqual(file_digest(copy / "manifest.yaml"), pin)
        self.assertEqual(
            retain_feature_bundle(self.retained, copy, expected_manifest_sha256=pin), copy
        )

        other_run = self.root / "run-b"
        other_source = shutil.copytree(self.source, other_run / "source")
        source_index = load_manifest(other_source)
        asset = source_index["files"]["model_assets"]["model.asset"]
        (other_source / asset).write_bytes(b"run B calibration")
        source_index["training"] = {"step": 2}
        save_config(source_index, other_source / "manifest.yaml")
        other = retain_feature_bundle(other_source, other_run / "rltoken/features")
        self.assertEqual(feature_metadata(other)[1], feature_metadata(self.retained)[1])
        self.assertNotEqual(file_digest(other / asset), file_digest(self.retained / asset))
        self.assertNotEqual(
            load_manifest(other)["source_deployment_sha256"],
            load_manifest(self.retained)["source_deployment_sha256"],
        )
        for source in (self.source, self.retained):
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(ValueError, "configured feature source"),
            ):
                retain_feature_bundle(source, other)

        exp = SimpleNamespace(run_dir=other_run, logger=self.trainer.experiment.logger)
        online = clone_config(self.cfg)
        configs = Path(__file__).resolve().parents[3] / "examples/configs/rltoken"
        prewarm = load_config(configs / "td3_prewarm.yaml")
        for cfg, trainer in ((online, RLTokenTrainer), (prewarm, RLTokenPrewarmTrainer)):
            cfg.train.resume_from = str(checkpoint)
            cfg.exp.run_dir = str(other_run)
            kwargs = {"runtime_factory": LocalRuntime} if cfg is online else {}
            with (
                self.subTest(trainer=trainer),
                self.assertRaisesRegex(ValueError, "does not match checkpoint"),
            ):
                trainer(cfg, exp, **kwargs).run()

    def test_resolved_provider_and_deployment_configs_cannot_extend_external_yaml(self):
        deployed = self.publish()
        external = self.root / "external.yaml"
        external.write_text("train: {}\n")
        for root in (self.source, deployed):
            path = root / "config_resolved.yaml"
            path.write_text(f"extends: {external}\n" + path.read_text())
        infer = _wrap(dict(model=dict(artifact=str(deployed))))
        for load in (
            lambda: load_provider(self.source),
            lambda: retain_feature_bundle(self.source, self.root / "fresh"),
            lambda: retain_feature_bundle(self.source, self.retained),
            lambda: restore_policy_config(infer),
        ):
            with self.assertRaisesRegex(ValueError, "extends"):
                load()

        with (
            patch("builtins.open", wraps=open) as builtin_open,
            patch.object(Path, "open", autospec=True, side_effect=Path.open) as path_open,
            self.assertRaisesRegex(ValueError, "Resolved config cannot use extends"),
        ):
            InferencePolicy(infer)
        opened = {Path(call.args[0]) for call in (*builtin_open.call_args_list,
                                                   *path_open.call_args_list)}
        self.assertIn(deployed / "config_resolved.yaml", opened)
        self.assertNotIn(external, opened)

    def test_online_and_prewarm_resume_verify_checkpoint_feature_manifest_digest(self):
        role = LearnerRole(
            self.cfg, self.trainer.run_id, self.trainer.epoch, self.trainer.task_fingerprint, None
        )
        try:
            role.initialize(SimpleNamespace(device="cpu", channels=None))
            checkpoint = Path(role.snapshot())
        finally:
            role.close()
        feature = self.retained / "manifest.yaml"
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        self.assertEqual(manifest["feature_manifest_sha256"], file_digest(feature))
        modified = load_manifest(self.retained)
        del modified["files"]["model_assets"]["model.asset"]
        replacement = self.root / "changed.yaml"
        save_config(modified, replacement)
        replacement.replace(feature)
        # Even if the run seal now agrees, the saved checkpoint binds its original bytes.
        replacement.write_text(file_digest(feature))
        replacement.chmod(0o400)
        replacement.replace(self.retained.with_suffix(".sha256"))
        self.cfg.train.resume_from = str(checkpoint)
        with self.assertRaisesRegex(ValueError, "Resume feature manifest checksum"):
            RLTokenTrainer(self.cfg, self.trainer.experiment)
        configs = Path(__file__).resolve().parents[3] / "examples/configs/rltoken"
        prewarm = load_config(configs / "td3_prewarm.yaml")
        prewarm.train.resume_from = str(checkpoint)
        with self.assertRaisesRegex(ValueError, "Resume feature manifest checksum"):
            RLTokenPrewarmTrainer(prewarm, self.trainer.experiment).run()

    def test_0600_asset_stays_owner_read_only_under_umask_022(self):
        files = load_manifest(self.source)["files"]
        private_files = (
            files["model_assets"]["model.asset"],
            files["weights"],
            files["nexus_weights"],
            files["resolved_config"],
            "manifest.yaml",
        )
        for relative in private_files:
            (self.source / relative).chmod(0o600)
        previous = os.umask(0o022)
        try:
            retained = retain_feature_bundle(self.source, self.root / "private-asset")
            for relative in private_files:
                self.assertEqual(stat.S_IMODE((retained / relative).stat().st_mode), 0o400)
            self.cfg.model.features.artifact = str(retained)
            self.role.engine.policy = RLTokenPolicy(
                self.cfg, mode="infer", retained_feature_root=retained
            )
            with patch("os.link", side_effect=OSError(errno.EXDEV, "link")):
                deployed = self.publish()
            for relative in private_files:
                copied = deployed / "deployment/features" / relative
                self.assertEqual(stat.S_IMODE(copied.stat().st_mode), 0o400)
        finally:
            os.umask(previous)

    def test_failed_snapshot_preserves_history_and_cleans_staging(self):
        role = LearnerRole(
            self.cfg, self.trainer.run_id, self.trainer.epoch, self.trainer.task_fingerprint, None
        )
        role.initialize(SimpleNamespace(device="cpu", channels=None))
        self.addCleanup(role.close)
        pinned = self.role.feature_manifest_sha256()
        old = Path(role.snapshot(feature_manifest_sha256=pinned))
        current = Path(role.snapshot(feature_manifest_sha256=pinned))
        root = current.parent
        pointer = root / "latest_resume"
        self.cfg.train.keep_last_checkpoints = 1
        feature = self.retained / "manifest.yaml"
        original = feature.rename(self.root / "manifest.yaml")

        def rejects(error):
            with self.assertRaises(error):
                role.snapshot(feature_manifest_sha256=pinned)
            self.assertEqual(pointer.read_text(), current.name)
            self.assertEqual(set(root.iterdir()), {old, current, pointer, root / ".published.json"})

        rejects(FileNotFoundError)
        feature.symlink_to(original)
        rejects(ValueError)
        feature.unlink()
        modified = load_manifest(self.source)
        save_config(modified, feature)
        rejects(ValueError)
        # Even changing the independent run seal cannot change the digest pinned at startup.
        seal = self.retained.with_suffix(".sha256")
        seal.unlink()
        seal.write_text(file_digest(feature) + "\n")
        seal.chmod(0o400)
        rejects(ValueError)
        with patch("torch.save", side_effect=OSError("injected write failure")):
            rejects(OSError)

    def test_interrupted_successful_deployment_pointer_remains_loadable(self):
        old = self.publish()
        original = Path.replace

        events = []

        def sync(path):
            if path == self.pointer.parent:
                events.append("sync")
            _sync_dir(path)

        def interrupt_after_replace(path, target):
            original(path, target)
            if target == self.pointer:
                events.append("replaced")
                raise KeyboardInterrupt

        with (
            patch.object(Path, "replace", interrupt_after_replace),
            patch("dobot_nexus.algorithms.rltoken.training._pointer_state", return_value="unknown"),
            patch("dobot_nexus.algorithms.rltoken.training._sync_dir", sync),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.publish()
        self.assertEqual(events, ["replaced", "sync"])
        active = self.pointer.parent / self.pointer.read_text()
        self.assertNotEqual(active, old)
        self.assertTrue((self.actions(self.pointer) == self.actions(old)).all())

    def test_late_rollout_export_never_displaces_previous_success(self):
        self.cfg.train.keep_last_deployments = 2
        previous = self.publish()
        staged = threading.Event()
        finish = threading.Event()
        worker = None

        def pause_after_staging(path):
            _sync_dir(path)
            if path.parent == previous.parent and path.suffix == ".staging":
                staged.set()
                self.assertTrue(finish.wait(10))

        def timeout(method, path):
            nonlocal worker
            self.assertEqual(method, "export_deployment")
            pending = Path(path)
            self.assertEqual(pending.with_suffix(".reserve").read_text(), self.trainer.epoch)
            worker = executor.submit(self.role.export_deployment, pending)

            def wait():
                self.assertTrue(staged.wait(10))
                raise TimeoutError("worker still exporting")

            return SimpleNamespace(wait=wait)

        with (
            ThreadPoolExecutor(max_workers=1) as executor,
            patch("dobot_nexus.algorithms.rltoken.roles._sync_dir", pause_after_staging),
        ):
            try:
                with self.assertRaises(TimeoutError):
                    self.trainer._publish_deployment(SimpleNamespace(call=timeout), self.version)
                self.assertFalse(list(previous.parent.glob("*.reserve")))
                prune_deployments(self.pointer.parent, 2, current_epoch="new-epoch")
                self.assertEqual(len(list(previous.parent.glob("*.staging"))), 1)
                self.assertEqual(len(list(previous.parent.glob("*.stage.lock"))), 1)
            finally:
                finish.set()
            with self.assertRaisesRegex(RuntimeError, "reservation missing or cancelled"):
                worker.result(timeout=10)
        self.assertEqual(set(previous.parent.iterdir()), {previous, self.history})
        self.assertEqual(self.pointer.parent / self.pointer.read_text(), previous)
        active = self.publish()
        self.assertEqual(set(active.parent.glob("actor-*")), {active, previous})
        self.assertTrue((self.actions(active) == self.actions(previous)).all())

    def test_crashed_coordinator_reservation_expires_before_old_worker_finishes(self):
        active = self.publish()
        pending = active.parent / ".pending-actor-1-deadc0de"
        reserve = pending.with_suffix(".reserve")
        before = set(active.parent.iterdir())
        reserve.write_text(self.trainer.epoch)
        prune_deployments(self.pointer.parent, 1, current_epoch=self.trainer.epoch)
        self.assertEqual(set(active.parent.iterdir()), before)
        with self.assertRaisesRegex(RuntimeError, "reservation missing or cancelled"):
            self.role.export_deployment(pending)
        self.assertEqual(set(active.parent.iterdir()), before)

    def test_distinct_live_coordinators_can_overlap_publication(self):
        self.cfg.train.keep_last_deployments = 2
        cfg = clone_config(self.cfg)
        b = RLTokenTrainer(cfg, self.trainer.experiment)
        self.assertNotEqual(b.epoch, self.trainer.epoch)
        rollout = RolloutRole(cfg, b.run_id, b.epoch)
        rollout.initialize(SimpleNamespace(channels=None, device="cpu"))
        self.addCleanup(rollout.close)
        version = ActorVersion(b.run_id, b.epoch, 1)
        policy = rollout.engine.policy
        rollout.engine.weights.apply(ActorSnapshot.capture(
            policy.actor, version, policy.feature_spec.feature_id
        ))
        published = []

        def call(method, path):
            reply = rollout.export_deployment(path)
            published.append(self.publish())
            self.assertEqual(Path(path).with_suffix(".reserve").read_text(), b.epoch)
            self.assertTrue(Path(path).is_dir())
            return SimpleNamespace(wait=lambda: reply)

        active = b._publish_deployment(SimpleNamespace(call=call), version)
        self.assertEqual(self.pointer.parent / self.pointer.read_text(), active)
        self.assertEqual(set(active.parent.iterdir()), {active, *published, self.history})

    def test_prune_claims_only_dead_worker_staging_and_reservations(self):
        import fcntl

        active = self.publish()
        root = active.parent
        before = set(root.iterdir())
        pending = root / ".pending-actor-1-deadc0de"
        staging = pending.with_suffix(".staging")
        lock = pending.with_suffix(".stage.lock")
        reserve = pending.with_suffix(".reserve")
        for live in (True, False):
            with self.subTest(live=live):
                reserve.write_text("dead-coordinator")
                staging.mkdir()
                (staging / "weights").write_bytes(b"partial")
                with lock.open("w") as fd:
                    if live:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    prune_deployments(self.pointer.parent, 1)
                    self.assertFalse(reserve.exists())
                    self.assertEqual(staging.is_dir(), live)
                    self.assertEqual(lock.is_file(), live)
                prune_deployments(self.pointer.parent, 1)
                self.assertEqual(set(root.iterdir()), before)
        # Death immediately before mkdir or after releasing the lock also recovers.
        lock.touch()
        prune_deployments(self.pointer.parent, None)
        staging.mkdir()
        prune_deployments(self.pointer.parent, None)
        self.assertEqual(set(root.iterdir()), before)

    def test_interrupted_rpc_releases_reservation(self):
        class GetTimeoutError(Exception):
            pass

        active = self.publish()
        before = set(active.parent.iterdir())
        for error in (GetTimeoutError(), KeyboardInterrupt()):
            with self.subTest(error=error), self.assertRaises(type(error)):
                self.trainer._publish_deployment(Mock(call=Mock(side_effect=error)), self.version)
            self.assertEqual(set(active.parent.iterdir()), before)

    def test_symlinked_deployments_directory_fails_before_rpc_or_lock(self):
        deployments = self.pointer.parent / "deployments"
        deployments.symlink_to(self.pointer.parent, target_is_directory=True)
        rollout = Mock()
        with self.assertRaisesRegex(ValueError, "real directory"):
            self.trainer._publish_deployment(rollout, self.version)
        rollout.call.assert_not_called()
        with self.assertRaisesRegex(ValueError, "real directory"), _publication_lock(deployments):
            self.fail("Acquired a symlinked lock")

    def test_deployment_interrupt_during_pointer_sync_retries_before_escape(self):
        old = self.publish()
        attempts = []

        def sync(path):
            if path == self.pointer.parent and path / self.pointer.read_text() != old:
                attempts.append(self.pointer.read_text())
                if len(attempts) == 1:
                    raise KeyboardInterrupt
            _sync_dir(path)

        with (
            patch("dobot_nexus.algorithms.rltoken.training._sync_dir", sync),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.publish()
        self.assertEqual(attempts, [self.pointer.read_text()] * 2)
        self.assertTrue((self.actions(self.pointer) == self.actions(old)).all())

    def test_overlapping_exports_retain_previous_publication(self):
        self.cfg.train.keep_last_deployments = 2
        published = []

        def finish_earlier(method, path):
            reply = self.role.export_deployment(path)
            os.utime(path, ns=(1, 1))
            # B publishes while A's acknowledgment is still in flight.
            export = self.role.export_deployment

            def finish_later(pending):
                result = export(pending)
                future = 2_000_000_000_000_000_000
                os.utime(pending, ns=(future, future))
                return result

            with patch.object(self.role, "export_deployment", finish_later):
                published.append(self.publish())
            return SimpleNamespace(wait=lambda: reply)

        self.trainer._publish_deployment(SimpleNamespace(call=finish_earlier), self.version)
        second = self.pointer.parent / self.pointer.read_text()
        self.assertGreater(second.stat().st_mtime_ns, published[0].stat().st_mtime_ns)
        third = self.publish()
        self.assertEqual(set(third.parent.iterdir()), {second, third, self.history})

    def test_pending_symlink_rejected_before_promoting_export(self):
        previous = self.publish()

        def reply(path):
            Path(path).symlink_to(previous, target_is_directory=True)
            return dict(
                path=path,
                run_id=self.version.run_id,
                learner_epoch=self.version.learner_epoch,
                actor_update=self.version.actor_update,
            )

        with (
            patch.object(self.role, "export_deployment", reply),
            self.assertRaisesRegex(ValueError, "real pending"),
        ):
            self.publish()
        self.assertEqual(set(previous.parent.iterdir()), {previous, self.history})
        self.assertEqual(self.pointer.parent / self.pointer.read_text(), previous)

    def test_deployment_history_discards_candidate_abandoned_before_pointer_swap(self):
        self.cfg.train.keep_last_deployments = 2
        a = self.publish()
        b = shutil.copytree(a, a.parent / "actor-999-abcdef12")
        future = a.stat().st_mtime_ns + 10**12
        os.utime(b, ns=(future, future))
        self.assertEqual(json.loads(self.history.read_text()), [a.name])
        c = self.publish()
        self.assertEqual(set(c.parent.iterdir()), {a, c, self.history})
        self.assertEqual(json.loads(self.history.read_text()), [a.name, c.name])
        self.assertTrue((self.actions(self.pointer) == self.actions(a)).all())

    def test_deployment_history_recovers_pointer_swapped_before_history_write(self):
        self.cfg.train.keep_last_deployments = 2
        a = self.publish()
        with (
            patch("dobot_nexus.algorithms.rltoken.training._write_published_history",
                  side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.publish()
        b = self.pointer.parent / self.pointer.read_text()
        self.assertNotEqual(a, b)
        self.assertEqual(json.loads(self.history.read_text()), [a.name])
        c = self.publish()
        self.assertEqual(set(c.parent.iterdir()), {b, c, self.history})
        self.assertEqual(json.loads(self.history.read_text()), [b.name, c.name])
        self.assertTrue((self.actions(self.pointer) == self.actions(b)).all())

    def test_durable_deployment_survives_history_io_error(self):
        for keep_last in (None, 1):
            with self.subTest(keep_last=keep_last):
                self.cfg.train.keep_last_deployments = keep_last
                previous = self.publish()
                old_history = self.history.read_bytes()
                with (
                    patch("dobot_nexus.algorithms.rltoken.training._write_published_history",
                          side_effect=OSError(errno.EIO, "history write")),
                    self.assertLogs("dobot_nexus.algorithms.rltoken.checkpoint") as log,
                ):
                    actor = self.publish()
                self.assertIn("skipping retention", log.output[0])
                self.assertEqual(self.pointer.parent / self.pointer.read_text(), actor)
                self.assertEqual(self.history.read_bytes(), old_history)
                self.assertTrue((self.actions(previous) == self.actions(actor)).all())
                self.cfg.train.keep_last_deployments = 2
                final = self.publish()
                self.assertEqual(json.loads(self.history.read_text()), [actor.name, final.name])
                self.assertEqual(set(final.parent.iterdir()), {actor, final, self.history})

    def test_late_uncooperative_worker_is_reclaimed_during_next_publication(self):
        self.cfg.train.keep_last_deployments = 2
        previous = self.publish()
        paths = []

        def timeout(method, path):
            paths.append(Path(path))
            raise TimeoutError

        with self.assertRaises(TimeoutError):
            self.trainer._publish_deployment(SimpleNamespace(call=timeout), self.version)
        (late,) = paths
        self.assertFalse(late.with_suffix(".reserve").exists())
        late.mkdir()
        (late / "weights").write_bytes(b"late worker")
        active = self.publish()
        self.assertEqual(set(active.parent.iterdir()), {previous, active, self.history})
        self.assertEqual(self.pointer.parent / self.pointer.read_text(), active)

    def test_failed_deployment_prune_is_retried_on_next_publication(self):
        self.cfg.train.keep_last_deployments = 1
        old = self.publish()
        unrelated = old.parent / ".prune-old"
        unrelated.mkdir()
        link = old.parent / (".prune-" + "c" * 32)
        link.symlink_to(unrelated, target_is_directory=True)

        def fail_after_unlink(path):
            (path / "manifest.yaml").unlink()
            raise OSError("partial deletion")

        with (
            patch("shutil.rmtree", side_effect=fail_after_unlink),
            self.assertLogs("dobot_nexus.algorithms.rltoken.checkpoint", level="WARNING"),
        ):
            active = self.publish()
        (isolated,) = set(active.parent.glob(".prune-*")) - {link, unrelated}
        self.assertTrue(isolated.is_dir())
        final = self.publish()
        self.assertFalse(isolated.exists())
        self.assertTrue(link.is_symlink() and unrelated.is_dir())
        self.assertEqual(self.pointer.parent / self.pointer.read_text(), final)
        self.assertEqual(self.actions(final).shape, (1, 2, 2))

    def test_live_provider_fallback_without_retained_source(self):
        files = load_manifest(self.source)["files"]
        private_files = (
            files["weights"],
            files["nexus_weights"],
            files["resolved_config"],
            "manifest.yaml",
            files["model_assets"]["model.asset"],
        )
        for relative in private_files:
            (self.source / relative).chmod(0o600)
        self.cfg.model.features.artifact = str(self.source)
        policy = RLTokenPolicy(self.cfg, mode="infer")
        self.assertIsNone(policy._retained_root)
        self.role.engine.policy = policy
        with patch("os.link", side_effect=AssertionError("unbound source")):
            deployed = self.publish()
        self.assertEqual(self.actions(deployed).shape, (1, 2, 2))
        for relative in private_files:
            self.assertEqual(
                stat.S_IMODE((deployed / "deployment/features" / relative).stat().st_mode), 0o600
            )
        with (
            patch.object(
                policy.provider,
                "artifact_assets",
                return_value={"model.asset": self.root / "asset-2.bin"},
            ),
            self.assertRaisesRegex(ValueError, "Undeclared"),
        ):
            self.publish()
        self.assertEqual(self.pointer.parent / self.pointer.read_text(), deployed)
