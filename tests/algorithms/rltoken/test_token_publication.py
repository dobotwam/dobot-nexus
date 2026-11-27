import logging
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dobot_nexus.algorithms.rltoken.artifacts import load_provider
from dobot_nexus.algorithms.rltoken.serialization import load_payload
from dobot_nexus.algorithms.rltoken.token_pretraining import (
    TokenPretrainTrainer, _finish_publication, _publish_step, _resume_state,
)
from dobot_nexus.core.config import clone_config

from . import test_policy


class TokenPublicationTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        fixture = test_policy.PolicyTest()
        fixture.root = self.root
        self.model, self.cfg, _ = load_provider(fixture.make_config().model.features.artifact)

    def publish(self, step, parent):
        return _publish_step(self.root, self.model, self.cfg, {"step": step}, {}, True, expected_parent=parent)

    def test_stale_writer_cannot_publish_over_newer_generation(self):
        with self.assertRaises(FileNotFoundError):
            _resume_state(self.root, "auto", False)
        self.publish(10, None)
        _, _, parent_a = _resume_state(self.root, "auto", False)
        _, _, parent_b = _resume_state(self.root, "auto", False)
        self.publish(20, parent_a)
        files = [self.root / name for name in ("last", "best", "resume.pt")]
        before = [p.read_bytes() for p in files]
        with self.assertRaisesRegex(ValueError, "Stale token writer"):
            self.publish(15, parent_b)
        self.assertEqual([p.read_bytes() for p in files], before)
        self.assertEqual(files[0].read_text(), "step-000000020")
        self.assertEqual(_resume_state(self.root, "auto", False)[0]["step"], 20)
        self.assertFalse((self.root / "step-000000015").exists())

    def test_recovery_preserves_explicit_resume_selection(self):
        parent = self.publish(10, None)
        explicit = self.root / "external.pt"
        shutil.copyfile(self.root / "resume.pt", explicit)
        replace = os.replace

        def interrupt(source, dest, **kwargs):
            if source == "last":
                raise OSError("interrupted pointer")
            return replace(source, dest, **kwargs)

        with patch.object(os, "replace", interrupt), self.assertRaisesRegex(OSError, "interrupted pointer"):
            self.publish(20, parent)
        self.assertEqual(stat.S_IMODE((self.root / ".publishing").stat().st_mode), 0o700)
        state, selected, _ = _resume_state(self.root, str(explicit), False)
        self.assertEqual((state["step"], selected), (10, str(explicit)))
        self.assertFalse((self.root / ".publishing").exists())
        self.assertEqual(_resume_state(self.root, "auto", False)[0]["step"], 20)

    def test_implicit_recovery_restricts_legacy_resume_even_when_trust_is_enabled(self):
        pending = self.root / ".publishing"
        pending.mkdir(mode=0o700)
        (self.root / "last").write_text("step-000000010")
        # A benign historical Python record needs explicit legacy pickle trust.
        torch.save({"step": 10, "record": SimpleNamespace(value=3)}, pending / "resume.pt")
        with patch("dobot_nexus.algorithms.rltoken.token_pretraining.load_payload", wraps=load_payload) as load:
            with self.assertRaisesRegex(ValueError, "Restricted RLToken load"):
                _resume_state(self.root, None, True)
            self.assertIs(load.call_args.kwargs["trusted_checkpoint"], False)
        self.assertFalse(pending.exists())
        self.assertEqual(_resume_state(self.root, "auto", True)[0]["record"].value, 3)

    def test_same_step_from_different_run_cannot_be_the_published_parent(self):
        self.publish(20, None)
        other = self.root / "other"
        other.mkdir()
        _publish_step(other, self.model, self.cfg, {"step": 20, "metrics": {"loss": 0.25}},
                      {}, True, expected_parent=None)
        other_state, _, parent = _resume_state(self.root, other / "resume.pt", False)
        local_state = load_payload(self.root / "resume.pt")
        self.assertNotEqual(other_state["generation"], local_state["generation"])
        files = [self.root / name for name in ("last", "best", "resume.pt")]
        before = [p.read_bytes() for p in files]
        with self.assertRaisesRegex(ValueError, "Stale token writer"):
            self.publish(21, parent)
        self.assertEqual([p.read_bytes() for p in files], before)
        self.assertFalse((self.root / "step-000000021").exists())

    def test_standalone_checkpoint_without_generation_can_seed_a_run(self):
        torch.save({"step": 10}, self.root / "resume.pt")
        _, _, parent = _resume_state(self.root, "auto", False)
        self.publish(11, parent)
        state, _, next_parent = _resume_state(self.root, "auto", False)
        self.assertEqual(state["step"], 11)
        self.assertTrue(state["generation"])
        self.assertNotEqual(next_parent, parent)

    def test_external_seed_starts_a_local_best_history(self):
        cfg = clone_config(self.cfg)
        cfg.task = "train"
        cfg.model.pop("artifact")
        cfg.model.pop("checkpoint")
        cfg.model.rltoken = {"training_mode": "token_only"}
        cfg.algorithm = clone_config({"token": {"train_episodes": [0], "validation_episodes": [1],
                                                "min_zero_ratio": 1.2, "min_shuffle_ratio": 1.2}})
        stats = self.root / "norm_stats.json"
        stats.write_text("{}")
        cfg.data.update(batch_size=2, norm_stats_path=str(stats))
        cfg.train.update(max_steps=1, save_every=1)
        model_type = type(self.model)
        training_step = model_type.training_step
        loss = 1.0

        def token_step(model, batch):
            output = training_step(model, batch)
            output.metrics.update(token_loss=loss, zero_ratio=2.0, shuffle_ratio=2.0)
            return output

        experiment = SimpleNamespace(run_dir=self.root / "source-run", logger=logging.getLogger(__name__))
        with patch.object(model_type, "training_step", token_step):
            source = TokenPretrainTrainer(cfg, experiment).run()
            self.assertEqual(load_payload(source.checkpoint)["best"], 1.0)
            experiment.run_dir = self.root / "destination"
            cfg.train.update(max_steps=2, resume_from=source.checkpoint)
            loss = 2.0
            seeded = TokenPretrainTrainer(cfg, experiment).run()
            root = experiment.run_dir / "token"
            self.assertEqual((root / "best").read_text(), "step-000000002")
            self.assertTrue((root / "step-000000002/manifest.yaml").is_file())
            self.assertEqual(load_payload(seeded.checkpoint)["best"], 2.0)

            cfg.train.update(max_steps=3, resume_from="auto")
            loss = 3.0
            TokenPretrainTrainer(cfg, experiment).run()
            self.assertEqual((root / "best").read_text(), "step-000000002")

    def test_shared_writable_pending_directory_is_rejected_before_any_move(self):
        pending = self.root / ".publishing"
        pending.mkdir(mode=0o700)
        (pending / "last").write_text("step-000000010")
        (pending / "resume.pt").write_bytes(b"staged")
        for mode in (0o720, 0o777):
            pending.chmod(mode)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "writable only"):
                _finish_publication(self.root)
            for name in ("last", "resume.pt"):
                self.assertTrue((pending / name).is_file())
                self.assertFalse((self.root / name).exists())

    def test_symlinked_pending_directory_is_rejected_before_moving_files(self):
        outside = self.root / "other-run"
        outside.mkdir()
        (outside / "resume.pt").write_bytes(b"other")
        (self.root / ".publishing").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "real directory"):
            _finish_publication(self.root)
        self.assertEqual((outside / "resume.pt").read_bytes(), b"other")
        self.assertFalse((self.root / "resume.pt").exists())

    def test_pending_tree_is_validated_before_first_rename(self):
        pending = self.root / ".publishing"
        bundle = pending / "artifact"
        bundle.mkdir(parents=True)
        (pending / "last").write_text("step-000000010")
        (pending / "resume.pt").write_bytes(b"resume")
        for link in (bundle / "weights", pending / "best"):
            with self.subTest(link=link):
                link.symlink_to(self.root / "missing")
                with self.assertRaises(ValueError):
                    _finish_publication(self.root)
                self.assertTrue(bundle.is_dir())
                self.assertFalse((self.root / "step-000000010").exists())
                link.unlink()
