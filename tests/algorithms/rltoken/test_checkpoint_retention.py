import errno
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from dobot_nexus.algorithms.rltoken.checkpoint import (
    _pointer_state,
    _sync_dir,
    load_resume,
    prune_checkpoints,
    prune_deployments,
    save_resume,
)
from dobot_nexus.algorithms.rltoken.configuration import HeadConfig, LearningConfig
from dobot_nexus.algorithms.rltoken.identity import content_digest
from dobot_nexus.algorithms.rltoken.learner import create_learner
from dobot_nexus.algorithms.rltoken.pretraining import RLTokenPrewarmTrainer
from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.training import validate_training
from dobot_nexus.core.config import load_config
from dobot_nexus.training.resume import capture_rng_state

from .fakes import feature_spec, transition


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def learner(self, td3=False):
        replay = Replay(self.root / ("td3" if td3 else "v1"), feature_spec(), 2)
        self.addCleanup(replay.close)
        heads = HeadConfig(2, (8,), residual={"hidden_dims": [8]} if td3 else None)
        config = LearningConfig(
            batch_size=2, warmup=1, critic_actor_ratio=1, td3={} if td3 else None
        )
        return create_learner(feature_spec(), heads, config, replay)

    def test_legacy_resume_payload_next_update_and_private_digest(self):
        for td3, fmt in [(False, "rltoken_resume_v1"), (True, "rltoken_td3_resume_v3")]:
            with self.subTest(fmt=fmt):
                learner = self.learner(td3)
                learner.update(1, incoming=[transition(i) for i in range(4)])
                rng = learner.replay.generator.get_state()
                with patch.object(
                    learner.replay, "sample", side_effect=AssertionError("deep copy")
                ):
                    learner.prefetch(2)
                borrowed = learner._checkpoint_state_dict()
                self.assertFalse(learner.prefetched)
                self.assertTrue(torch.equal(rng, learner.replay.generator.get_state()))
                public = learner.state_dict()
                self.assertEqual(content_digest(borrowed), content_digest(public))
                key = next(iter(public["replay"]["rows"]))
                self.assertIs(borrowed["replay"]["rows"][key][0], learner.replay.rows[key][0])
                self.assertIsNot(public["replay"]["rows"][key][0], learner.replay.rows[key][0])
                for module in ("actor", "target", "critic"):
                    tensor = next(iter(borrowed[module].values()))
                    tensor.zero_()
                self.assertEqual(content_digest(public), content_digest(learner.state_dict()))

                # A v1/v3 writer saved deep snapshots and the old Replay outcomes may
                # need reconstructing from the journal. No new writer is used here.
                state = public
                state["replay"].pop("episode_outcomes_version")
                state["replay"].pop("episode_outcomes")
                state.update(rng=capture_rng_state(), task_fingerprint="task", run_id="legacy")
                path = self.root / fmt
                path.mkdir()
                torch.save(state, path / "state.pt")
                (path / "manifest.json").write_text(
                    json.dumps(
                        {
                            "format": fmt,
                            "state_digest": content_digest(state),
                            "task_fingerprint": "task",
                        }
                    )
                )
                with patch.object(
                    learner.replay, "sample", side_effect=AssertionError("deep copy")
                ):
                    expected = learner.update()
                digest = content_digest(learner.state_dict())
                restored = self.learner(td3)
                with self.assertRaisesRegex(ValueError, "trusted"):
                    load_resume(path, restored, task_fingerprint="task")
                self.assertEqual(load_resume(path, restored, task_fingerprint="task",
                                             trusted_checkpoint=True), "legacy")
                self.assertEqual(restored.update(), expected)
                self.assertEqual(content_digest(restored.state_dict()), digest)

    def test_retention_values_checked_at_both_entry_points(self):
        configs = Path(__file__).resolve().parents[3] / "examples/configs/rltoken"
        online = load_config(configs / "td3_online.yaml")
        offline = load_config(configs / "td3_prewarm.yaml")
        for cfg, validate in [
            (online, validate_training),
            (offline, lambda c: RLTokenPrewarmTrainer(c, None)),
        ]:
            validate(cfg)
            for field in ("keep_last_checkpoints", "keep_last_deployments"):
                for value in [None, 1, 2]:
                    cfg.train[field] = value
                    validate(cfg)
                for value in [True, False, 0, -1, 2.0, "2"]:
                    cfg.train[field] = value
                    with (
                        self.subTest(field=field, value=value),
                        self.assertRaisesRegex(ValueError, "train." + field),
                    ):
                        validate(cfg)
                del cfg.train[field]

    def test_prune_only_managed_real_directories_and_protect_active_rollback(self):
        for checkpoints in (True, False):
            self.check_prune(checkpoints)

    def check_prune(self, checkpoints):
        if checkpoints:
            root = self.root / "checkpoints"
            pointer = root / "latest_resume"
            pattern = "step-{:09d}-{}"
            malformed = "step-00000001-12345abc"
            prefix = ""

            def prune(k, active):
                prune_checkpoints(root, k, active_name=active)
        else:
            root = self.root / "deployments"
            pointer = self.root / "latest_deployment.txt"
            pattern = "actor-{}-{}"
            malformed = "actor-01-12345abc"
            prefix = "deployments/"

            def prune(k, active):
                prune_deployments(self.root, k, active_relative=active)

        root.mkdir()
        active = pattern.format(1, "00000000")
        # Same timestamp ties resolve by step then name. A newer export of a
        # smaller step outranks both. The active rollback always has its own slot.
        names = [
            active,
            pattern.format(90, "00000001"),
            pattern.format(90, "00000002"),
            pattern.format(80, "00000003"),
            pattern.format(2, "00000004"),
        ]
        for name, stamp in zip(names, [10, 20, 20, 20, 30], strict=True):
            path = root / name
            path.mkdir()
            (path / "weights").write_bytes(b"weights")
            (path / "weights").chmod(0o444)
            os.utime(path, ns=(stamp, stamp))
        unrelated = [root / n for n in (".staging", ".prune-old", "other", malformed)]
        for path in unrelated:
            path.mkdir()
        link = root / pattern.format(1, "deadc0de")
        link.symlink_to(unrelated[0], target_is_directory=True)
        file = root / pattern.format(5, "deadbeef")
        file.touch()
        for wrong in [
            None,
            prefix + link.name,
            "../" + active,
            prefix + file.name,
            prefix + malformed,
            prefix + names[2],
        ]:
            if wrong is not None:
                pointer.write_text(wrong)
            with self.assertLogs("dobot_nexus.algorithms.rltoken.checkpoint", level="WARNING"):
                prune(1, prefix + active)
            self.assertTrue(all((root / name).is_dir() for name in names))
        pointer.write_text(prefix + active)
        prune(None, prefix + active)
        self.assertTrue(all((root / name).is_dir() for name in names))
        prune(3, prefix + active)
        self.assertEqual(
            {name for name in names if (root / name).exists()}, {active, names[2], names[4]}
        )
        prune(2, prefix + active)
        self.assertEqual({name for name in names if (root / name).exists()}, {active, names[4]})
        prune(1, prefix + active)
        self.assertEqual({name for name in names if (root / name).exists()}, {active})
        self.assertTrue(link.is_symlink() and file.is_file() and all(p.is_dir() for p in unrelated))

    def test_failed_pointer_swap_and_partial_removal_leave_resume_usable(self):
        learner = self.learner()
        root = self.root / "checkpoints"

        def save(keep_last):
            return save_resume(
                root,
                learner,
                task_fingerprint="task",
                run_id="run",
                keep_last=keep_last,
            )

        old = save(1)
        with (
            patch.object(Path, "replace", side_effect=OSError("pointer")),
            self.assertRaises(OSError),
        ):
            save(1)
        self.assertEqual((root / "latest_resume").read_text(), old.name)
        self.assertEqual(set(root.glob("step-*")), {old})

        def fail_after_unlink(path):
            self.assertTrue(path.name.startswith(".prune-"))
            (path / "state.pt").unlink()
            raise OSError("injected unlink failure")

        with (
            patch("shutil.rmtree", side_effect=fail_after_unlink),
            self.assertLogs("dobot_nexus.algorithms.rltoken.checkpoint") as log,
        ):
            new = save(1)
        self.assertIn("injected unlink failure", log.output[0])
        self.assertEqual((root / "latest_resume").read_text(), new.name)
        self.assertEqual(set(root.glob("step-*")), {new})
        self.assertEqual(load_resume(new, learner, task_fingerprint="task"), "run")
        (leftover,) = root.glob(".prune-*")
        unrelated = root / ".prune-old"
        unrelated.mkdir()
        link = root / (".prune-" + "b" * 32)
        link.symlink_to(unrelated, target_is_directory=True)
        final = save(1)
        self.assertFalse(leftover.exists())
        self.assertTrue(unrelated.is_dir() and link.is_symlink())
        self.assertEqual((root / "latest_resume").read_text(), final.name)

    def test_resume_interrupt_after_pointer_replacement_keeps_loadable_candidate(self):
        learner = self.learner()
        root = self.root / "checkpoints"
        pointer = root / "latest_resume"
        previous = save_resume(root, learner, task_fingerprint="task", run_id="before")
        original = Path.replace
        events = []

        def sync(path):
            if path == root and events:
                events.append("sync")
            _sync_dir(path)

        def interrupt_after_replace(path, target):
            original(path, target)
            if target == pointer:
                events.append("replaced")
                raise KeyboardInterrupt

        with (
            patch.object(Path, "replace", interrupt_after_replace),
            patch(
                "dobot_nexus.algorithms.rltoken.checkpoint._pointer_state", return_value="unknown"
            ),
            patch("dobot_nexus.algorithms.rltoken.checkpoint._sync_dir", sync),
            self.assertRaises(KeyboardInterrupt),
        ):
            save_resume(root, learner, task_fingerprint="task", run_id="after", keep_last=1)
        self.assertEqual(events, ["replaced", "sync"])
        active = root / pointer.read_text()
        self.assertNotEqual(active, previous)
        self.assertEqual(load_resume(active, learner, task_fingerprint="task"), "after")
        self.assertTrue(previous.is_dir())

    def test_first_post_replacement_root_sync_is_retried_on_interrupt(self):
        learner = self.learner()
        root = self.root / "checkpoints"
        pointer = root / "latest_resume"
        previous = save_resume(root, learner, task_fingerprint="task", run_id="before")
        attempts = []

        def sync(path):
            if path == root and pointer.read_text() != previous.name:
                attempts.append(pointer.read_text())
                if len(attempts) == 1:
                    raise KeyboardInterrupt
            _sync_dir(path)

        with (
            patch("dobot_nexus.algorithms.rltoken.checkpoint._sync_dir", sync),
            self.assertRaises(KeyboardInterrupt),
        ):
            save_resume(root, learner, task_fingerprint="task", run_id="after", keep_last=1)
        self.assertEqual(attempts, [pointer.read_text()] * 2)
        self.assertEqual(
            load_resume(root / pointer.read_text(), learner, task_fingerprint="task"), "after"
        )
        self.assertTrue(previous.is_dir())

    def test_overlapping_checkpoint_writers_rank_by_publication(self):
        learner = self.learner()
        root = self.root / "checkpoints"

        def save(run):
            return save_resume(root, learner, task_fingerprint="task", run_id=run, keep_last=2)

        staged = []
        published = []

        def sync(path):
            _sync_dir(path)
            if path.parent == root and path.name.startswith(".step-"):
                staged.append(path)
                if len(staged) == 1:
                    os.utime(path, ns=(1, 1))
                    published.append(save("B"))
                else:
                    future = 2_000_000_000_000_000_000
                    os.utime(path, ns=(future, future))

        with patch("dobot_nexus.algorithms.rltoken.checkpoint._sync_dir", sync):
            second = save("A")
        self.assertGreater(second.stat().st_mtime_ns, published[0].stat().st_mtime_ns)
        third = save("C")
        self.assertEqual(set(root.glob("step-*")), {second, third})
        self.assertEqual(load_resume(second, learner, task_fingerprint="task"), "A")

    def test_checkpoint_history_recovers_both_pointer_crash_windows(self):
        learner = self.learner()

        def save(root, run):
            return save_resume(root, learner, task_fingerprint="task", run_id=run, keep_last=2)

        for swapped in (False, True):
            with self.subTest(pointer_swapped=swapped):
                root = self.root / str(swapped)
                a = save(root, "A")
                pointer = root / "latest_resume"
                history = root / ".published.json"
                if swapped:
                    with (
                        patch("dobot_nexus.algorithms.rltoken.checkpoint._write_published_history",
                              side_effect=KeyboardInterrupt),
                        self.assertRaises(KeyboardInterrupt),
                    ):
                        save(root, "B")
                    b = root / pointer.read_text()
                    self.assertNotEqual(a, b)
                else:
                    b = shutil.copytree(a, root / "step-999999999-abcdef12")
                    future = a.stat().st_mtime_ns + 10**12
                    os.utime(b, ns=(future, future))
                    self.assertEqual(pointer.read_text(), a.name)
                self.assertEqual(json.loads(history.read_text()), [a.name])
                c = save(root, "C")
                previous = b if swapped else a
                self.assertEqual(set(root.glob("step-*")), {previous, c})
                self.assertEqual(json.loads(history.read_text()), [previous.name, c.name])
                self.assertEqual(pointer.read_text(), c.name)
                self.assertEqual(
                    load_resume(previous, learner, task_fingerprint="task"), "B" if swapped else "A"
                )

    def test_durable_checkpoint_survives_history_io_error(self):
        learner = self.learner()
        for keep_last in (None, 1):
            with self.subTest(keep_last=keep_last):
                root = self.root / str(keep_last)
                kwargs = dict(task_fingerprint="task", run_id="before", keep_last=keep_last)
                previous = save_resume(root, learner, **kwargs)
                history = root / ".published.json"
                kwargs["run_id"] = "durable"
                with (
                    patch("dobot_nexus.algorithms.rltoken.checkpoint._write_published_history",
                          side_effect=OSError(errno.EIO, "history write")),
                    self.assertLogs("dobot_nexus.algorithms.rltoken.checkpoint") as log,
                ):
                    current = save_resume(root, learner, **kwargs)
                self.assertIn("skipping retention", log.output[0])
                self.assertEqual((root / "latest_resume").read_text(), current.name)
                self.assertEqual(load_resume(current, learner, task_fingerprint="task"), "durable")
                self.assertEqual(set(root.glob("step-*")), {previous, current})
                self.assertEqual(json.loads(history.read_text()), [previous.name])
                final = save_resume(root, learner, task_fingerprint="task", run_id="C", keep_last=2)
                self.assertEqual(json.loads(history.read_text()), [current.name, final.name])
                self.assertEqual(set(root.glob("step-*")), {current, final})

    def test_pointer_discard_requires_positive_evidence(self):
        pointer = self.root / "pointer"
        self.assertEqual(_pointer_state(pointer, "new"), "inactive")
        pointer.write_text("old")
        self.assertEqual(_pointer_state(pointer, "new"), "inactive")
        pointer.write_text("new\n")
        self.assertEqual(_pointer_state(pointer, "new"), "active")
        pointer.write_bytes(b"\xff")
        self.assertEqual(_pointer_state(pointer, "new"), "unknown")
        pointer.unlink()
        pointer.symlink_to(self.root / "missing")
        self.assertEqual(_pointer_state(pointer, "new"), "unknown")
