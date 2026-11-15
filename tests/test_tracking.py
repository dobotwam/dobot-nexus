from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dobot_nexus.core.config import Config, clone_config
from dobot_nexus.training.tracking import (
    CompositeTracker,
    NullTracker,
    TensorBoardTracker,
    WandbTracker,
    build_tracker,
)


class TrackingTest(unittest.TestCase):
    def test_disabled_and_worker_ranks_use_null_tracker(self) -> None:
        experiment = SimpleNamespace(run_dir=Path("unused"))
        disabled = Config({"tracking": {"wandb": {"enabled": False}}})
        enabled = Config({"tracking": {"wandb": {"enabled": True}}})
        both = Config({"tracking": {"wandb": {"enabled": True}, "tensorboard": {"enabled": True}}})
        with mock.patch.dict(sys.modules, {"wandb": None, "torch.utils.tensorboard": None}):
            for config, main in ((Config({}), True), (disabled, True), (enabled, False)):
                tracker = build_tracker(config, experiment, is_main_process=main)
                self.assertIsInstance(tracker, NullTracker)
                tracker.log({"loss": 1.0}, step=1)
                tracker.finish()
            self.assertIsInstance(build_tracker(both, experiment, is_main_process=False), NullTracker)
            with self.assertRaisesRegex(ImportError, "tracking.*extra"):
                build_tracker(enabled, experiment, is_main_process=True)

    def test_tensorboard_dependency_error_precedes_wandb_start(self):
        config = clone_config({"tracking": {
            "tensorboard": {"enabled": True}, "wandb": {"enabled": True},
        }})
        with (
            mock.patch.dict(sys.modules, {"torch.utils.tensorboard": None}),
            mock.patch("dobot_nexus.training.tracking.WandbTracker") as wandb,
        ):
            with self.assertRaisesRegex(ImportError, "tracking,tensorboard"):
                build_tracker(config, SimpleNamespace(run_dir="unused"), is_main_process=True)
            wandb.assert_not_called()

    def test_tensorboard_and_wandb_sync_have_one_metric_writer_and_ordered_cleanup(self):
        calls = []
        run, writer = mock.Mock(), mock.Mock()
        run.finish.side_effect = lambda **kw: calls.append("finish")
        writer.close.side_effect = lambda: calls.append("close")

        def init(**kwargs):
            calls.append("init")
            return run

        def create_writer(**kwargs):
            calls.append("writer")
            return writer

        wandb = SimpleNamespace(init=mock.Mock(side_effect=init))
        config = clone_config({"tracking": {
            "wandb": {"enabled": True, "mode": "offline"},
            "tensorboard": {"enabled": True},
        }})
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(sys.modules, {
            "wandb": wandb,
            "torch.utils.tensorboard": SimpleNamespace(SummaryWriter=create_writer),
        }):
            tracker = build_tracker(config, SimpleNamespace(run_dir=directory), is_main_process=True)
            self.assertIsInstance(tracker, CompositeTracker)
            self.assertTrue(wandb.init.call_args.kwargs["sync_tensorboard"])
            tracker.log({"loss": 1.25}, step=42)
            writer.add_scalar.assert_called_once_with("loss", 1.25, global_step=42)
            run.log.assert_not_called()
            tracker.finish(exit_code=1)
            run.finish.assert_called_once_with(exit_code=1)
        self.assertEqual(calls, ["init", "writer", "close", "finish"])

    def test_sync_can_be_disabled_to_write_both_backends_independently(self):
        run, writer = mock.Mock(), mock.Mock()
        wandb = SimpleNamespace(init=mock.Mock(return_value=run))
        config = clone_config({"tracking": {
            "wandb": {"enabled": True, "sync_tensorboard": False},
            "tensorboard": {"enabled": True},
        }})
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(sys.modules, {"wandb": wandb}),
            mock.patch("dobot_nexus.training.tracking._summary_writer_type",
                       return_value=mock.Mock(return_value=writer)),
        ):
            tracker = build_tracker(config, SimpleNamespace(run_dir=directory), is_main_process=True)
            self.assertFalse(wandb.init.call_args.kwargs["sync_tensorboard"])
            tracker.log({"loss": 0.5}, step=3)
            writer.add_scalar.assert_called_once_with("loss", 0.5, global_step=3)
            run.log.assert_called_once_with({"loss": 0.5}, step=3)
            tracker.finish()
            writer.close.assert_called_once_with()
            run.finish.assert_called_once_with(exit_code=0)

    def test_sync_requires_tensorboard(self):
        config = clone_config({"tracking": {
            "wandb": {"enabled": True, "sync_tensorboard": True},
        }})
        experiment = SimpleNamespace(run_dir="unused")
        with self.assertRaisesRegex(ValueError, "requires tracking.tensorboard"):
            build_tracker(config, experiment, is_main_process=True)

    def test_writer_initialization_failure_finishes_wandb(self):
        config = clone_config({"tracking": {
            "wandb": {"enabled": True}, "tensorboard": {"enabled": True},
        }})
        experiment = SimpleNamespace(run_dir="unused")
        run = mock.Mock()
        with mock.patch("dobot_nexus.training.tracking.WandbTracker", return_value=run), (
            mock.patch("dobot_nexus.training.tracking._summary_writer_type",
                       return_value=mock.Mock(side_effect=OSError("cannot write")))
        ):
            with self.assertRaisesRegex(OSError, "cannot write"):
                build_tracker(config, experiment, is_main_process=True)
        run.finish.assert_called_once_with(exit_code=1)

    def test_composite_cleanup_attempts_all_trackers_after_a_failure(self):
        first, second = mock.Mock(), mock.Mock()
        first.finish.side_effect = OSError("cannot close")
        with self.assertRaisesRegex(OSError, "cannot close"):
            CompositeTracker(first, second).finish(exit_code=1)
        second.finish.assert_called_once_with(exit_code=1)

    @unittest.skipIf(importlib.util.find_spec("tensorboard") is None, "tensorboard is not installed")
    def test_tensorboard_only_writes_readable_local_events_without_wandb(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "dobot_nexus.training.tracking.WandbTracker", side_effect=AssertionError("W&B used")
        ):
            config = clone_config({"tracking": {"tensorboard": {"enabled": True}}})
            tracker = build_tracker(config, SimpleNamespace(run_dir=directory), is_main_process=True)
            self.assertIsInstance(tracker, TensorBoardTracker)
            tracker.log({"loss": 1.25, "lr": 0.01}, step=42)
            tracker.finish()
            events = EventAccumulator(str(Path(directory) / "tensorboard")).Reload()
            self.assertEqual(events.Scalars("loss")[0].step, 42)
            self.assertEqual(events.Scalars("loss")[0].value, 1.25)
            self.assertAlmostEqual(events.Scalars("lr")[0].value, 0.01)

    def test_rank_zero_initializes_logs_and_finishes_wandb(self) -> None:
        run = mock.Mock()
        wandb = SimpleNamespace(init=mock.Mock(return_value=run))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "test_run"
            run_dir.mkdir(parents=True)
            config = Config(
                {
                    "exp": Config({"name": "test_run", "seed": 7}),
                    "tracking": Config(
                        {
                            "wandb": Config(
                                {
                                    "enabled": True,
                                    "project": "robotics",
                                    "entity": "team",
                                    "group": "kbd",
                                    "mode": "offline",
                                    "tags": ["pi05", "34d"],
                                }
                            )
                        }
                    ),
                }
            )
            experiment = SimpleNamespace(run_dir=run_dir)
            with mock.patch.dict(sys.modules, {"wandb": wandb}):
                tracker = build_tracker(config, experiment, is_main_process=True)

            self.assertIsInstance(tracker, WandbTracker)
            kwargs = wandb.init.call_args.kwargs
            self.assertEqual(kwargs["project"], "robotics")
            self.assertEqual(kwargs["entity"], "team")
            self.assertEqual(kwargs["group"], "kbd")
            self.assertEqual(kwargs["name"], "test_run")
            self.assertEqual(kwargs["mode"], "offline")
            self.assertEqual(kwargs["tags"], ["pi05", "34d"])
            self.assertEqual(Path(kwargs["dir"]), run_dir / "wandb")
            self.assertEqual(kwargs["config"]["exp"]["seed"], 7)

            tracker.log({"loss": 1.25, "lr": 1e-5}, step=10)
            run.log.assert_called_once_with({"loss": 1.25, "lr": 1e-5}, step=10)
            tracker.finish(exit_code=0)
            run.finish.assert_called_once_with(exit_code=0)

class TensorBoardResumeTest(unittest.TestCase):
    def test_flush_queue_and_resume_settings(self):
        writer, replacement = mock.Mock(), mock.Mock()
        factory = mock.Mock(side_effect=[writer, replacement])
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            'dobot_nexus.training.tracking._summary_writer_type', return_value=factory
        ):
            tracker = TensorBoardTracker(SimpleNamespace(run_dir=directory),
                                         {'flush_secs': 3, 'max_queue': 8})
            tracker.log({'loss': 2}, step=3)
            writer.flush.assert_called_once_with()
            tracker.resume(2)
            writer.close.assert_called_once_with()
            self.assertEqual(factory.call_args.kwargs['purge_step'], 3)
            self.assertEqual(factory.call_args.kwargs['flush_secs'], 3)
            self.assertEqual(factory.call_args.kwargs['max_queue'], 8)
            tracker.finish()
            replacement.close.assert_called_once_with()
        for settings in ({'flush_secs': 0}, {'flush_secs': float('inf')},
                         {'flush_secs': True}, {'max_queue': False}, {'max_queue': 0}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                TensorBoardTracker(SimpleNamespace(run_dir='unused'), settings)

    @unittest.skipIf(importlib.util.find_spec('tensorboard') is None, 'tensorboard is not installed')
    def test_resume_purges_unsaved_steps_and_events_are_visible_before_close(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as directory:
            experiment = SimpleNamespace(run_dir=directory)
            tracker = TensorBoardTracker(experiment, {})
            for step in range(1, 5):
                tracker.log({'loss': float(step)}, step=step)
            events = EventAccumulator(str(Path(directory)/'tensorboard')).Reload()
            self.assertEqual([event.step for event in events.Scalars('loss')], [1, 2, 3, 4])
            tracker.finish()
            tracker = TensorBoardTracker(experiment, {})
            tracker.resume(2)
            tracker.log({'loss': 30.}, step=3)
            tracker.finish()
            events = EventAccumulator(str(Path(directory)/'tensorboard')).Reload()
            self.assertEqual([(event.step, event.value) for event in events.Scalars('loss')],
                             [(1, 1.), (2, 2.), (3, 30.)])

    def test_failure_closes_tracking_session_and_preserves_original_error(self):
        from dobot_nexus.training.tracking import tracking_session
        tracker = mock.Mock()
        tracker.finish.side_effect = OSError('writer failed to close')
        with mock.patch('dobot_nexus.training.tracking.build_tracker', return_value=tracker):
            with self.assertLogs('dobot_nexus.training.tracking', level='ERROR'):
                with self.assertRaisesRegex(RuntimeError, 'model failed'):
                    with tracking_session({}, SimpleNamespace(), is_main_process=True, resume_step=2):
                        raise RuntimeError('model failed')
        tracker.resume.assert_called_once_with(2)
        tracker.finish.assert_called_once_with(exit_code=1)

    def test_cuda_metrics_use_mib_and_do_not_query_cuda_for_cpu(self):
        from dobot_nexus.training.tracking import device_metrics
        with mock.patch('torch.cuda.memory_allocated', return_value=2*2**20) as allocated, \
             mock.patch('torch.cuda.memory_reserved', return_value=3*2**20) as reserved:
            self.assertEqual(device_metrics(SimpleNamespace(type='cpu')), {})
            allocated.assert_not_called()
            device = SimpleNamespace(type='cuda')
            self.assertEqual(device_metrics(device), {'system/cuda_allocated_mib': 2.,
                                                      'system/cuda_reserved_mib': 3.})
            allocated.assert_called_once_with(device)
            reserved.assert_called_once_with(device)


if __name__ == "__main__":
    unittest.main()
