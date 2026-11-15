"""CPU contracts between the training loop and optional metric trackers."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from dobot_nexus.core.config import Config, clone_config
from dobot_nexus.core.experiment import Experiment
from dobot_nexus.training.bc_trainer import BCTrainer


class TrainingTrackingTest(unittest.TestCase):
    def test_optimizer_steps_and_tracking_do_not_change_rng_or_weights(self):
        # Consume RNG in both initialization and logging, like an external SDK.
        tracker = mock.Mock()
        tracker.log.side_effect = lambda *args, **kwargs: torch.rand(1)

        def create_tracker(*args, **kwargs):
            torch.rand(1)
            return tracker

        with tempfile.TemporaryDirectory() as directory:
            config = clone_config({
                "schema_version": 1, "task": "train", "algorithm": "bc",
                "exp": {"name": "tracked", "output_dir": directory, "seed": 17},
                "model": {
                    "name": "dummy", "target": "tests.fakes:StochasticDummyVLAModule",
                    "checkpoint": "unused", "state_dim": 2, "action_dim": 2,
                    "action_horizon": 3,
                },
                "data": {"batch_size": 2, "num_workers": 0},
                "runtime": {"launcher": "local", "device": "cpu", "precision": "fp32"},
                "optim": {"lr": 0.01}, "scheduler": {"name": "constant"},
                "train": {
                    "max_steps": 4, "log_every": 2, "save_every": 4,
                    "grad_accum_steps": 2,
                },
            })
            experiment = Experiment.from_config(config)
            with mock.patch(
                "dobot_nexus.training.bc_trainer.build_tracker", side_effect=create_tracker
            ):
                result = BCTrainer(config, experiment).run()
            self.assertEqual([call.kwargs["step"] for call in tracker.log.call_args_list], [1, 2, 4])
            self.assertTrue(
                {"loss", "lr", "grad_norm", "steps_per_sec"} <= tracker.log.call_args.args[0].keys()
            )
            tracker.finish.assert_called_once_with(exit_code=0)
            config = clone_config(config)
            config.exp.name = "untracked"
            config.exp.pop("run_dir", None)
            reference = BCTrainer(config, Experiment.from_config(config)).run()
            weights = "deployment/pytorch/full_weights.pt"
            tracked = torch.load(Path(result.checkpoint) / weights, weights_only=True)
            untracked = torch.load(Path(reference.checkpoint) / weights, weights_only=True)
            for key in tracked:
                torch.testing.assert_close(tracked[key], untracked[key], atol=0, rtol=0)

    def test_failure_or_interrupt_closes_tracker_without_hiding_the_error(self):
        for failure in (RuntimeError("training failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                tracker = mock.Mock()
                tracker.finish.side_effect = RuntimeError("SDK finish failed")
                runtime = SimpleNamespace(is_main_process=True, end_training=mock.Mock())
                with (
                    tempfile.TemporaryDirectory() as directory,
                    mock.patch("dobot_nexus.training.bc_trainer.build_runtime", return_value=runtime),
                    mock.patch("dobot_nexus.training.bc_trainer.build_tracker", return_value=tracker),
                ):
                    experiment = SimpleNamespace(run_dir=directory, logger=mock.Mock())
                    trainer = BCTrainer(Config({}), experiment)
                    with mock.patch.object(trainer, "_run", side_effect=failure):
                        with self.assertRaises(type(failure)) as caught:
                            trainer.run()
                    self.assertIs(caught.exception, failure)
                    tracker.finish.assert_called_once_with(exit_code=1)
                    runtime.end_training.assert_called_once_with()

class TensorBoardTrainingTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('tensorboard'),
                         'tensorboard is not installed')
    def test_real_training_resume_writes_optimizer_step_curves(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tests.test_checkpoint_resume import _config

        with tempfile.TemporaryDirectory() as directory:
            def run(steps, resume=None):
                config = _config(directory, name='tensorboard', max_steps=steps,
                                 resume_from=resume, save_every=2)
                config.train.log_every = 2
                config.train.grad_accum_steps = 2
                config.scheduler.total_steps = 5
                config.tracking = clone_config({'tensorboard': {'enabled': True}})
                return BCTrainer(config, Experiment.from_config(config)).run()
            first = run(2)
            # Simulate a scalar recorded after the last durable optimizer checkpoint.
            from dobot_nexus.training.tracking import TensorBoardTracker
            tracker = TensorBoardTracker(SimpleNamespace(run_dir=first.run_dir), {})
            tracker.log({'loss': 999.}, step=3)
            tracker.finish()
            result = run(5, 'auto')
            events = EventAccumulator(str(Path(result.run_dir)/'tensorboard')).Reload()
            self.assertEqual([e.step for e in events.Scalars('loss')], [1, 2, 4, 5])
            self.assertTrue({'loss', 'lr', 'grad_norm', 'steps_per_sec', 'eta_seconds'}
                            <= set(events.Tags()['scalars']))
            self.assertEqual(events.Scalars('eta_seconds')[-1].value, 0.)
            self.assertTrue(all(e.value > 0 for e in events.Scalars('steps_per_sec')))


if __name__ == "__main__":
    unittest.main()
