import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dobot_nexus.core.config import Config
from dobot_nexus.core.experiment import Experiment
from dobot_nexus.training.bc_trainer import BCTrainer


class ExperimentTest(unittest.TestCase):
    def config(self, root):
        return Config(task="train", model=Config(name="dummy"),
                      exp=Config(name="shared", output_dir=root, auto_suffix=True))

    def test_rank_zero_shares_directory_before_any_worker_writes(self):
        with tempfile.TemporaryDirectory() as root:
            shared = []
            main = SimpleNamespace(is_main_process=True,
                                   broadcast_object=lambda value: (shared.append(value), value)[1])
            first = Experiment.from_config(self.config(root), runtime=main)
            worker = SimpleNamespace(is_main_process=False,
                                     broadcast_object=lambda value: shared.pop(0))
            with mock.patch("dobot_nexus.core.experiment.datetime") as clock, \
                 mock.patch("dobot_nexus.core.experiment.save_config") as save:
                second = Experiment.from_config(self.config(root), runtime=worker)
            clock.now.assert_not_called()
            save.assert_not_called()
            self.assertEqual(first.run_dir, second.run_dir)
            self.assertEqual(second.config.exp.run_dir, str(first.run_dir))
            self.assertTrue((second.run_dir / "config_resolved.yaml").is_file())
            self.assertEqual(len(list(Path(root).iterdir())), 1)

    def test_distributed_call_without_runtime_fails_before_creating_files(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, WORLD_SIZE="2"):
            with self.assertRaisesRegex(RuntimeError, "BCTrainer"):
                Experiment.from_config(self.config(root))
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_trainer_initializes_runtime_before_experiment(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = SimpleNamespace(is_main_process=True, broadcast_object=lambda value: value,
                                      end_training=mock.Mock())
            with mock.patch("dobot_nexus.training.bc_trainer.build_runtime", return_value=runtime):
                trainer = BCTrainer(self.config(root))
            self.assertIs(trainer.runtime, runtime)
            self.assertTrue((trainer.experiment.run_dir / "config_resolved.yaml").is_file())

    def test_directory_failure_is_reported_to_every_rank(self):
        with tempfile.TemporaryDirectory() as root:
            shared = []
            main = SimpleNamespace(is_main_process=True,
                                   broadcast_object=lambda value: (shared.append(value), value)[1])
            with mock.patch("dobot_nexus.core.experiment.save_config", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(RuntimeError, "disk full"):
                    Experiment.from_config(self.config(root), runtime=main)
            worker = SimpleNamespace(is_main_process=False,
                                     broadcast_object=lambda value: shared.pop(0))
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                Experiment.from_config(self.config(root), runtime=worker)
