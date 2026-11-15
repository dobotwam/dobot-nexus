from __future__ import annotations

import dataclasses
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from dobot_nexus.algorithms.recap.labeling import run_returns
from dobot_nexus.algorithms.recap.value.processing import ReCapValueProcessor
from dobot_nexus.algorithms.recap.value.training import train_value
from tests.algorithms.recap.fakes import FakeSource, FakeTokenizer, FakeValue, offline_config


@unittest.skipUnless(importlib.util.find_spec('tensorboard'), 'tensorboard is not installed')
class ValueTensorBoardTest(unittest.TestCase):
    def test_value_training_and_resume_emit_scalars_without_changing_weights(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory))
            config = dataclasses.replace(config, tracking={'tensorboard': {'enabled': True}},
                        train=dataclasses.replace(config.train, max_steps=4, stop_steps=2,
                                                  save_every=2, log_every=2, val_episodes=1))
            run_returns(config)
            with mock.patch('dobot_nexus.algorithms.recap.value.dataset.build_source', FakeSource), \
                 mock.patch('dobot_nexus.algorithms.recap.value.training.build_value_model',
                            side_effect=lambda _: FakeValue()), \
                 mock.patch('dobot_nexus.algorithms.recap.value.checkpoint.build_value_model',
                            side_effect=lambda _: FakeValue()), \
                 mock.patch.object(ReCapValueProcessor, 'tokenizer', new_callable=mock.PropertyMock,
                                   return_value=FakeTokenizer()):
                first = train_value(config)
                resumed = dataclasses.replace(config, train=dataclasses.replace(
                    config.train, stop_steps=4, resume_from=first.checkpoint))
                result = train_value(resumed)
                reference_config = dataclasses.replace(config, tracking={}, train=dataclasses.replace(
                    config.train, stop_steps=4, output_dir=Path(directory)/'reference'))
                reference = train_value(reference_config)
            actual = torch.load(result.checkpoint, weights_only=True)['model']
            expected = torch.load(reference.checkpoint, weights_only=True)['model']
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
            events = EventAccumulator(str(config.train.output_dir/'tensorboard')).Reload()
            self.assertEqual([e.step for e in events.Scalars('value_loss')], [1, 2, 4])
            self.assertTrue({'value_loss', 'expected_value_mae', 'lr', 'grad_norm',
                             'step_seconds', 'data_seconds', 'val_loss', 'val_mae'}
                            <= set(events.Tags()['scalars']))
