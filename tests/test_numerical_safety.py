import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dobot_nexus.core.artifacts import _validate_state_dict
from dobot_nexus.core.contracts import PolicyOutput, TrainOutput
from dobot_nexus.core.experiment import Experiment
from dobot_nexus.training import BCTrainer
from dobot_nexus.training.runtime import LocalRuntime
from tests.fakes import DummyVLAModule
from tests.test_checkpoint_resume import _config


class NumericalSafetyTest(unittest.TestCase):
    def test_nonfinite_loss_and_gradients_do_not_update_or_publish(self):
        original = DummyVLAModule.training_step
        for bad_gradient in (False, True):
            with self.subTest(bad_gradient=bad_gradient), tempfile.TemporaryDirectory() as root:
                config = _config(root, name='invalid', max_steps=1)
                def training_step(model, batch):
                    output = original(model, batch)
                    if bad_gradient:
                        output.loss.register_hook(lambda grad: grad * float('nan'))
                        return output
                    return TrainOutput(loss=output.loss * float('nan'))
                with patch.object(DummyVLAModule, 'training_step', training_step), \
                     patch('torch.optim.AdamW.step') as step:
                    with self.assertRaises(FloatingPointError):
                        BCTrainer(config, Experiment.from_config(config)).run()
                    step.assert_not_called()
                self.assertFalse(list(Path(root).rglob('manifest.yaml')))

    def test_any_rank_nonfinite_flag_stops_local_rank(self):
        runtime = LocalRuntime(_config('unused', name='invalid', max_steps=1))
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.all_reduce', side_effect=lambda value, **kw: value.zero_()):
            with self.assertRaises(FloatingPointError):
                runtime.require_finite(torch.tensor(1.), label='training loss')

    def test_invalid_deployment_weights_and_actions_are_rejected(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaisesRegex(RuntimeError, 'non-finite'):
                _validate_state_dict({'weight': torch.tensor(value)}, label='test')
            with self.assertRaisesRegex(ValueError, 'finite'):
                PolicyOutput(np.full((1, 2, 3), value))
        with self.assertRaisesRegex(ValueError, 'valid_mask'):
            PolicyOutput(np.zeros((1, 2, 3)), valid_mask=np.ones((1, 3), bool))
