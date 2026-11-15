from __future__ import annotations

import io
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from dobot_nexus.training.resume import capture_rng_state, restore_rng_state


class _Executable:
    def __reduce__(self):
        return (eval, ("1 + 1",))


class SafeCheckpointTest(unittest.TestCase):
    def test_model_loaders_reject_executable_pickle(self):
        from dobot_nexus.models.pi0.training import checkpoint as pi0
        from dobot_nexus.models.pi05.training import checkpoint as pi05
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weights.pt'
            torch.save({'state_dict': {'weight': _Executable()}}, path)
            with patch('builtins.eval', side_effect=AssertionError('executed')):
                for module in (pi0, pi05):
                    with self.subTest(module=module.__name__):
                        with self.assertRaises(pickle.UnpicklingError):
                            module._load_file(path)

    def test_rng_roundtrip_is_restricted_loadable_and_exact(self):
        state = capture_rng_state()
        expected = (random.random(), np.random.random(3), torch.rand(3))
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restore_rng_state(torch.load(buffer, weights_only=True))
        self.assertEqual(random.random(), expected[0])
        np.testing.assert_array_equal(np.random.random(3), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)

    def test_dcp_resume_requires_explicit_trust_before_reading_metadata(self):
        from dobot_nexus.training.runtime import _load_dcp_training_state
        with self.assertRaisesRegex(ValueError, 'trust_resume_checkpoint'):
            _load_dcp_training_state(model=None, optimizer=None, scheduler=None, scaler=None,
                checkpoint_dir='/missing', process_index=0, world_size=1)
