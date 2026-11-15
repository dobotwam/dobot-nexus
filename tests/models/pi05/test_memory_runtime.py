import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from dobot_nexus.inference.session import SessionOptions
from dobot_nexus.models.pi05.memory.configuration import MemoryConfig
from dobot_nexus.models.pi05.memory.runtime import HistoryBuffer, MemorySession

FRAME_BYTES = 16*2048*4


class MemoryRuntimeTest(unittest.TestCase):
    def test_history_grows_lazily_and_retains_sliding_window(self):
        for window in (None, 3):
            buffer = HistoryBuffer(MemoryConfig(budget=32, window_frames=window))
            self.assertEqual(buffer.nbytes, 0)
            for i in range(8):
                buffer.append(np.full((16, 2048), i, np.float32), i)
                self.assertEqual(buffer.nbytes, min(i+1, window or 8192)*FRAME_BYTES)
                snapshot = buffer.snapshot()
                self.assertTrue(np.isfinite(snapshot.features).all())
            buffer.reset()
            self.assertEqual(buffer.nbytes, 0)

    def test_environment_history_and_batch_limits_precede_encoding(self):
        encoder = Mock(side_effect=lambda frames: np.zeros((len(frames), 16, 2048), np.float32))
        policy = SimpleNamespace(spec=SimpleNamespace(action_horizon=3))
        session = MemorySession(policy, encoder, MemoryConfig(budget=32), SessionOptions(1),
                                max_environments=1, max_history_bytes=FRAME_BYTES,
                                max_batch_frames=2)
        session.push_frames([0], [0], env_id='a', prompt='pick')
        for frames, steps, env in [([0], [1], 'a'), ([0], [0], 'b'),
                                   ([0, 0, 0], [1, 2, 3], 'a')]:
            with self.assertRaises(ValueError):
                session.push_frames(frames, steps, env_id=env, prompt='pick')
        self.assertEqual(encoder.call_count, 1)
        session.reset()
        session.push_frames([0], [0], env_id='b', prompt='pick')
        self.assertEqual(encoder.call_count, 2)
