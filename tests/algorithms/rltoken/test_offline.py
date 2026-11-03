import tempfile
import unittest
from pathlib import Path

import torch

from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.transitions import Finalization
from .fakes import feature_spec, transition


class OfflineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_finalization_is_durable_idempotent_and_rejects_unknown(self):
        replay = Replay(self.root/'journal', feature_spec(), 2)
        self.addCleanup(replay.close)
        row = transition(rewards=None, accepted=None, reward_finalized=False)
        self.assertEqual(replay.commit(row).status, 'pending')
        self.assertEqual(replay.budget, 0)
        label = Finalization(row.key, torch.tensor([0., 1.]), True)
        first = replay.commit(label)
        self.assertEqual(replay.commit(label), first)
        self.assertEqual(replay.budget, 1)
        self.assertFalse(replay.pending())
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            replay.commit(Finalization(row.key, torch.zeros(2), True))
        with self.assertRaisesRegex(ValueError, 'unknown'):
            replay.commit(Finalization(('unknown',), torch.zeros(2), True))
