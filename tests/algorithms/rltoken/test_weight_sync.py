import copy
import unittest
from dataclasses import replace

import torch

from dobot_nexus.algorithms.rltoken.actor_critic import Actor
from dobot_nexus.algorithms.rltoken.configuration import HeadConfig
from dobot_nexus.algorithms.rltoken.weight_sync import ActorSnapshot, ActorVersion, WeightSlots

from .fakes import feature_spec


class WeightSyncTest(unittest.TestCase):
    def test_snapshots_are_independent_and_applied_versions_are_strict(self):
        actor = Actor(feature_spec(), HeadConfig(2, (8,)))
        slots = WeightSlots(copy.deepcopy(actor), "test-features-v1", "run", "epoch")
        with self.assertRaisesRegex(RuntimeError, "Initial"):
            with slots.request():
                pass
        version = ActorVersion("run", "epoch", 1)
        snapshot = ActorSnapshot.capture(actor, version, "test-features-v1")
        slots.apply(snapshot)
        with torch.no_grad():
            next(actor.parameters()).add_(1)
        with slots.request() as (serving, applied):
            self.assertEqual(applied, version)
            torch.testing.assert_close(
                next(serving.parameters()), next(iter(snapshot.state.values()))
            )
        self.assertEqual(slots.apply(snapshot), version)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            slots.apply(ActorSnapshot.capture(actor, version, snapshot.feature_id))
        with self.assertRaisesRegex(ValueError, "Stale"):
            slots.apply(
                ActorSnapshot.capture(actor, replace(version, actor_update=0), snapshot.feature_id)
            )
        with self.assertRaisesRegex(ValueError, "epoch"):
            slots.apply(
                ActorSnapshot.capture(
                    actor, replace(version, learner_epoch="old"), snapshot.feature_id
                )
            )
        with self.assertRaisesRegex(ValueError, "checksum"):
            slots.apply(replace(snapshot, digest="corrupt"))
