import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from dobot_nexus.algorithms.rltoken.configuration import HeadConfig, LearningConfig
from dobot_nexus.algorithms.rltoken.identity import content_digest
from dobot_nexus.algorithms.rltoken.learner import Learner
from dobot_nexus.algorithms.rltoken.replay import Replay

from .fakes import feature_spec, transition


class LearnerTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.counter = 0

    def learner(self, **kwargs):
        self.counter += 1
        config = LearningConfig(batch_size=2, warmup=2, updates_per_new_transition=4, **kwargs)
        replay = Replay(
            self.root / f"{self.counter}.sqlite",
            feature_spec(),
            2,
            capacity=8,
            updates_per_new_transition=4,
        )
        self.addCleanup(replay.close)
        return Learner(feature_spec(), HeadConfig(2, (8,)), config, replay)

    def test_admission_progresses_during_warmup_and_update_clocks_are_separate(self):
        learner = self.learner(critic_actor_ratio=2, target_update_frequency=2)
        before_target = copy.deepcopy(learner.target.state_dict())
        result = learner.update(incoming=[transition()])
        self.assertEqual((result["updates"], len(result["acks"])), (0, 1))
        result = learner.update(incoming=[transition(1)])
        self.assertEqual(
            (result["critic_updates"], result["actor_updates"], result["budget"]), (1, 0, 7)
        )
        self.assertEqual(content_digest(before_target), content_digest(learner.target.state_dict()))
        learner.update()
        self.assertEqual((learner.critic_updates, learner.actor_updates), (2, 1))
        self.assertEqual(
            (learner.critic_scheduler.last_epoch, learner.actor_scheduler.last_epoch), (2, 1)
        )
        self.assertNotEqual(
            content_digest(before_target), content_digest(learner.target.state_dict())
        )
        actor_ids = {
            id(p) for group in learner.actor_optimizer.param_groups for p in group["params"]
        }
        critic_ids = {
            id(p) for group in learner.critic_optimizer.param_groups for p in group["params"]
        }
        self.assertFalse(actor_ids & critic_ids)
        self.assertTrue(
            all(
                not p.requires_grad and id(p) not in actor_ids | critic_ids
                for p in learner.target.parameters()
            )
        )

    def test_optimizer_failure_prevents_snapshot_and_further_updates(self):
        learner = self.learner()
        with patch.object(learner.actor_optimizer, "step", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                learner.update(incoming=[transition(), transition(1)])
        self.assertTrue(learner.failed)
        with self.assertRaisesRegex(RuntimeError, "session failed"):
            learner.state_dict()
        with self.assertRaisesRegex(RuntimeError, "session failed"):
            learner.update()
