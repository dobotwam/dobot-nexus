import unittest

import torch

from dobot_nexus.algorithms.rltoken.actor_critic import Actor, TwinQ
from dobot_nexus.algorithms.rltoken.configuration import HeadConfig, LearningConfig
from dobot_nexus.algorithms.rltoken.losses import actor_loss, critic_loss, td_target

from .fakes import feature_spec


class LossTest(unittest.TestCase):
    def test_actor_uses_q1_and_the_same_action_sample_for_bc(self):
        class DirectActor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(0.2))
                self.calls = 0

            def forward(self, *args, **kwargs):
                self.calls += 1
                return self.value.expand(1, 2, 1)

        class DirectQ(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, latent, state, actions):
                q1 = actions.sum((1, 2)) * self.scale
                return torch.stack((q1, -100 * q1), -1)

        actor, critic = DirectActor(), DirectQ()
        batch = {
            "latent": None,
            "state": None,
            "reference_condition": None,
            "bc_target": torch.zeros(1, 2, 1),
        }
        loss, metrics = actor_loss(actor, critic, batch, LearningConfig())
        loss.backward()
        self.assertAlmostEqual(loss.item(), -0.4 + 0.04, places=6)
        self.assertAlmostEqual(actor.value.grad.item(), -2 + 0.4, places=6)
        self.assertEqual(actor.calls, 1)
        self.assertIsNone(critic.scale.grad)

    def setUp(self):
        torch.set_num_threads(1)

    def test_chunk_discount_and_terminal_subset(self):
        result = td_target(
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
            torch.tensor([True, False]),
            torch.tensor([4.0]),
            0.5,
        )
        torch.testing.assert_close(result, torch.tensor([3.0, 2.0]))
        self.assertEqual(critic_loss(torch.tensor([[2.0, 4.0]]), torch.tensor([3.0])).item(), 1.0)
        terminal = td_target(torch.tensor([[1.0, 2.0]]), torch.tensor([False]), torch.empty(0), 0.5)
        self.assertEqual(terminal.item(), 2.0)

    def test_actor_gradient_through_q_but_not_q_parameters(self):
        for spec in (feature_spec(), feature_spec(7, 2, 3, 4)):
            config = HeadConfig(2, (8,))
            actor, critic = Actor(spec, config), TwinQ(spec, config)
            reference = torch.ones(3, 2, spec.action_dim) * 0.5
            batch = {
                "latent": torch.ones(3, spec.latent_dim),
                "state": torch.ones(3, spec.proprio_dim),
                "reference_condition": reference.clone(),
                "bc_target": reference.clone(),
            }
            loss, _ = actor_loss(
                actor, critic, batch, LearningConfig(bc_weight=0, reference_dropout=1)
            )
            loss.backward()
            self.assertGreater(sum(p.grad.abs().sum().item() for p in actor.parameters()), 0)
            self.assertTrue(all(p.grad is None and p.requires_grad for p in critic.parameters()))
            torch.testing.assert_close(batch["reference_condition"], reference)
            torch.testing.assert_close(batch["bc_target"], reference)

    def test_deterministic_actor_and_seeded_exploration(self):
        spec = feature_spec()
        actor = Actor(spec, HeadConfig(2, (8,)))
        args = (torch.ones(2, 4), torch.ones(2, 3), torch.zeros(2, 3, 2))
        a = actor(*args, generator=torch.Generator().manual_seed(9))
        b = actor(*args, generator=torch.Generator().manual_seed(9))
        torch.testing.assert_close(a, b)
        self.assertTrue(a.abs().le(1).all())
        self.assertFalse(torch.equal(a, actor(*args, deterministic=True)))
