import unittest

import torch

from dobot_nexus.modules.value.distributions import expected_value
from dobot_nexus.modules.value.losses import categorical_loss


class ValueLossTest(unittest.TestCase):
    def test_expected_value_on_uniform_and_concentrated_distributions(self):
        logits = torch.tensor([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 0.0, 1000.0]])
        torch.testing.assert_close(
            expected_value(logits, num_bins=3), torch.tensor([-0.5, -1.0, 0.0])
        )
        torch.testing.assert_close(
            expected_value(logits, num_bins=3, v_min=0, v_max=2), torch.tensor([1.0, 0.0, 2.0])
        )
        for invalid in (torch.zeros(201), torch.zeros(1, 5)):
            with self.assertRaises(ValueError):
                expected_value(invalid)

    def test_two_hot_targets_recovered_from_cross_entropy_gradient(self):
        logits = torch.zeros(4, 3, requires_grad=True)
        loss, metrics = categorical_loss(logits, torch.tensor([-1.0, -0.75, -0.5, 0.0]), num_bins=3)
        loss.sum().backward()
        expected = torch.tensor(
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        torch.testing.assert_close(1 / 3 - logits.grad, expected)
        self.assertTrue(torch.isfinite(metrics["mae"]))
        for bad in (float("nan"), 0.1, -1.1):
            with self.assertRaises(ValueError):
                categorical_loss(torch.zeros(1, 201), torch.tensor([bad]))
