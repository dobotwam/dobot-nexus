import unittest

import torch

from dobot_nexus.algorithms.recap.conditioning import condition_prompt, select_condition


class ConditioningTest(unittest.TestCase):
    def test_condition_mask_endpoints_and_prompt_guard(self):
        positive = torch.tensor([True, True, False, False])
        u = torch.tensor([0.05, 0.2, 0.5, 1.0])
        self.assertEqual(
            select_condition(positive, unconditional_prob=0.1, random_values=u).tolist(),
            [False, True, True, True],
        )
        self.assertTrue(
            torch.equal(
                select_condition(positive, unconditional_prob=0, random_values=u),
                torch.ones_like(positive),
            )
        )
        self.assertFalse(select_condition(positive, unconditional_prob=1, random_values=u).any())
        self.assertEqual(condition_prompt("pick", "positive"), "pick\nAdvantage: positive")
        for task in ("", "pick\nAdvantage: positive", "pick advantage : negative"):
            with self.assertRaises(ValueError):
                condition_prompt(task, "base")
