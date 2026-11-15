import unittest

import numpy as np

from dobot_nexus.algorithms.recap.targets import (
    compute_episode_returns,
    compute_n_step_advantages,
    label_advantages,
)
from dobot_nexus.algorithms.recap.types import ReturnScale


class TargetsTest(unittest.TestCase):
    def test_unknown_interventions_do_not_override_advantage_labels(self):
        labels = label_advantages(
            [-3, -2, -1, 2], [0, 0, 0, 0], positive_ratio=0.25,
            interventions=[True, False, None, None],
        )
        np.testing.assert_array_equal(labels.positive, [True, False, False, True])
        for invalid in ([1, False, None, None], ["true", False, None, None], [None]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                label_advantages(
                    [-3, -2, -1, 2], [0, 0, 0, 0], positive_ratio=0.25,
                    interventions=invalid,
                )

    def test_success_failure_and_single_frame(self):
        scale = ReturnScale({0: 5})
        success = compute_episode_returns(5, True, task=0, scale=scale)
        failure = compute_episode_returns(5, False, task=0, scale=scale)
        np.testing.assert_allclose(success.returns, [-0.4, -0.3, -0.2, -0.1, 0])
        np.testing.assert_allclose(failure.returns, [-0.9, -0.8, -0.7, -0.6, -0.5])
        np.testing.assert_allclose(success.rewards, [-0.1, -0.1, -0.1, -0.1, 0])
        for flag, expected in ((True, 0), (False, -0.5)):
            result = compute_episode_returns(1, flag, task=0, scale=ReturnScale({0: 1}))
            self.assertEqual(result.returns[0], expected)
        with self.assertRaises(ValueError):
            compute_episode_returns(3, "false", task=0, scale=scale)

    def test_scales_depend_on_task_length_not_observed_failure(self):
        scale = ReturnScale({0: 100, 1: 200})
        self.assertEqual(scale.denominator(0), 200)
        a = compute_episode_returns(100, True, task=0, scale=scale)
        b = compute_episode_returns(100, True, task=1, scale=scale)
        self.assertAlmostEqual(float(a.returns[0]), -0.495)
        self.assertAlmostEqual(float(b.returns[0]), -0.2475)
        for args in (({}, 1), ({0: 0}, 1), ({0: 1}, -1)):
            with self.assertRaises(ValueError):
                ReturnScale(*args)

    def test_monte_carlo_has_zero_advantage_at_episode_boundaries(self):
        scale = ReturnScale({0: 5})
        eps = [
            compute_episode_returns(length, success, task=0, scale=scale)
            for length, success in ((5, True), (2, False), (1, True))
        ]
        for n in (1, 2, 50):
            advantages = compute_n_step_advantages(
                np.concatenate([e.rewards for e in eps]),
                np.concatenate([e.returns for e in eps]),
                [0, 5, 7, 8],
                n_step=n,
            )
            np.testing.assert_allclose(advantages, 0, atol=8e-8)

    def test_hand_computed_bootstrap_and_terminal(self):
        actual = compute_n_step_advantages([-0.5, -0.5, 0], [-0.9, -0.5, -0.2], [0, 3], n_step=2)
        np.testing.assert_allclose(actual, [-0.3, 0, 0.2], atol=1e-7)
        with self.assertRaises(ValueError):
            compute_n_step_advantages([-1], [-0.2], [0, 0, 1], n_step=1)

    def test_task_thresholds_and_human_override(self):
        out = label_advantages(
            [0, 1, 100, 101],
            [0, 0, 1, 1],
            positive_ratio=0.5,
            interventions=np.array([True, False, False, False]),
        )
        self.assertEqual(out.thresholds, {0: 0.5, 1: 100.5})
        np.testing.assert_array_equal(out.positive, [True, True, False, True])
        disabled = label_advantages(
            [0, 1],
            [0, 0],
            positive_ratio=0.5,
            interventions=np.array([True, False]),
            force_intervention_positive=False,
        )
        np.testing.assert_array_equal(disabled.positive, [False, True])
        self.assertTrue(label_advantages([0, 0], [0, 0], positive_ratio=0.3).positive.all())
