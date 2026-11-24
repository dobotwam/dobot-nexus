from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

import numpy as np

from dobot_nexus.data import (
    ActionTransform,
    NormStats,
    apply_action_transform,
    invert_action_transform,
    load_norm_stats,
    normalize,
    save_norm_stats,
    unnormalize,
    validate_norm_stats,
)
from dobot_nexus.data.images import resize_with_pad


class SharedTransformsTest(unittest.TestCase):
    def setUp(self):
        self.stats = NormStats(
            mean=np.array([1.0, 2.0, 3.0]),
            std=np.array([1.0, 0.0, 2.0]),
            q01=np.array([-3.0, 2.0, 0.0]),
            q99=np.array([8.0, 2.0, 3.0]),
        )

    def test_both_modes_preserve_outliers_and_constant_dimensions(self):
        value = np.array([[[10.0, -1.0], [-4.0, 2.0]]])
        for mode in ("quantile", "zscore"):
            with self.subTest(mode=mode):
                encoded = normalize(value, self.stats, mode=mode)
                if mode == "zscore":
                    expected = (value - [1.0, 2.0]) / (np.array([1.0, 0.0]) + 1e-6)
                    np.testing.assert_array_equal(encoded, normalize(value, self.stats))
                else:
                    expected = (value - [-3.0, 2.0]) / (np.array([11.0, 0.0]) + 1e-6) * 2 - 1
                np.testing.assert_array_equal(encoded, expected)
                np.testing.assert_allclose(unnormalize(encoded, self.stats, mode=mode), value)

    def test_float16_quantiles_round_trip_across_a_wide_range(self):
        high = np.array([60000], np.float16)
        stats = NormStats(np.zeros_like(high), np.ones_like(high), q01=-high, q99=high)
        values = np.array([[-60000], [0], [60000]], np.float16)
        with np.errstate(over="raise", invalid="raise"):
            encoded = normalize(values, stats, mode="quantile")
            np.testing.assert_allclose(encoded, [[-1], [0], [1]], atol=1e-9)
            restored = unnormalize(encoded.astype(np.float16), stats, mode="quantile")
            np.testing.assert_allclose(restored, values, atol=1e-6)

    def test_invalid_statistics_and_modes(self):
        for changes in (
            {"mean": np.zeros(1)},
            {"std": np.zeros(1)},
            {"std": np.array(1.0)},
            {"mean": np.zeros((1, 3)), "std": np.ones((1, 3))},
            {"std": np.ones(4)},
            {"mean": np.array([np.inf, 0, 0])},
            {"std": np.array([0, np.nan, 0])},
            {"std": np.array([0, 0, -1])},
            {"std": np.array(["1", "2", "3"])},
            {"q01": None},
            {"q99": None},
            {"q01": np.ones((1, 3))},
            {"q99": np.ones(2)},
            {"q01": np.ones(1), "q99": np.ones(1)},
            {"q01": np.array([np.nan, 0, 0])},
            {"q99": np.array([1.0, np.inf, 1.0])},
            {"q99": np.array([1.0, 1.0, 1.0])},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "actions"):
                validate_norm_stats(
                    replace(self.stats, **changes), 2, mode="quantile", name="actions"
                )
            if "mean" in changes or "std" in changes:
                with self.subTest(mode="zscore", changes=changes), self.assertRaises(ValueError):
                    normalize(np.zeros(2), replace(self.stats, **changes))
        validate_norm_stats(replace(self.stats, q01=None, q99=None), 2)
        if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
            oversized = np.full(3, np.finfo(np.longdouble).max)
            with self.assertRaisesRegex(ValueError, "finite"):
                validate_norm_stats(replace(self.stats, mean=oversized), 2)
        for transform in (normalize, unnormalize):
            with self.assertRaisesRegex(ValueError, "normalization"):
                transform(np.ones(2), self.stats, mode="unsupported")

    def test_integer_arithmetic_matches_floating_reference(self):
        value = np.array([0, 255], dtype=np.uint8)
        stats = NormStats(
            mean=np.array([128, 250], np.uint8),
            std=np.array([2, 5], np.uint8),
            q01=np.array([64, -100], np.int8),
            q99=np.array([100, 100], np.int8),
        )
        expected = {
            "zscore": (
                (value.astype(float) - [128, 250]) / (np.array([2, 5]) + 1e-6),
                value * (np.array([2.0, 5.0]) + 1e-6) + [128, 250],
            ),
            "quantile": (
                (value.astype(float) - [64, -100]) / (np.array([36, 200]) + 1e-6) * 2 - 1,
                (value.astype(float) + 1) / 2 * (np.array([36, 200]) + 1e-6) + [64, -100],
            ),
        }
        for mode, (normalized, restored) in expected.items():
            with self.subTest(mode=mode):
                np.testing.assert_allclose(normalize(value, stats, mode=mode), normalized)
                np.testing.assert_allclose(unnormalize(value, stats, mode=mode), restored)
                np.testing.assert_allclose(
                    unnormalize(normalized, stats, mode=mode), value, atol=1e-12
                )

    def test_saved_horizon_zscores_broadcast_over_batch_and_slice_the_action_axis(self):
        mean = np.arange(12.0).reshape(3, 4)
        stats = NormStats(mean, mean + 1, q01=mean - 1, q99=mean + 1)
        actions = np.arange(12.0).reshape(2, 3, 2)
        expected = (actions - mean[:, :2]) / (stats.std[:, :2] + 1e-6)
        with tempfile.TemporaryDirectory() as directory:
            restored = load_norm_stats(save_norm_stats(directory, {"actions": stats}))["actions"]
        for candidate in (stats, restored):
            validate_norm_stats(candidate, 2)
            np.testing.assert_allclose(normalize(actions, candidate), expected)
            np.testing.assert_allclose(unnormalize(expected, candidate), actions)
            with self.assertRaisesRegex(ValueError, "1D"):
                validate_norm_stats(candidate, 2, mode="quantile")
            with self.assertRaises(ValueError):
                validate_norm_stats(candidate, 5)

    def test_quantiles_must_match_the_full_serialized_mean_shape(self):
        invalid = replace(self.stats, q01=np.zeros(2), q99=np.ones(2))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "q01/q99"):
                validate_norm_stats(invalid, 2, mode="quantile")
            with self.assertRaises(ValueError):
                load_norm_stats(save_norm_stats(directory, {"actions": invalid}))
            restored = load_norm_stats(save_norm_stats(directory, {"actions": self.stats}))
        validate_norm_stats(restored["actions"], 2, mode="quantile")
        value = np.array([[7.0, 2.0]])
        encoded = normalize(value, restored["actions"], mode="quantile")
        np.testing.assert_allclose(unnormalize(encoded, self.stats, mode="quantile"), value)

    def test_zscores_cannot_expand_the_prediction_horizon(self):
        mean = np.arange(3.0).reshape(3, 1)
        stats = NormStats(mean, np.ones_like(mean))
        for operation in (normalize, unnormalize):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(ValueError, "expand input shape"):
                    operation(np.zeros((1, 1, 1)), stats)
                value = np.ones((2, 3, 1))
                self.assertEqual(operation(value, stats).shape, value.shape)
                np.testing.assert_allclose(unnormalize(normalize(value, stats), stats), value)

    def test_delta_rejects_inexact_int64_and_uint64_before_conversion(self):
        transform = ActionTransform.create("delta", [True, False], action_dim=2)
        limit = 2**53
        for op in (apply_action_transform, invert_action_transform):
            for dtype, too_large in ((np.uint64, limit + 1), (np.int64, -limit - 1)):
                zero = np.zeros(2, dtype)
                for values in ([too_large, 0], [0, too_large]):
                    bad = np.array(values, dtype)
                    for actions, state in ((bad, zero), (zero, bad)):
                        with self.subTest(op=op, dtype=dtype), self.assertRaisesRegex(
                            ValueError, "exact float64"
                        ):
                            op(actions, state, transform)
            boundary = np.array([-limit, limit], np.int64)
            np.testing.assert_array_equal(op(boundary, np.zeros(2), transform), boundary)

    def test_unsigned_action_deltas_do_not_wrap_in_either_direction(self):
        transform = ActionTransform.create("delta", [True, False], action_dim=2)
        state = np.array([[2, 255]], np.uint8)
        actions = np.array([[[1, 7], [255, 9]]], np.uint8)
        delta = apply_action_transform(actions, state, transform)
        self.assertTrue(np.issubdtype(delta.dtype, np.floating))
        np.testing.assert_array_equal(delta, [[[-1, 7], [253, 9]]])
        np.testing.assert_array_equal(invert_action_transform(delta, state, transform), actions)
        absolute = invert_action_transform(actions, state, transform)
        self.assertTrue(np.issubdtype(absolute.dtype, np.floating))
        np.testing.assert_array_equal(absolute, [[[3, 7], [257, 9]]])
        np.testing.assert_array_equal(actions, [[[1, 7], [255, 9]]])

    def test_integer_delta_and_sum_must_be_exact_after_arithmetic(self):
        transform = ActionTransform.create("delta", [True, False], action_dim=2)
        limit = 2**53
        for op, sign in ((apply_action_transform, -1), (invert_action_transform, 1)):
            for direction in (1, -1):
                actions = np.array([direction * limit, limit], np.int64)
                state = np.array([direction * sign, sign], np.int64)
                with self.assertRaisesRegex(ValueError, "result.*exact float64"):
                    op(actions, state, transform)
                # The next even integer is exactly representable. The second channel is absolute.
                result = op(actions, state * 2, transform)
                self.assertEqual(int(result[0]), direction * (limit + 2))
                self.assertEqual(int(result[1]), limit)

    def test_mixed_float32_and_integer_operands_round_trip_in_float64(self):
        transform = ActionTransform.create("delta", [True, False], action_dim=2)
        state = np.array([[2**24 + 1, 2**24 + 1]], np.int64)
        actions = np.array([[[0, 9], [1, 7]]], np.float32)
        delta = apply_action_transform(actions, state, transform)
        self.assertEqual(delta.dtype, np.float64)
        np.testing.assert_array_equal(delta, [[[-16777217, 9], [-16777216, 7]]])
        np.testing.assert_array_equal(invert_action_transform(delta, state, transform), actions)
        absolute = invert_action_transform(actions, state, transform)
        self.assertEqual(absolute.dtype, np.float64)
        np.testing.assert_array_equal(absolute, [[[16777217, 9], [16777218, 7]]])
        np.testing.assert_array_equal(apply_action_transform(absolute, state, transform), actions)

    def test_rgb_resize_geometry_bilinear_and_odd_padding(self):
        from PIL import Image

        for h, w, rh, rw in ((2, 5, 2, 7), (5, 2, 7, 2), (1, 10, 1, 7)):
            with self.subTest(shape=(h, w)):
                image = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
                padded = np.zeros((7, 7, 3), np.uint8)
                y, x = (7 - rh) // 2, (7 - rw) // 2
                padded[y : y + rh, x : x + rw] = Image.fromarray(image).resize(
                    (rw, rh), resample=getattr(Image, "Resampling", Image).BILINEAR
                )
                result = resize_with_pad(image, 7)
                self.assertEqual(result.dtype, np.uint8)
                np.testing.assert_array_equal(result, padded)
        square = np.zeros((224, 224, 3), np.uint8)
        self.assertIs(resize_with_pad(square), square)
