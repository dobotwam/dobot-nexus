import unittest

import numpy as np
import torch

from dobot_nexus.data.images import convert_image


class ImageConversionTest(unittest.TestCase):
    def test_rgb_layout_dtype_scaling_and_contiguity(self):
        hwc = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
        inputs = (hwc, hwc.transpose(2, 0, 1), torch.from_numpy(hwc), hwc.astype(np.int16))
        for value in inputs:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                actual = convert_image(value)
                np.testing.assert_array_equal(actual, hwc)
                self.assertTrue(actual.flags.c_contiguous)
        normalized = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        for value in (normalized, torch.tensor(normalized, dtype=torch.bfloat16)):
            np.testing.assert_array_equal(convert_image(value), [[[0, 127, 255]]])
        np.testing.assert_array_equal(convert_image(np.array([[[-1.0, 128.0, 300.0]]])),
                                      [[[0, 128, 255]]])

    def test_invalid_rank_and_channels_are_rejected(self):
        for shape in ((4, 6), (4, 6, 4), (2, 4, 6, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                convert_image(np.zeros(shape))
