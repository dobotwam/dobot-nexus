import dataclasses
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dobot_nexus.modules.value.fusion.configuration import ValueModelConfig
from dobot_nexus.modules.value.fusion.processing import (
    FusionProcessor,
    preprocess_image,
    state_tokens,
    validate_state_stats,
)


class FusionProcessingTest(unittest.TestCase):
    def setUp(self):
        self.config = ValueModelConfig(Path("vision"), Path("language"), Path("tokenizer"))
        self.stats = {"min": [0, 0], "max": [4, 1]}
        self.tokenizer = mock.Mock(
            return_value={"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}
        )

    def processor(self, **kwargs):
        return FusionProcessor(
            self.config, camera_names=("wrist", "front"), tokenizer=self.tokenizer, **kwargs
        )

    def test_explicit_camera_order_and_text_recipe_without_dataset_keys(self):
        images = {
            "front": np.zeros((4, 8, 3), dtype=np.uint8),
            "wrist": np.full((8, 4, 3), 255, dtype=np.uint8),
        }
        observation = self.processor(state_stats=self.stats).encode_observation(
            images=[images], prompts=["  INSERT_Plug!\n "], states=[np.array([2, 2])]
        )
        self.tokenizer.assert_called_once_with(
            ["Task: INSERT Plug!, State: 128 255 " + " ".join(["127"] * 30) + "\nValue: "],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=200,
        )
        self.assertEqual(
            set(observation),
            {"input_ids", "attention_mask", "image_tensor", "image_attention_mask"},
        )
        self.assertEqual(observation["image_tensor"].shape, (1, 2, 3, 384, 384))
        self.assertTrue((observation["image_tensor"][:, 0] == 1).all())
        self.assertTrue((observation["image_tensor"][:, 1] == -1).all())
        torch.testing.assert_close(observation["image_attention_mask"], torch.ones(1, 2).long())

    def test_state_is_optional_only_when_disabled(self):
        with self.assertRaisesRegex(ValueError, "requires saved min/max"):
            self.processor()
        with self.assertRaisesRegex(ValueError, "missing state"):
            self.processor(state_stats=self.stats).format_prompt("insert plug")
        self.config = dataclasses.replace(self.config, include_state=False)
        processor = self.processor()
        self.assertEqual(processor.format_prompt("insert_plug\n"), "Task: insert plug\nValue: ")
        for bad in (None, "", "\n"):
            with self.subTest(prompt=bad), self.assertRaises(ValueError):
                processor.format_prompt(bad)

    def test_state_clipping_padding_and_existing_dimension_limit(self):
        ids = state_tokens([-1, 9], self.stats).split()
        self.assertEqual(ids, ["0", "255"] + ["127"] * 30)
        self.assertEqual(
            state_tokens([1] * 40, {"min": [0] * 40, "max": [1] * 40}).split(), ["255"] * 32
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            state_tokens([float("nan")], self.stats)
        for stats in (
            {"min": [], "max": []},
            {"min": [1], "max": [0]},
            {"min": [0], "max": [1, 2]},
            {"min": [0], "max": [float("inf")]},
        ):
            with self.subTest(stats=stats), self.assertRaises(ValueError):
                validate_state_stats(stats)

    def test_invalid_images_and_batch_dimensions_are_rejected(self):
        for image in (
            np.zeros((4, 8, 3), dtype=np.float32),
            np.zeros((4, 8), dtype=np.uint8),
            np.zeros((0, 8, 3), dtype=np.uint8),
        ):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                preprocess_image(image)
        processor = self.processor(state_stats=self.stats)
        for images, prompts, states in (([], [], None), ([{}], [], None), ([{}], ["x"], [])):
            with self.assertRaises(ValueError):
                processor.encode_observation(images=images, prompts=prompts, states=states)
        with self.assertRaisesRegex(ValueError, "require cameras"):
            processor.encode_observation(images=[{}], prompts=["x"], states=[np.array([0, 0])])
        for cameras in ((), ("front", "front"), ("",)):
            with self.assertRaises(ValueError):
                FusionProcessor(self.config, camera_names=cameras, state_stats=self.stats)
