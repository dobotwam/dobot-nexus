"""CPU contracts for Pi0.5 augmentation, flow precision and training RNG."""

from __future__ import annotations

import sys
import unittest

import torch


class AugmentationTest(unittest.TestCase):
    def test_rng_advances_and_explicit_generator_does_not_consume_global_rng(self):
        from dobot_nexus.models.pi05.torch_core.augmentation import augment_image

        image = torch.linspace(-1, 1, 2 * 32 * 32 * 3).reshape(2, 32, 32, 3)
        rng = torch.Generator().manual_seed(9)
        saved = rng.get_state()
        global_state = torch.get_rng_state()
        first = augment_image(image, geometry=True, rng=rng)
        second = augment_image(image, geometry=True, rng=rng)
        self.assertFalse(torch.equal(first, second))
        rng.set_state(saved)
        torch.testing.assert_close(
            first, augment_image(image, geometry=True, rng=rng), rtol=0, atol=0
        )
        torch.testing.assert_close(global_state, torch.get_rng_state(), rtol=0, atol=0)
        self.assertTrue(torch.isfinite(first).all())

    def test_wrist_geometry_is_identity_when_color_is_disabled(self):
        from dobot_nexus.models.pi05.torch_core.augmentation import (
            augment_image,
            sample_parameters,
        )

        image = torch.rand(2, 32, 32, 3) * 2 - 1
        parameters = sample_parameters(image, rng=torch.Generator().manual_seed(1))
        parameters[:, 6] = 0
        actual = augment_image(image, geometry=False, parameters=parameters)
        torch.testing.assert_close(actual, image, rtol=0, atol=0)
        cropped = augment_image(image, geometry=True, parameters=parameters)
        self.assertFalse(torch.equal(cropped, image))

    def test_training_default_is_random_eval_does_not_augment(self):
        from dobot_nexus.models.pi05.torch_core import model

        image = torch.linspace(-1, 1, 224 * 224 * 3).reshape(1, 224, 224, 3)
        obs = model.Observation(
            images={key: image.clone() for key in model.IMAGE_KEYS},
            image_masks={},
            state=torch.zeros(1, 32),
        )
        before = torch.get_rng_state()
        a = model.preprocess_observation(obs, train=True)
        torch.testing.assert_close(
            a.images["left_wrist_0_rgb"], a.images["right_wrist_0_rgb"], rtol=0, atol=0
        )
        b = model.preprocess_observation(obs, train=True)
        self.assertFalse(torch.equal(a.images["base_0_rgb"], b.images["base_0_rgb"]))
        torch.set_rng_state(before)
        c = model.preprocess_observation(obs, train=True)
        torch.testing.assert_close(a.images["base_0_rgb"], c.images["base_0_rgb"], rtol=0, atol=0)
        evaluation = model.preprocess_observation(obs, train=False)
        for key in obs.images:
            torch.testing.assert_close(evaluation.images[key], obs.images[key], rtol=0, atol=0)



class FlowTest(unittest.TestCase):
    def core_and_inputs(self):
        from dobot_nexus.models.pi05.torch_core.source import load_core_source

        source = load_core_source()
        module = sys.modules[source.package_name + ".pi0"]

        class Core:
            embed_dtype = torch.bfloat16
            action_horizon = 2
            action_out_proj = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)

            def embed_prefix(self, observation):
                return torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1)

            def embed_suffix(self, observation, actions, time):
                self.actions, self.time = actions, time
                self.images = observation.images
                return (
                    actions.to(torch.bfloat16),
                    torch.ones(1, 2, dtype=torch.bool),
                    torch.ones(2),
                    None,
                )

            def llm(self, tokens, **kwargs):
                return tokens, None

        image = torch.linspace(-1, 1, 224 * 224 * 3).reshape(1, 224, 224, 3)
        observation = source.Observation(
            images={k: image for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
            image_masks={},
            state=torch.zeros(1, 32),
        )
        return module.Pi0.compute_loss, Core(), observation

    def test_interpolation_target_and_loss_remain_float32_and_backward_works(self):
        compute, core, observation = self.core_and_inputs()
        actions = torch.tensor([[[0.12345, 0.23456], [0.45678, -0.99999]]])
        noise = torch.tensor([[[0.82347, -0.51432], [0.71029, 0.09128]]])
        time = torch.tensor([0.573829])
        loss = compute(core, observation, actions, noise=noise, time=time)
        expected_x = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        torch.testing.assert_close(core.actions, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(core.time, time, rtol=0, atol=0)
        v = core.action_out_proj(expected_x.to(torch.bfloat16)).float()
        torch.testing.assert_close(loss, ((v - (noise - actions)) ** 2).mean(-1), rtol=0, atol=0)
        self.assertEqual(loss.dtype, torch.float32)
        loss.mean().backward()
        self.assertTrue(torch.isfinite(core.action_out_proj.weight.grad).all())

    def test_flow_rng_isolated_and_checkpoint_rng_restore_repeats_training(self):
        from dobot_nexus.training.resume import capture_rng_state, restore_rng_state

        compute, core, observation = self.core_and_inputs()
        actions = torch.zeros(1, 2, 2)
        rng = torch.Generator().manual_seed(4)
        before = torch.get_rng_state()
        first = compute(core, observation, actions, train=True, rng=rng)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        rng.manual_seed(4)
        torch.testing.assert_close(first, compute(core, observation, actions, train=True, rng=rng))
        state = capture_rng_state()
        expected = compute(core, observation, actions, train=True, rng=torch.default_generator)
        restore_rng_state(state)
        actual = compute(core, observation, actions, train=True, rng=torch.default_generator)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)



if __name__ == "__main__":
    unittest.main()
