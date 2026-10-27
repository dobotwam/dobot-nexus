import unittest

import torch

from dobot_nexus.algorithms.rltoken.token import (
    TokenConfig,
    TokenDecoder,
    TokenEncoder,
    reconstruction_loss,
)


class TokenTest(unittest.TestCase):
    def test_bfloat16_parameters_accept_float_teacher_and_accumulate_fp32_loss(self):
        encoder = TokenEncoder(self.config).to(torch.bfloat16)
        decoder = TokenDecoder(self.config).to(torch.bfloat16)
        prefix = torch.randn(1, 4, 6)
        valid = torch.ones(1, 4, dtype=torch.bool)
        output = decoder(encoder(prefix, valid), prefix, valid)
        loss = reconstruction_loss(output, prefix, valid)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def setUp(self):
        torch.set_num_threads(1)
        self.config = TokenConfig(6, width=8, heads=2, layers=1, max_tokens=8)

    def test_detached_teacher_and_encoder_receive_token_gradients(self):
        prefix = torch.randn(2, 4, 6, requires_grad=True)
        valid = torch.tensor([[True, True, False, False], [False] * 4])
        encoder, decoder = TokenEncoder(self.config), TokenDecoder(self.config)
        prediction = decoder(encoder(prefix, valid), prefix, valid)
        loss = reconstruction_loss(prediction, prefix, valid)
        loss.backward()
        self.assertIsNone(prefix.grad)
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
        )
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters())
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            reconstruction_loss(prediction, prefix, torch.zeros_like(valid))

    def test_teacher_is_shifted_and_causal(self):
        decoder = TokenDecoder(self.config).eval()
        prefix = torch.randn(1, 4, 6)
        valid = torch.ones(1, 4, dtype=torch.bool)
        latent = torch.randn(1, 8)
        before = decoder(latent, prefix, valid)
        prefix[:, 2:] += 100
        after = decoder(latent, prefix, valid)
        torch.testing.assert_close(before[:, :3], after[:, :3])

    def test_initialization_preserves_rng_and_masked_loss_uses_valid_elements(self):
        state = torch.get_rng_state().clone()
        TokenEncoder(self.config)
        TokenDecoder(self.config)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        target = torch.tensor([[[1.0, 3.0], [float("nan"), 0.0]]])
        self.assertEqual(
            reconstruction_loss(
                torch.zeros_like(target), target, torch.tensor([[True, False]])
            ).item(),
            5.0,
        )
        with self.assertRaises(ValueError):
            TokenConfig(6, width=7, heads=2)
