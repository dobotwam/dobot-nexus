from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from dobot_nexus.training.resume import capture_rng_state, restore_rng_state
from dobot_nexus.training.seed import seed_all, seed_dataloader, training_seed


class SeedTest(unittest.TestCase):
    def test_seed_controls_initialization_and_all_training_rngs(self):
        def sample(rank):
            seed_all(42)
            weight = torch.nn.Linear(3, 2).weight.detach().clone()
            seed_all(training_seed(42, rank))
            return weight, random.random(), np.random.random(3), torch.randn(3)

        first, repeat, other = sample(0), sample(0), sample(1)
        torch.testing.assert_close(first[0], repeat[0], rtol=0, atol=0)
        torch.testing.assert_close(first[0], other[0], rtol=0, atol=0)
        self.assertEqual(first[1], repeat[1])
        np.testing.assert_array_equal(first[2], repeat[2])
        torch.testing.assert_close(first[3], repeat[3], rtol=0, atol=0)
        self.assertNotEqual(first[1], other[1])
        self.assertFalse(torch.equal(first[3], other[3]))

    def test_resume_overrides_initial_seed(self):
        seed_all(42)
        torch.randn(5)
        state = capture_rng_state()
        expected = (random.random(), np.random.random(), torch.randn(3))
        seed_all(999)
        restore_rng_state(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.randn(3), expected[2], rtol=0, atol=0)

    def test_sampler_is_independent_of_training_and_worker_rng(self):
        def order(rank, extra_worker_draws):
            loader = torch.utils.data.DataLoader(torch.arange(20), batch_size=2, shuffle=True)
            seed_dataloader(loader, 42)
            self.assertIsNot(loader.generator, loader.sampler.generator)
            torch.rand(extra_worker_draws, generator=loader.generator)
            seed_all(training_seed(42, rank))
            torch.randn(rank + 1)
            return torch.cat(list(loader))

        torch.testing.assert_close(order(0, 0), order(3, 9), rtol=0, atol=0)

    def test_existing_sampler_generator_is_preserved(self):
        generator = torch.Generator().manual_seed(123)
        loader = torch.utils.data.DataLoader(torch.arange(8), shuffle=True, generator=generator)
        seed_dataloader(loader, 42)
        self.assertIs(loader.sampler.generator, generator)
        self.assertIsNot(loader.generator, generator)

    def test_invalid_seed_is_rejected(self):
        for seed in (-1, True, 1.5, "42"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                seed_all(seed)
