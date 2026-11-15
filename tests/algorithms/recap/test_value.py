from __future__ import annotations

import dataclasses
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dobot_nexus.algorithms.recap.annotations import read_policy_labels, read_returns
from dobot_nexus.algorithms.recap.data import build_frame_index
from dobot_nexus.algorithms.recap.labeling import predict_values, run_labeling, run_returns
from dobot_nexus.algorithms.recap.value.checkpoint import (
    capture_value_rng_state,
    load_value_checkpoint,
    restore_value_rng_state,
)
from dobot_nexus.algorithms.recap.value.processing import (
    ReCapValueProcessor,
    ValueSample,
    forward_value,
    preprocess_image,
    state_tokens,
)
from dobot_nexus.algorithms.recap.value.training import (
    learning_rate_multiplier,
    split_episodes,
    train_value,
)
from dobot_nexus.training.resume import preserve_rng_state

from .fakes import FakeSource, FakeTokenizer, FakeValue, offline_config


class ValueTest(unittest.TestCase):

    def test_value_checkpoint_rng_roundtrip_preserves_all_cpu_streams(self):
        with preserve_rng_state():
            state = capture_value_rng_state()
            # Lists preserve the existing schema and allow weights_only=True loading.
            self.assertIsInstance(state["numpy"][1], list)
            buffer = io.BytesIO()
            torch.save(state, buffer)
            expected = random.random(), np.random.random(4), torch.rand(4)
            buffer.seek(0)
            restored = torch.load(buffer, weights_only=True)
            restore_value_rng_state(restored)
            self.assertEqual(random.random(), expected[0])
            np.testing.assert_array_equal(np.random.random(4), expected[1])
            torch.testing.assert_close(torch.rand(4), expected[2], rtol=0, atol=0)
            self.assertIsInstance(restored["numpy"][1], list)

    def test_processor_stretch_template_state_and_target_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory))
            index = build_frame_index(config.dataset)
            tokenizer = mock.Mock(wraps=FakeTokenizer())
            stats = {"min": [0.0, 0.0], "max": [4.0, 1.0]}
            processor = ReCapValueProcessor(config.value, state_stats=stats, tokenizer=tokenizer)
            image = np.full((8, 16, 3), 255, dtype=np.uint8)
            sample = ValueSample(
                index.keys[0],
                {c: image for c in config.dataset.image_fields},
                "INSERT_Plug!",
                -0.5,
                np.array([2.0, 2.0]),
            )
            batch = processor.collate([sample])
            prompt = tokenizer.call_args.args[0][0]
            self.assertTrue(prompt.startswith("Task: INSERT Plug!, State: 128 255 127"))
            self.assertTrue(prompt.endswith("\nValue: "))
            self.assertEqual(batch.observation["image_tensor"].shape, (1, 3, 3, 384, 384))
            self.assertTrue((preprocess_image(image) == 1).all())
            self.assertNotIn("target_value", batch.observation)
            model = FakeValue()
            a = forward_value(model, batch, with_targets=False)
            b = forward_value(
                model, dataclasses.replace(batch, targets=torch.tensor([0.0])), with_targets=False
            )
            torch.testing.assert_close(a.values, b.values)
            self.assertIsNone(a.loss)
            self.assertEqual(state_tokens([-1, 9], stats).split()[:2], ["0", "255"])

    def test_episode_split_and_dobot_learning_rate_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory))
            train = dataclasses.replace(
                config.train,
                val_episodes=1,
                max_steps=10,
                warmup_steps=2,
                decay_steps=6,
                stable_steps=2,
                learning_rate=1.0,
                decay_lr=0.1,
            )
            a, b = split_episodes(build_frame_index(config.dataset), train)
            self.assertFalse(set(a) & set(b))
            self.assertEqual(set(a + b), {3, 8})
            curve = learning_rate_multiplier(train)
            np.testing.assert_allclose(
                [curve(i) for i in (0, 1, 2, 4, 7, 10)], [0.1, 0.55, 1, 1, 0.55, 0.1]
            )

    def test_train_resume_and_label_roundtrip(self):
        class StochasticValue(FakeValue):
            def forward(self, observation):
                logits = super().forward(observation)
                return torch.nn.functional.dropout(logits, p=0.3, training=self.training)

        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory))
            config = dataclasses.replace(
                config,
                train=dataclasses.replace(
                    config.train, max_steps=4, save_every=1, val_episodes=1, log_every=2
                ),
            )
            run_returns(config)
            with (
                mock.patch("dobot_nexus.algorithms.recap.value.dataset.build_source", FakeSource),
                mock.patch(
                    "dobot_nexus.algorithms.recap.value.training.build_value_model",
                    side_effect=lambda _: StochasticValue(),
                ),
                mock.patch(
                    "dobot_nexus.algorithms.recap.value.checkpoint.build_value_model",
                    side_effect=lambda _: StochasticValue(),
                ),
                mock.patch.object(
                    ReCapValueProcessor,
                    "tokenizer",
                    new_callable=mock.PropertyMock,
                    return_value=FakeTokenizer(),
                ),
            ):
                with mock.patch("dobot_nexus.algorithms.recap.value.training.LOGGER.info") as log:
                    trained = train_value(config)
                self.assertEqual([call.args[1] for call in log.call_args_list], [1, 2, 4])
                self.assertIsNotNone(trained.best_checkpoint)
                self.assertTrue(np.isfinite(trained.metrics["val_loss"]))
                self.assertTrue(np.isfinite(trained.metrics["val_mae"]))
                resumed_config = dataclasses.replace(
                    config,
                    train=dataclasses.replace(
                        config.train,
                        output_dir=Path(directory) / "resumed",
                        resume_from=config.train.output_dir / "step_00000001.pt",
                        log_every=3,  # Reporting frequency must not invalidate exact resume.
                    ),
                )
                resumed = train_value(resumed_config)
                left = torch.load(trained.checkpoint, weights_only=True)
                right = torch.load(resumed.checkpoint, weights_only=True)
                torch.testing.assert_close(left["model"], right["model"], rtol=0, atol=0)
                self.assertEqual(
                    left["training_state"]["scheduler"], right["training_state"]["scheduler"]
                )
                stopped_config = dataclasses.replace(
                    config,
                    train=dataclasses.replace(
                        config.train, output_dir=Path(directory) / "stopped", stop_steps=2
                    ),
                )
                stopped = train_value(stopped_config)
                self.assertEqual(stopped.global_step, 2)
                stopped_payload = torch.load(stopped.checkpoint, weights_only=True)
                original_two = torch.load(
                    config.train.output_dir / "step_00000002.pt", weights_only=True
                )
                torch.testing.assert_close(
                    stopped_payload["model"], original_two["model"], rtol=0, atol=0
                )
                self.assertEqual(
                    stopped_payload["training_state"]["scheduler"],
                    original_two["training_state"]["scheduler"],
                )
                continued = train_value(
                    dataclasses.replace(
                        config,
                        train=dataclasses.replace(
                            config.train,
                            output_dir=Path(directory) / "continued",
                            resume_from=stopped.checkpoint,
                        ),
                    )
                )
                torch.testing.assert_close(
                    torch.load(continued.checkpoint, weights_only=True)["model"],
                    left["model"],
                    rtol=0,
                    atol=0,
                )
                loaded = load_value_checkpoint(
                    trained.checkpoint, device=torch.device("cpu"), config=config
                )
                with mock.patch.dict(
                    sys.modules, {"dobot_nexus.algorithms.recap.value.training": None}
                ):
                    a = predict_values(config, trained.checkpoint)
                changed = dataclasses.replace(
                    config, labeling=dataclasses.replace(config.labeling, batch_size=1)
                )
                b = predict_values(changed, trained.checkpoint)
                np.testing.assert_allclose(a.values, b.values, atol=1e-7)
                run_labeling(config, trained.checkpoint)
                labels = read_policy_labels(config.annotation_dir, expected_index=a.index)
                self.assertEqual(labels.manifest["value_checkpoint_id"], loaded.checkpoint_id)
                returned = read_returns(config.annotation_dir, expected_index=a.index)
                self.assertEqual(returned.scale, loaded.scale)
                self.assertTrue(labels.positive[returned.interventions].all())
                left["model"]["unexpected"] = torch.zeros(1)
                corrupt = Path(directory) / "bad.pt"
                torch.save(left, corrupt)
                with self.assertRaises(RuntimeError):
                    load_value_checkpoint(corrupt, device=torch.device("cpu"), config=config)
