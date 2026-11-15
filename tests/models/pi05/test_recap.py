from __future__ import annotations

import dataclasses
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from dobot_nexus.algorithms.recap.annotations import read_returns, write_labels
from dobot_nexus.algorithms.recap.data import build_frame_index
from dobot_nexus.algorithms.recap.labeling import run_returns
from dobot_nexus.algorithms.recap.targets import label_advantages
from dobot_nexus.core.artifacts import load_manifest
from dobot_nexus.core.config import clone_config
from dobot_nexus.core.contracts import ObservationBatch, SamplingOptions
from dobot_nexus.core.experiment import Experiment
from dobot_nexus.inference import InferencePolicy
from dobot_nexus.models.pi05.configuration_recap import LEGACY_MODEL_TARGET, MODEL_TARGET
from dobot_nexus.models.pi05.data.pipeline import _Tokenizer
from dobot_nexus.models.pi05.data.recap import route_observation
from dobot_nexus.models.pi05.modeling_pi05 import Pi05Model
from dobot_nexus.models.pi05.modeling_recap import Pi05ReCapModel
from dobot_nexus.training import BCTrainer
from tests.algorithms.recap.fakes import FakeSource, FakeTokenizer, offline_config

from .recap_fakes import fake_core_source, policy_config


class ModelTest(unittest.TestCase):

    def test_strict_tokenizer_keeps_original_bc_truncation(self):
        tokenizer = _Tokenizer(Path("unused"), 2)
        tokenizer._processor = FakeTokenizer()
        self.assertEqual(len(tokenizer.encode("pick", np.zeros(2))[0]), 2)
        with self.assertRaisesRegex(ValueError, "exceeding"):
            tokenizer.encode("pick", np.zeros(2), strict=True)

    def test_standard_trainer_artifact_portable_inference_and_original_core_keys(self):
        self._check_policy_roundtrip(MODEL_TARGET)

    def test_saved_legacy_target_supports_training_export_and_inference(self):
        self._check_policy_roundtrip(LEGACY_MODEL_TARGET)

    def _check_policy_roundtrip(self, target):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offline = offline_config(root)
            run_returns(offline)
            index = build_frame_index(offline.dataset)
            write_labels(
                offline.annotation_dir,
                index=index,
                predicted_values=np.full(8, -0.5),
                labels=label_advantages(
                    np.arange(8),
                    index.task_indices,
                    positive_ratio=0.3,
                    interventions=read_returns(
                        offline.annotation_dir, expected_index=index
                    ).interventions,
                ),
                value_checkpoint_id="fixture",
                config=offline,
            )
            config = policy_config(root, offline)
            config.model.target = target
            config.model.max_token_len = 256
            with (
                mock.patch(
                    "dobot_nexus.models.pi05.modeling_pi05.load_core_source",
                    side_effect=fake_core_source,
                ),
                mock.patch("dobot_nexus.models.pi05.data.pipeline.build_source", FakeSource),
                mock.patch.object(_Tokenizer, "_get_processor", return_value=FakeTokenizer()),
                mock.patch(
                    "dobot_nexus.models.pi05.modeling_pi05.Pi05DataAdapter",
                    side_effect=AssertionError("RECAP must construct its adapter through the hook"),
                ),
            ):
                model = Pi05ReCapModel.from_config(config, mode="train")
                bad_horizon = clone_config(config)
                bad_horizon.model.action_horizon = 3
                with self.assertRaisesRegex(ValueError, "n_step"):
                    Pi05ReCapModel.from_config(bad_horizon, mode="train")
                bad_camera = clone_config(config)
                mapping = bad_camera.data.field_mapping.images
                mapping["left_wrist"], mapping["right_wrist"] = (
                    mapping["right_wrist"], mapping["left_wrist"]
                )
                with self.assertRaisesRegex(ValueError, "image_fields"):
                    Pi05ReCapModel.from_config(bad_camera, mode="train")
                loader = model.build_train_dataloader(config)
                self.assertEqual(
                    config.model.operator_numerics, model.pi05_config.operator_numerics
                )
                self.assertEqual(config.model.max_token_len, model.pi05_config.max_token_len)
                raw = next(iter(loader))
                batch = model.adapter.encode_train_batch(raw, torch.device("cpu"))
                original = batch["observation"].tokenized_prompt.clone()
                use = torch.ones(raw["positive"].shape, dtype=torch.bool)
                routed = route_observation(batch, use)
                torch.testing.assert_close(routed.tokenized_prompt, raw["condition_tokens"])
                torch.testing.assert_close(batch["observation"].tokenized_prompt, original)
                self.assertIs(routed.images, batch["observation"].images)
                extras = {"memory_inputs": (torch.zeros(1), torch.ones(1), torch.ones(1))}
                with mock.patch.object(
                    model.core_model, "compute_loss", return_value=torch.zeros(2, 2)
                ) as compute:
                    model.training_step({**batch, "model_kwargs": extras})
                    self.assertIs(
                        compute.call_args.kwargs["memory_inputs"], extras["memory_inputs"]
                    )
                self.assertNotIn("model_kwargs", batch)
                torch.testing.assert_close(
                    model.training_step(batch).loss,
                    Pi05Model.training_step(
                        model,
                        {
                            "observation": route_observation(
                                batch, torch.ones_like(batch["positive"])
                            ),
                            "actions": batch["actions"],
                        },
                    ).loss,
                )
                self.assertEqual(set(model.state_dict()), {"core_model.weight"})
                # Chunk beginning on model frame 0 includes actual human action at frame 1.
                sample = loader.dataset[0]
                decoded = model.adapter.pipeline.decode_actions(
                    sample["actions"][None], np.array([[0.0, 0.0]])
                )
                np.testing.assert_allclose(decoded[0], [[1, 3], [2, 3]], atol=1e-5)
                experiment = Experiment.from_config(config)
                result = BCTrainer(config, experiment).run()
                artifact = Path(result.checkpoint)
                manifest = load_manifest(artifact)
                self.assertNotIn("jax_weights", manifest["files"])
                self.assertFalse((artifact / "deployment/openpi_jax").exists())
                self.assertTrue((artifact / manifest["files"]["weights"]).is_file())
                self.assertEqual(manifest["model"]["recap"]["unconditional_prob"], 0)
                self.assertEqual(manifest["model"]["normalization"], "quantile")
                self.assertEqual(manifest["model"]["training_recipe"], "openpi_augmax041_fp32_v1")
                shutil.rmtree(offline.dataset.path)
                shutil.rmtree(offline.annotation_dir)
                config.data.tokenizer_path = str(root / "missing-tokenizer")
                infer = clone_config(config)
                infer.task = "infer"
                infer.model.artifact = str(artifact)
                infer.model.pop("recap")
                infer.model.pop("max_token_len")
                infer.model.inference_backend = "torch"
                infer.data.normalization = "zscore"  # Artifact must be authoritative.
                infer["inference"] = {"recap": {"condition": "positive"}}
                image = np.zeros((1, 224, 224, 3), dtype=np.uint8)
                obs = ObservationBatch(
                    images={k: image for k in ("base", "left_wrist", "right_wrist")},
                    state=np.zeros((1, 2), dtype=np.float32),
                    prompt=["insert plug"],
                )
                policy = InferencePolicy(infer)
                self.assertEqual(policy.config.model.target, target)
                self.assertEqual(policy.model.adapter.pipeline.normalization, "quantile")
                out = policy.predict_chunk(obs, SamplingOptions(seed=3))
                self.assertEqual(out.actions.shape, (1, 2, 2))
                self.assertEqual(out.actions.dtype, np.float32)
                self.assertEqual(obs.prompt, ["insert plug"])
                self.assertEqual(policy.model.recap_config.unconditional_prob, 0)
                self.assertEqual(policy.model.pi05_config.max_token_len, 256)
                bad_length = clone_config(infer)
                bad_length.model.max_token_len = 200
                with self.assertRaisesRegex(
                    ValueError, "Cannot override trained model.max_token_len"
                ):
                    InferencePolicy(bad_length)
                positive_encoded = policy.model.adapter.encode_observation(obs)
                infer["inference"] = {"recap": {"condition": "base"}}
                base_policy = InferencePolicy(infer)
                manual = base_policy.model.adapter.pipeline.transform_runtime_numpy(
                    dataclasses.replace(obs, prompt=["insert plug\nAdvantage: positive"])
                )[0]
                np.testing.assert_array_equal(
                    positive_encoded.observation.tokenized_prompt.numpy(),
                    manual["tokenized_prompt"],
                )
                self.assertFalse(
                    np.array_equal(out.actions, base_policy.predict_chunk(obs).actions)
                )
                bad = clone_config(infer)
                bad.model.recap = {"unconditional_prob": 0.5}
                with self.assertRaisesRegex(ValueError, "override"):
                    InferencePolicy(bad)
                saved_path = artifact / "config_resolved.yaml"
                saved_text = saved_path.read_text()
                saved = yaml.safe_load(saved_text)
                saved["model"]["target"] = (
                    MODEL_TARGET if target == LEGACY_MODEL_TARGET else LEGACY_MODEL_TARGET
                )
                try:
                    saved_path.write_text(yaml.safe_dump(saved))
                    with self.assertRaisesRegex(ValueError, "not a RECAP policy Artifact"):
                        InferencePolicy(infer)
                finally:
                    saved_path.write_text(saved_text)
                infer.model.inference_backend = "jax"
                with self.assertRaisesRegex(ValueError, "reference_jax"):
                    InferencePolicy(infer)
