"""Pi0 ACP policy routing, trainer and portable Artifact contracts."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml

from dobot_nexus.algorithms.recap.annotations import read_returns, write_labels
from dobot_nexus.algorithms.recap.data import build_frame_index
from dobot_nexus.algorithms.recap.labeling import run_returns
from dobot_nexus.algorithms.recap.targets import label_advantages
from dobot_nexus.core.artifacts import load_manifest, resume_fingerprint
from dobot_nexus.core.config import clone_config, load_config
from dobot_nexus.core.contracts import ObservationBatch, SamplingOptions
from dobot_nexus.inference import InferencePolicy
from dobot_nexus.models.pi0.configuration_pi0 import Pi0Config
from dobot_nexus.models.pi0.configuration_recap import resolve_policy_config
from dobot_nexus.models.pi0.data.pipeline import _Tokenizer
from dobot_nexus.models.pi0.data.recap import ReCapPipeline, route_observation
from dobot_nexus.models.pi0.modeling_pi0 import Pi0Model
from dobot_nexus.models.pi0.modeling_recap import Pi0ReCapModel
from dobot_nexus.training import BCTrainer
from tests.algorithms.recap.fakes import FakeSource, FakeTokenizer, offline_config
from tests.models.pi0.fixtures import ROOT, infer_config, small_backbone, train_config


class Pi0ReCapTest(unittest.TestCase):
    def test_default_bc_dataloader_preserves_legacy_resume_fingerprint(self):
        legacy = load_config(ROOT / "examples/configs/train/pi0.yaml")
        self.assertNotIn("max_token_len", legacy.model)
        fingerprint = resume_fingerprint(legacy)
        model = Pi0Model.__new__(Pi0Model)
        torch.nn.Module.__init__(model)
        model.mode = "train"
        model.adapter = mock.Mock()
        model.adapter.pipeline.build_dataloader.return_value = ([], {})
        for explicit, resolved in ((None, 48), (48, 48), (None, 64)):
            with self.subTest(explicit=explicit, resolved=resolved):
                config = clone_config(legacy)
                if explicit is not None:
                    config.model.max_token_len = explicit
                model.pi0_config = replace(
                    Pi0Config.from_config(config, mode="train"), max_token_len=resolved
                )
                model.build_train_dataloader(config)
                if explicit is None and resolved == 48:
                    self.assertNotIn("max_token_len", config.model)
                    self.assertEqual(resume_fingerprint(config), fingerprint)
                else:
                    self.assertEqual(config.model.max_token_len, resolved)
                    self.assertNotEqual(resume_fingerprint(config), fingerprint)

    def test_strict_tokenization_includes_the_pi0_delimiter(self):
        tokenizer = _Tokenizer(Path("unused"), 5)
        tokenizer._processor = FakeTokenizer()
        with self.assertLogs(level="WARNING"):
            tokens, mask = tokenizer.encode("pick")
        self.assertEqual(len(tokens), 5)
        self.assertTrue(mask.all())
        pipeline = ReCapPipeline.__new__(ReCapPipeline)
        pipeline.tokenizer = tokenizer
        with self.assertRaisesRegex(ValueError, "exceeding"):
            pipeline.tokenize_prompt("pick")
        tokenizer.max_length = 6
        self.assertTrue(tokenizer.encode("pick", strict=True)[1].all())

    def test_config_requires_annotations_and_positive_integer_token_limit(self):
        config = clone_config(
            {
                "model": {
                    "checkpoint": "base.pt",
                    "state_dim": 2,
                    "action_dim": 2,
                    "action_horizon": 2,
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "data.recap.annotations"):
            resolve_policy_config(config, mode="train")
        self.assertEqual(Pi0Config.from_config(config, mode="train").max_token_len, 48)
        for invalid in (0, -1, True, 2.5, "48", None):
            config.model.max_token_len = invalid
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "positive integer"),
            ):
                Pi0Config.from_config(config, mode="train")

    def test_trainer_artifact_and_all_inference_conditions(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            small_backbone(expert_width=16) as source,
            mock.patch("dobot_nexus.models.pi0.data.pipeline.build_source", FakeSource),
            mock.patch.object(_Tokenizer, "_get_processor", return_value=FakeTokenizer()),
        ):
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
            config = train_config(root, "recap", source)
            config.model.recap.unconditional_prob = 0.0
            config.data.path = str(offline.dataset.path)
            self.assertEqual(config.data.format_version, "v2.1")
            config.data.field_mapping.images = dict(offline.dataset.image_fields)
            config.data.recap.annotations = str(offline.annotation_dir)
            for field in ("n_step", "image_fields"):
                bad = clone_config(config)
                if field == "n_step":
                    bad.model.action_horizon = 3
                else:
                    cameras = bad.data.field_mapping.images
                    cameras.left_wrist, cameras.right_wrist = (
                        cameras.right_wrist,
                        cameras.left_wrist,
                    )
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                    Pi0ReCapModel(bad, mode="train")

            model = Pi0ReCapModel(config, mode="train")
            loader = model.build_train_dataloader(config)
            injected = "pick_Advantage: positive"
            with self.assertRaisesRegex(ValueError, "reserved Advantage"):
                model.adapter.pipeline.transform_frame(
                    {**loader.dataset.source[0], "prompt": injected}
                )
            raw = loader.collate_fn([loader.dataset[0], loader.dataset[1]])
            batch = model.adapter.encode_train_batch(raw, torch.device("cpu"))
            observation = batch["observation"]
            original = observation.tokenized_prompt.clone()
            original_mask = observation.tokenized_prompt_mask.clone()
            use = torch.tensor([True, False])
            routed = route_observation(batch, use)
            self.assertIs(routed.images, observation.images)
            torch.testing.assert_close(routed.tokenized_prompt[0], batch["condition_tokens"][0])
            torch.testing.assert_close(routed.tokenized_prompt[1], original[1])
            torch.testing.assert_close(observation.tokenized_prompt, original)
            torch.testing.assert_close(observation.tokenized_prompt_mask, original_mask)
            self.assertEqual(
                batch["condition_tokens"][:, 1].tolist(),
                [5 if flag else 6 for flag in batch["positive"].tolist()],
            )
            item = loader.dataset[0]
            decoded = model.adapter.pipeline.decode_actions(item["actions"][None], np.zeros((1, 2)))
            np.testing.assert_allclose(decoded[0], [[1, 3], [2, 3]], atol=1e-5)
            artifact = Path(BCTrainer(config).run().checkpoint)
            manifest = load_manifest(artifact)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["model"]["recap"]["unconditional_prob"], 0)
            self.assertEqual(
                manifest["model"]["recap_annotations"]["value_checkpoint_id"], "fixture"
            )
            self.assertEqual(manifest["training"]["metrics"]["recap_conditioned_fraction"], 1)
            self.assertIn("recap_positive_fraction", manifest["training"]["metrics"])
            weights = torch.load(artifact / manifest["files"]["weights"], weights_only=True)
            base_keys = torch.load(config.model.checkpoint, weights_only=True).keys()
            self.assertEqual(weights.keys(), base_keys)
            self.assertIn("state_proj.weight", weights)
            self.assertIn("action_time_mlp_in.weight", weights)
            self.assertIn("llm.layers.0.pre_ffw_norms.1.scale", weights)
            shutil.rmtree(offline.dataset.path)
            shutil.rmtree(offline.annotation_dir)
            for path in (
                config.model.checkpoint,
                config.data.norm_stats_path,
                config.data.tokenizer_path,
            ):
                Path(path).unlink()

            infer = infer_config(artifact, "recap")
            image = np.zeros((1, 24, 24, 3), np.uint8)
            obs = ObservationBatch(
                images=dict.fromkeys(("base", "left_wrist", "right_wrist"), image),
                state=np.zeros((1, 2), np.float32),
                prompt=["insert plug"],
            )
            actions = {}
            for condition, token in (("base", 2), ("positive", 5), ("negative", 6)):
                infer.inference.recap.condition = condition
                policy = InferencePolicy(infer)
                with self.assertRaisesRegex(ValueError, "reserved Advantage"):
                    policy.predict_chunk(replace(obs, prompt=[injected]))
                self.assertEqual(policy.model.pi0_config.max_token_len, 12)
                encoded = policy.model.adapter.encode_observation(obs)
                self.assertEqual(encoded.observation.tokenized_prompt[0, 1].item(), token)
                actions[condition] = policy.predict_chunk(obs, SamplingOptions(seed=7)).actions
                self.assertEqual(actions[condition].shape, (1, 2, 2))
                self.assertEqual(obs.prompt, ["insert plug"])
            for condition in ("positive", "negative"):
                self.assertFalse(np.array_equal(actions[condition], actions["base"]))

            for key, value in (("recipe", "other"), ("unconditional_prob", 0.5)):
                bad = clone_config(infer)
                bad.model.recap = {key: value}
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "override"):
                    InferencePolicy(bad)
            bad = clone_config(infer)
            bad.model.max_token_len = 48
            with self.assertRaisesRegex(ValueError, "override trained model.max_token_len"):
                InferencePolicy(bad)
            manifest_path = artifact / "manifest.yaml"
            for field, value in (("recap", {}), ("max_token_len", 49)):
                manifest_path.write_text(
                    yaml.safe_dump(
                        {
                            **manifest,
                            "model": {**manifest["model"], field: value},
                        }
                    )
                )
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "disagrees"):
                    InferencePolicy(infer)
            manifest_path.write_text(yaml.safe_dump(manifest))
            saved = artifact / "config_resolved.yaml"
            payload = yaml.safe_load(saved.read_text())
            base_target = "dobot_nexus.models.pi0.modeling_pi0:Pi0Model"
            recap_target = manifest["model"]["target"]
            for meta_target, saved_target in (
                (base_target, recap_target),
                (recap_target, base_target),
            ):
                with self.subTest(manifest_target=meta_target, saved_target=saved_target):
                    payload["model"]["target"] = saved_target
                    saved.write_text(yaml.safe_dump(payload))
                    manifest["model"]["target"] = meta_target
                    manifest_path.write_text(yaml.safe_dump(manifest))
                    with self.assertRaisesRegex(ValueError, "model.target disagrees"):
                        InferencePolicy(infer)
                    with self.assertRaisesRegex(ValueError, "model.target disagrees"):
                        Pi0Model(infer, mode="infer")
                    with self.assertRaisesRegex(ValueError, "RECAP policy Artifact"):
                        resolve_policy_config(infer, mode="infer")
