from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dobot_nexus.core.config import clone_config, load_config, save_config
from dobot_nexus.core.contracts import ObservationBatch
from dobot_nexus.models.pi05.configuration_pi05 import Pi05Config
from dobot_nexus.models.pi05.data import Pi05Pipeline
from dobot_nexus.models.pi05.memory.checkpoint import load_memory_checkpoint
from dobot_nexus.models.pi05.modeling_pi05 import Pi05Model
from dobot_nexus.models.pi05.training.checkpoint import load_core_checkpoint

ROOT = Path(__file__).resolve().parents[3]


class ProjectionCore(torch.nn.Module):
    def __init__(self, dim, *, memory=False):
        super().__init__()
        self.action_in_proj = torch.nn.Linear(dim, 4)
        self.action_out_proj = torch.nn.Linear(4, dim)
        self.other = torch.nn.Parameter(torch.ones(3))
        if memory:
            self.memory_encoder = torch.nn.Linear(4, 4)
            self.llm = torch.nn.Module()
            self.llm.layers = torch.nn.ModuleList()


def _kbd_config():
    return load_config(ROOT / "examples/recipes/pi05_kbd_pick_0916_full34.yaml")


class ActionDimensionConfigTest(unittest.TestCase):
    def test_config_accepts_larger_boundary_and_rejects_invalid_dimensions(self):
        config = _kbd_config()
        parsed = Pi05Config.from_config(config, mode="train")
        self.assertEqual(parsed.core_kwargs()["action_dim"], 64)
        self.assertEqual((parsed.state_dim, parsed.action_dim), (34, 34))
        for value in (32, 0, -1, 64.5, True, "64"):
            with self.subTest(value=value):
                config.model.model_action_dim = value
                with self.assertRaisesRegex(ValueError, "model_action_dim"):
                    Pi05Config.from_config(config, mode="train")


class ProjectionCheckpointTest(unittest.TestCase):
    def test_expansion_preserves_pretrained_weights_and_rng_on_meta_model(self):
        source = ProjectionCore(32).state_dict()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "base.pt"
            torch.save(source, checkpoint)
            results = []
            for _ in range(2):
                with torch.device("meta"):
                    model = ProjectionCore(64)
                rng = torch.get_rng_state().clone()
                load_core_checkpoint(model, checkpoint, allow_action_expansion=True)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertFalse(any(p.is_meta for p in model.parameters()))
                state = model.state_dict()
                torch.testing.assert_close(state["action_in_proj.weight"][:, :32],
                                           source["action_in_proj.weight"], rtol=0, atol=0)
                for key in ("action_out_proj.weight", "action_out_proj.bias"):
                    torch.testing.assert_close(state[key][:32], source[key], rtol=0, atol=0)
                for key in ("action_in_proj.bias", "other"):
                    torch.testing.assert_close(state[key], source[key], rtol=0, atol=0)
                self.assertGreater(torch.count_nonzero(state["action_in_proj.weight"][:, 32:]), 0)
                self.assertGreater(torch.count_nonzero(state["action_out_proj.weight"][32:]), 0)
                self.assertEqual(torch.count_nonzero(state["action_out_proj.bias"][32:]), 0)
                results.append(state)
            for key in results[0]:
                torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)

            # An expanded trained checkpoint reloads without any initialization.
            torch.save(results[0], checkpoint)
            restored = ProjectionCore(64)
            load_core_checkpoint(restored, checkpoint)
            for key, value in restored.state_dict().items():
                torch.testing.assert_close(value, results[0][key], rtol=0, atol=0)

    def test_deployment_shrinking_and_unrelated_mismatches_remain_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            for source_dim, target_dim, expand, corrupt in (
                (32, 64, False, False), (64, 32, True, False),
                (32, 64, True, True),
            ):
                with self.subTest(source=source_dim, target=target_dim, corrupt=corrupt):
                    state = ProjectionCore(source_dim).state_dict()
                    if corrupt:
                        state["other"] = torch.zeros(5)
                    torch.save(state, checkpoint)
                    with self.assertRaisesRegex(RuntimeError, "does not match"):
                        load_core_checkpoint(ProjectionCore(target_dim), checkpoint,
                                             allow_action_expansion=expand)

    def test_model_only_allows_expansion_during_training(self):
        with mock.patch("dobot_nexus.models.pi05.modeling_pi05.load_core_checkpoint") as load:
            for mode in ("train", "infer"):
                model = SimpleNamespace(mode=mode, core_model=object(),
                                        pi05_config=SimpleNamespace(checkpoint="weights.pt"))
                Pi05Model._load_core_checkpoint(model)
                load.assert_called_with(model.core_model, "weights.pt",
                                        allow_action_expansion=mode == "train")


class MemoryProjectionCheckpointTest(unittest.TestCase):
    def test_memory_base_initialization_uses_the_same_projection_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "base.pt"
            source = ProjectionCore(32)
            torch.save(source.state_dict(), checkpoint)
            with torch.device("meta"):
                model = ProjectionCore(64, memory=True)
            load_memory_checkpoint(model, checkpoint, allow_initialization=True)
            self.assertFalse(any(p.is_meta for p in model.parameters()))
            torch.testing.assert_close(model.action_in_proj.weight[:, :32],
                                       source.action_in_proj.weight, rtol=0, atol=0)
            torch.save(ProjectionCore(32, memory=True).state_dict(), checkpoint)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_memory_checkpoint(model, checkpoint)


class ActionBoundaryArtifactTest(unittest.TestCase):
    def test_artifact_restores_boundary_and_rejects_conflicting_metadata_or_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _kbd_config()
            save_config(train, root / "config_resolved.yaml")
            manifest = {"schema_version": 2, "model": {"model_action_dim": 64}}
            save_config(manifest, root / "manifest.yaml")
            infer = clone_config(train)
            infer.model.artifact = str(root)
            infer.model.pop("model_action_dim")
            self.assertEqual(Pi05Config.from_config(infer, mode="infer").model_action_dim, 64)
            infer.model.model_action_dim = 128
            with self.assertRaisesRegex(ValueError, "Cannot override trained model.model_action_dim"):
                Pi05Config.from_config(infer, mode="infer")
            infer.model.pop("model_action_dim")
            manifest["model"]["model_action_dim"] = 32
            save_config(manifest, root / "manifest.yaml")
            with self.assertRaisesRegex(ValueError, "Artifact model_action_dim disagrees"):
                Pi05Config.from_config(infer, mode="infer")


class ActionDimensionPipelineTest(unittest.TestCase):
    def test_pipeline_pads_training_and_runtime_then_decodes_public_actions(self):
        for state_dim, action_dim, internal in ((14, 14, 32), (34, 34, 64), (12, 7, 32), (48, 42, 64)):
            with self.subTest(state=state_dim, action=action_dim), tempfile.TemporaryDirectory() as directory:
                config = _kbd_config()
                config.model.update(state_dim=state_dim, action_dim=action_dim, model_action_dim=internal)
                stats = Path(directory) / "norm_stats.json"
                stats.write_text(json.dumps({key: {"mean": [0.] * dim, "std": [1.] * dim}
                                             for key, dim in (("state", state_dim), ("actions", action_dim))}))
                config.data.norm_stats_path = str(stats)
                parsed = Pi05Config.from_config(config, mode="train")
                pipeline = Pi05Pipeline(config, parsed)
                pipeline.tokenizer._processor = SimpleNamespace(encode=lambda _text, add_bos: [1, 2])
                image = np.zeros((8, 8, 3), dtype=np.uint8)
                state = np.arange(state_dim, dtype=np.float32) / state_dim
                actions = np.tile(np.arange(action_dim, dtype=np.float32) / action_dim + 0.25, (50, 1))
                frame = {key: image for key in config.data.field_mapping.images.values()}
                frame.update({"observation.state": state, "action": actions, "task": "pick"})
                encoded = pipeline.transform_frame(frame)
                self.assertEqual(encoded["actions"].shape, (50, internal))
                np.testing.assert_array_equal(encoded["actions"][:, action_dim:], 0.)
                obs = ObservationBatch(images={key: image[None] for key in parsed.camera_names},
                                       state=state[None], prompt=["pick"])
                runtime, raw_state = pipeline.transform_runtime_numpy(obs)
                np.testing.assert_allclose(runtime["state"][0], encoded["state"], rtol=1e-6)
                np.testing.assert_array_equal(runtime["state"][:, state_dim:], 0.)
                decoded = pipeline.decode_actions(encoded["actions"][None], raw_state)
                self.assertEqual(decoded.shape, (1, 50, action_dim))
                np.testing.assert_allclose(decoded[0], actions, atol=1e-5)
                obs.state = np.zeros((1, state_dim + 1))
                with self.assertRaisesRegex(ValueError, "Expected state"):
                    pipeline.transform_runtime_numpy(obs)
                with self.assertRaisesRegex(ValueError, "does not cover configured action_dim"):
                    pipeline.decode_actions(np.zeros((1, 50, action_dim - 1)), raw_state)
