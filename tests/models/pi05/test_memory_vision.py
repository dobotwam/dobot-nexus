"""CPU contracts for portable frozen Memory vision and backend selection."""

from __future__ import annotations

import builtins
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dobot_nexus.core.config import clone_config
from dobot_nexus.models.pi05.deploy.memory_server import build_frame_encoder
from dobot_nexus.models.pi05.memory.configuration import resolve_config
from dobot_nexus.models.pi05.memory.feature_store import content_sha256
from dobot_nexus.models.pi05.memory.torch_vision import (
    TorchFrameTokenEncoder,
    load_vision_state,
    preprocess_frames,
)
from dobot_nexus.models.pi05.torch_core import siglip
from dobot_nexus.models.pi05.torch_core.numerics import pool_siglip_tokens

VARIANT = dict(width=8, depth=2, num_heads=2, mlp_dim=12, patch_size=(14, 14))


def vision_archive():
    """Small standalone scanned checkpoint in the published vision.npz schema."""
    shapes = {
        "embedding/kernel": (14, 14, 3, 8),
        "embedding/bias": (8,),
        "pos_embedding": (1, 256, 8),
        "head/kernel": (8, 2048),
        "head/bias": (2048,),
        "Transformer/encoder_norm/scale": (8,),
        "Transformer/encoder_norm/bias": (8,),
    }
    prefix = "Transformer/encoderblock"
    for norm in ("LayerNorm_0", "LayerNorm_1"):
        for leaf in ("scale", "bias"):
            shapes[f"{prefix}/{norm}/{leaf}"] = (2, 8)
    for dense, inputs, outputs in (("Dense_0", 8, 12), ("Dense_1", 12, 8)):
        shapes[f"{prefix}/MlpBlock_0/{dense}/kernel"] = (2, inputs, outputs)
        shapes[f"{prefix}/MlpBlock_0/{dense}/bias"] = (2, outputs)
    attention = f"{prefix}/MultiHeadDotProductAttention_0"
    for projection in ("query", "key", "value"):
        shapes[f"{attention}/{projection}/kernel"] = (2, 8, 2, 4)
        shapes[f"{attention}/{projection}/bias"] = (2, 2, 4)
    shapes[f"{attention}/out/kernel"] = (2, 2, 4, 8)
    shapes[f"{attention}/out/bias"] = (2, 8)
    rng = np.random.default_rng(10)
    return {
        key: (
            np.ones(shape, np.float32)
            if key.endswith("/scale")
            else rng.normal(0, 0.1, shape).astype(np.float32)
        )
        for key, shape in shapes.items()
    }


class MemoryVisionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "vision.npz"
        self.archive = vision_archive()
        np.savez(self.path, **self.archive)
        patch = mock.patch.object(siglip, "_decode_variant", return_value=VARIANT)
        patch.start()
        self.addCleanup(patch.stop)

    def test_scanned_weights_preserve_projection_order_and_axes(self):
        with torch.device("meta"):
            model = siglip.SigLIPViT(num_classes=2048)
        load_vision_state(self.archive, model)
        attention = "Transformer/encoderblock/MultiHeadDotProductAttention_0"
        weights = model.encoder.layers[1].attn.in_proj_weight
        for i, projection in enumerate(("query", "key", "value")):
            expected = self.archive[f"{attention}/{projection}/kernel"][1].reshape(8, 8).T
            np.testing.assert_array_equal(weights[i * 8 : (i + 1) * 8].detach(), expected)
        self.assertEqual(
            model.stem.weight[3, 2, 1, 0], self.archive["embedding/kernel"][1, 0, 2, 3]
        )
        self.assertEqual(model.head.weight[100, 3], self.archive["head/kernel"][3, 100])

    def test_checkpoint_rejects_missing_extra_wrong_shape_dtype_and_nonfinite(self):
        for failure in ("missing", "extra", "shape", "dtype", "nonfinite"):
            with self.subTest(failure=failure), torch.device("meta"):
                model = siglip.SigLIPViT(num_classes=2048)
            data = dict(self.archive)
            if failure == "missing":
                del data["head/bias"]
            elif failure == "extra":
                data["unknown"] = np.zeros(1, np.float32)
            elif failure == "shape":
                data["head/kernel"] = data["head/kernel"].T
            elif failure == "dtype":
                data["head/bias"] = data["head/bias"].astype(np.float16)
            else:
                data["head/bias"] = np.full(2048, np.nan, np.float32)
            with self.assertRaises(ValueError):
                load_vision_state(data, model)

    def test_torch_service_encodes_without_optional_runtimes_or_padding_batch(self):
        policy = SimpleNamespace(
            device="cpu",
            model=SimpleNamespace(
                inference_backend="torch",
                pi05_config=SimpleNamespace(cuda_graph=False, cuda_graph_max_entries=2),
                memory_config=SimpleNamespace(
                    vision_checkpoint=str(self.path), vision_sha256=content_sha256(self.path)
                ),
            ),
        )
        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"jax", "jaxlib", "flax", "orbax", "openpi"}:
                raise AssertionError(f"Torch history attempted optional runtime import: {name}")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded):
            encoder = build_frame_encoder(policy)
            encoder.batch_size = 2
            frames = [np.full((224, 224, 3), v, np.uint8) for v in (0, 100, 255)]
            with mock.patch.object(
                encoder.model, "forward", wraps=encoder.model.forward
            ) as forward:
                features = encoder(frames)
            self.assertEqual([c.args[0].shape[0] for c in forward.call_args_list], [2, 1])
            np.testing.assert_array_equal(features[2:], encoder(frames[2:]))
        self.assertEqual(features.shape, (3, 16, 2048))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertFalse(encoder.model.training)
        self.assertFalse(any(p.requires_grad for p in encoder.model.parameters()))
        self.assertEqual(encoder.model.stem.weight.dtype, torch.float32)
        self.assertEqual(encoder.model.encoder.norm.weight.dtype, torch.float32)
        self.assertEqual(encoder.model.head.weight.dtype, torch.bfloat16)

    def test_preprocessing_preserves_float_range_padding_and_input_validation(self):
        frame = np.full((448, 224, 3), 255, np.uint8)
        image = preprocess_frames([frame], "cpu")
        self.assertEqual(tuple(image.shape), (1, 224, 224, 3))
        self.assertEqual(image.dtype, torch.float32)
        torch.testing.assert_close(image[:, :, :56], torch.full_like(image[:, :, :56], -1))
        torch.testing.assert_close(image[:, :, 56:168], torch.ones_like(image[:, :, 56:168]))
        torch.testing.assert_close(image[:, :, 168:], torch.full_like(image[:, :, 168:], -1))
        for frames in ([], [frame.astype(np.float32)], [np.zeros((0, 10, 3), np.uint8)]):
            with self.assertRaises(ValueError):
                preprocess_frames(frames, "cpu")

    def test_vision_fingerprint_is_checked_before_loading(self):
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            TorchFrameTokenEncoder(self.path, expected_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            TorchFrameTokenEncoder(self.path, batch_size=0)

    def test_frozen_vision_engine_requires_cuda_for_graphs(self):
        from dobot_nexus.models.pi05.openpi_torch.vision import FrozenVisionEngine

        model = torch.nn.Identity()
        with self.assertRaisesRegex(ValueError, "positive"):
            FrozenVisionEngine(model, max_graphs=0)
        engine = FrozenVisionEngine(model, cuda_graph=True)
        with self.assertRaisesRegex(ValueError, "CUDA device"):
            engine(torch.zeros(1, 224, 224, 3))
        self.assertEqual(engine.metadata["cuda_graph_captures"], 0)

    def test_pooling_preserves_spatial_order_and_bfloat16_rounding(self):
        patches = torch.zeros((1, 16, 16, 2), dtype=torch.bfloat16)
        patches[0, 0, :4, 0] = torch.tensor([256, 1, -256, 1], dtype=torch.bfloat16)
        patches[0, 4:8, 8:12, 1] = 2
        pooled = pool_siglip_tokens(patches.reshape(1, 256, 2))
        expected = torch.zeros((1, 16, 2), dtype=torch.bfloat16)
        expected[0, 0, 0] = 1 / 16  # FP32 accumulation would instead produce 2/16.
        expected[0, 6, 1] = 2
        torch.testing.assert_close(pooled, expected, rtol=0, atol=0)

    def test_torch_ignores_stale_reference_path_but_preserves_identity(self):
        config = clone_config(
            {
                "model": {
                    "inference_backend": "torch",
                    "openpi_root": "/missing/training/checkout",
                    "memory_reference_sha256": "trained-reference",
                    "memory": {"vision_checkpoint": str(self.path)},
                }
            }
        )
        resolved, memory = resolve_config(config, mode="infer")
        self.assertNotIn("openpi_root", resolved.model)
        self.assertEqual(resolved.model.memory_reference_sha256, "trained-reference")
        self.assertEqual(memory.vision_sha256, content_sha256(self.path))
        self.assertEqual(config.model.openpi_root, "/missing/training/checkout")
        trained, _ = resolve_config(config, mode="train")
        self.assertEqual(trained.model.openpi_root, "/missing/training/checkout")
        self.assertEqual(trained.model.memory_reference_sha256, "trained-reference")
        config.model.inference_backend = "jax"
        with self.assertRaisesRegex(ValueError, "reference_jax"):
            resolve_config(config, mode="infer")

if __name__ == "__main__":
    unittest.main()
