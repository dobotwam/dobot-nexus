"""Frozen history features using the local Torch SigLIP and existing vision.npz."""

from __future__ import annotations

import numpy as np
import torch

from dobot_nexus.models.pi05.torch_core.model import resize_with_pad_torch
from dobot_nexus.models.pi05.torch_core.numerics import configure_frozen_siglip
from dobot_nexus.models.pi05.torch_core.siglip import SigLIPViT

from .feature_store import content_sha256


def load_vision_state(archive, model):
    """Map a standalone scanned SigLIP archive to the local module's keys.

    Validate source shapes before reshaping so a different architecture cannot
    silently load. No OpenPI, Flax or JAX model is needed to create a template.
    """
    state, consumed = {}, set()
    width, depth, heads = model.width, model.depth, model.num_heads
    head_dim = width // heads

    def take(key, shape):
        if key not in archive:
            raise ValueError(f"Missing memory vision parameter: {key}")
        value = archive[key]
        if value.shape != tuple(shape) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid FP32 memory vision parameter: {key}; expected shape {shape}")
        consumed.add(key)
        return value

    def put(key, value):
        state[key] = torch.from_numpy(np.ascontiguousarray(value))

    put(
        "stem.weight", take("embedding/kernel", (*model.patch_size, 3, width)).transpose(3, 2, 0, 1)
    )
    put("stem.bias", take("embedding/bias", (width,)))
    put("pos_embedding", take("pos_embedding", model.pos_embedding.shape))
    block = "Transformer/encoderblock"
    for source, target in (("LayerNorm_0", "norm1"), ("LayerNorm_1", "norm2")):
        for leaf, parameter in (("scale", "weight"), ("bias", "bias")):
            values = take(f"{block}/{source}/{leaf}", (depth, width))
            for index, value in enumerate(values):
                put(f"encoder.layers.{index}.{target}.{parameter}", value)
    for source, target, inputs, outputs in (
        ("Dense_0", "fc1", width, model.mlp_dim),
        ("Dense_1", "fc2", model.mlp_dim, width),
    ):
        for leaf, parameter, shape in (
            ("kernel", "weight", (depth, inputs, outputs)),
            ("bias", "bias", (depth, outputs)),
        ):
            values = take(f"{block}/MlpBlock_0/{source}/{leaf}", shape)
            for index, value in enumerate(values):
                put(
                    f"encoder.layers.{index}.mlp.{target}.{parameter}",
                    value.T if leaf == "kernel" else value,
                )
    attention = f"{block}/MultiHeadDotProductAttention_0"
    for leaf, parameter, shape in (
        ("kernel", "weight", (depth, width, heads, head_dim)),
        ("bias", "bias", (depth, heads, head_dim)),
    ):
        values = [take(f"{attention}/{name}/{leaf}", shape) for name in ("query", "key", "value")]
        for index in range(depth):
            parts = [
                v[index].reshape(width, width).T if leaf == "kernel" else v[index].reshape(width)
                for v in values
            ]
            put(f"encoder.layers.{index}.attn.in_proj_{parameter}", np.concatenate(parts))
    kernels = take(f"{attention}/out/kernel", (depth, heads, head_dim, width))
    biases = take(f"{attention}/out/bias", (depth, width))
    for index in range(depth):
        put(f"encoder.layers.{index}.attn.out_proj.weight", kernels[index].reshape(width, width).T)
        put(f"encoder.layers.{index}.attn.out_proj.bias", biases[index])
    for leaf, parameter in (("scale", "weight"), ("bias", "bias")):
        put(f"encoder.norm.{parameter}", take(f"Transformer/encoder_norm/{leaf}", (width,)))
    put("head.weight", take("head/kernel", (width, model.head.out_features)).T)
    put("head.bias", take("head/bias", (model.head.out_features,)))
    extra = set(archive) - consumed
    if extra:
        raise ValueError(f"Unexpected memory vision parameters: {sorted(extra)}")
    model.load_state_dict(state, strict=True, assign=True)


def preprocess_frames(frames, device):
    """Float normalization before antialiased resize; black padding is -1."""
    images = []
    for frame in frames:
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[-1] != 3
            or min(frame.shape[:2]) <= 0
        ):
            raise ValueError("History frames must be non-empty uint8 HWC RGB.")
        # Preserve the original CPU FP32 normalization before moving to Torch.
        image = torch.from_numpy(frame.astype(np.float32) / 255 * 2 - 1).to(device)
        images.append(resize_with_pad_torch(image, 224, 224, antialias=True))
    if not images:
        raise ValueError("Cannot encode an empty frame batch.")
    return torch.stack(images)


class TorchFrameTokenEncoder:
    """Inference-only encoder; uses Artifact weights independently of policy vision."""

    def __init__(self, checkpoint, *, device="cpu", expected_sha256=None, batch_size=16,
                 cuda_graph=False, max_graphs=2):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Memory vision batch_size must be a positive integer.")
        if expected_sha256 and content_sha256(checkpoint) != expected_sha256:
            raise ValueError("Memory vision content fingerprint mismatch.")
        self.device = torch.device(device)
        self.batch_size = batch_size
        with torch.device("meta"):
            model = SigLIPViT(num_classes=2048, dtype_mm="bfloat16")
        with np.load(checkpoint, allow_pickle=False) as archive:
            load_vision_state(archive, model)
        configure_frozen_siglip(model)
        self.model = model.to(self.device).eval().requires_grad_(False)
        from dobot_nexus.models.pi05.openpi_torch.vision import FrozenVisionEngine

        self.engine = FrozenVisionEngine(self.model, cuda_graph=cuda_graph, max_graphs=max_graphs)

    @property
    def metadata(self):
        return self.engine.metadata

    @torch.inference_mode()
    def __call__(self, frames):
        frames = list(frames)
        if not frames:
            raise ValueError("Cannot encode an empty frame batch.")
        output = []
        # An ambient policy autocast must not reduce the frozen FP32 stem.
        with torch.autocast(self.device.type, enabled=False):
            for start in range(0, len(frames), self.batch_size):
                images = preprocess_frames(frames[start : start + self.batch_size], self.device)
                output.append(self.engine(images).cpu().numpy())
        return np.concatenate(output)
