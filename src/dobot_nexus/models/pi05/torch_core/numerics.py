"""OpenPI operator arithmetic shared by Pi0.5 BC and FrameSample models."""

from __future__ import annotations

import importlib

import torch
from torch import nn
from torch.nn import functional as F


class OpenPiLinear(nn.Linear):
    """Preserve NNX Linear's input/parameter dtype promotion inside autocast."""

    def forward(self, value):
        dtype = torch.promote_types(value.dtype, self.weight.dtype)
        with torch.autocast(value.device.type, enabled=False):
            return F.linear(
                value.to(dtype), self.weight.to(dtype),
                self.bias.to(dtype) if self.bias is not None else None,
            )


class SiglipLayerNorm(nn.LayerNorm):
    """Frozen SigLIP: FP32 statistics/affine parameters, BF16 activations."""

    def forward(self, value):
        value = value.float()
        mean = value.mean(dim=-1, keepdim=True)
        variance = (value.square().mean(dim=-1, keepdim=True) - mean.square()).clamp_min(0)
        scale = torch.rsqrt(variance + self.eps) * self.weight
        return ((value - mean) * scale + self.bias).to(torch.bfloat16)


def configure_frozen_siglip(model):
    """Keep the independent memory encoder's FP32 stem and norm parameters.

    Dense/attention operands use BF16, matching the frozen Linen encoder. This
    profile is instance-local and does not change the trainable policy vision.
    """
    model.dtype_mm = torch.bfloat16
    for block in model.encoder.layers:
        block.attn.to(dtype=torch.bfloat16)
        block.mlp.to(dtype=torch.bfloat16)
        block.mlp.gelu_approximate = "tanh"
    model.head.to(dtype=torch.bfloat16)
    for parent in (model.encoder, *model.encoder.layers):
        for name, original in list(parent.named_children()):
            if isinstance(original, nn.LayerNorm):
                norm = SiglipLayerNorm(original.normalized_shape, eps=original.eps, device="meta")
                norm.weight, norm.bias = original.weight, original.bias
                setattr(parent, name, norm)


def pool_siglip_tokens(tokens):
    """Pool 16x16 patches in 4x4 windows with the frozen encoder's rounding.

    The reference reduce_window adds in spatial order in the input dtype.
    A Torch mean/sum would accumulate BF16 in FP32 and change the features.
    """
    batch, _, channels = tokens.shape
    grid = tokens.reshape(batch, 4, 4, 4, 4, channels)
    pooled = torch.zeros_like(grid[:, :, 0, :, 0])
    for row in range(4):
        for column in range(4):
            pooled = pooled + grid[:, :, row, :, column]
    return (pooled / 16).reshape(batch, 16, channels)


def apply_openpi_numerics(core, source):
    """Configure arithmetic without adding parameters or changing checkpoint keys."""
    gemma = importlib.import_module(source.package_name + ".gemma")
    core.preserve_projection_inputs = True
    core.sampling_noise_dtype = torch.float32
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        original = getattr(core, name)
        if isinstance(original, OpenPiLinear):
            continue
        # Construct on meta so replacing a projection does not consume RNG or
        # allocate/reinitialize weights. Original Parameter identities survive.
        projection = OpenPiLinear(
            original.in_features, original.out_features,
            bias=original.bias is not None, device="meta", dtype=original.weight.dtype,
        )
        projection.weight, projection.bias = original.weight, original.bias
        setattr(core, name, projection)
    for block in core.img.encoder.layers:
        block.mlp.gelu_approximate = "tanh"
    for module in core.llm.modules():
        if isinstance(module, gemma.RMSNorm):
            module.openpi_math = True
    for block in core.llm.layers:
        for mlp in block.mlps:
            mlp.gelu_approximate = "tanh"
