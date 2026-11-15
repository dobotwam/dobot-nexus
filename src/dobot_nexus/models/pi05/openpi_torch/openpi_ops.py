"""Native compilation boundaries for the OpenPI numerical profile."""

import torch

from dobot_nexus.models.pi05.torch_core.numerics import OpenPiLinear

from .legacy_ops import (
    add,
    einsum,
    gelu_glu,
    linear,
    matmul,
    rope,
    rope_angles,
    silu,
    softmax,
    sub,
    time_embedding,
)
from .legacy_ops import project as legacy_project

__all__ = [
    "add",
    "einsum",
    "gelu_glu",
    "linear",
    "matmul",
    "rope",
    "rope_angles",
    "silu",
    "softmax",
    "sub",
    "time_embedding",
    "project",
    "rms_inverse",
]


@torch.library.custom_op("nexus_pi05_openpi::rms_inverse", mutates_args=())
def rms_inverse(x: torch.Tensor) -> torch.Tensor:
    return torch.reciprocal(torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6))


@rms_inverse.register_fake
def _(x):
    return torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)


def project(module, value):
    if isinstance(module, OpenPiLinear):
        dtype = torch.promote_types(value.dtype, module.weight.dtype)
        with torch.autocast(value.device.type, enabled=False):
            return linear(
                value.to(dtype),
                module.weight.to(dtype),
                module.bias.to(dtype) if module.bias is not None else None,
            )
    return legacy_project(module, value)
