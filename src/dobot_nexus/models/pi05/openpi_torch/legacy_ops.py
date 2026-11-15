"""Native arithmetic boundaries for the legacy inference compile profile.

Inductor can fuse tensor casts, masks, layouts and BF16 pointwise work around
these calls. GEMMs, reductions, transcendentals and FP32 add/sub rounding keep
ATen implementations. This module is imported lazily by the inference engine;
training does not dispatch through these inference-only custom operators.
"""

import torch
import torch.nn.functional as F


@torch.library.custom_op("nexus_pi05_legacy::add", mutates_args=())
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


@add.register_fake
def _(x, y):
    return x + y


@torch.library.custom_op("nexus_pi05_legacy::linear", mutates_args=())
def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return F.linear(x, weight, bias)


@linear.register_fake
def _(x, weight, bias):
    return F.linear(x, weight, bias)


@torch.library.custom_op("nexus_pi05_legacy::matmul", mutates_args=())
def matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, weight)


@matmul.register_fake
def _(x, weight):
    return torch.matmul(x, weight)


@torch.library.custom_op("nexus_pi05_legacy::einsum", mutates_args=())
def einsum(equation: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum(equation, x, y)


@einsum.register_fake
def _(equation, x, y):
    return torch.einsum(equation, x, y)


@torch.library.custom_op("nexus_pi05_legacy::softmax", mutates_args=())
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return F.softmax(x, dim=dim)


@softmax.register_fake
def _(x, dim):
    return F.softmax(x, dim=dim)


@torch.library.custom_op("nexus_pi05_legacy::gelu_glu", mutates_args=())
def gelu_glu(gate: torch.Tensor, value: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    return F.gelu(gate, approximate=approximate) * value


@gelu_glu.register_fake
def _(gate, value, approximate="none"):
    return F.gelu(gate, approximate=approximate) * value


@torch.library.custom_op("nexus_pi05_legacy::silu", mutates_args=())
def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


@silu.register_fake
def _(x):
    return F.silu(x)


@torch.library.custom_op("nexus_pi05_legacy::rms_inverse", mutates_args=())
def rms_inverse(x: torch.Tensor) -> torch.Tensor:
    return torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-06)


@rms_inverse.register_fake
def _(x):
    return torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)


@torch.library.custom_op("nexus_pi05_legacy::rope_angles", mutates_args=())
def rope_angles(positions: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    exponents = (
        2.0 / head_dim * torch.arange(head_dim // 2, device=positions.device, dtype=torch.float32)
    )
    radians = positions[..., None].float() / (10000.0**exponents)[None, None, :]
    radians = radians[..., None, :]
    return (torch.sin(radians), torch.cos(radians))


@rope_angles.register_fake
def _(positions, head_dim):
    shape = (*positions.shape, 1, head_dim // 2)
    return (
        torch.empty(shape, device=positions.device, dtype=torch.float32),
        torch.empty(shape, device=positions.device, dtype=torch.float32),
    )


def rope(x, *, positions, max_wavelength=10000.0):
    sin, cos = positions
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([sub(x1 * cos, x2 * sin), add(x2 * cos, x1 * sin)], dim=-1).to(x.dtype)


@torch.library.custom_op("nexus_pi05_legacy::time_embedding", mutates_args=())
def time_embedding(
    pos: torch.Tensor, embedding_dim: int, min_period: float = 0.004, max_period: float = 4.0
) -> torch.Tensor:
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device=pos.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    radians = torch.einsum("i,j->ij", pos.float(), 1.0 / period * 2 * torch.pi)
    return torch.cat([torch.sin(radians), torch.cos(radians)], dim=-1).to(pos.dtype)


@time_embedding.register_fake
def _(pos, embedding_dim, min_period=0.004, max_period=4.0):
    return torch.empty((pos.shape[0], embedding_dim), device=pos.device, dtype=pos.dtype)


@torch.library.custom_op("nexus_pi05_legacy::sub", mutates_args=())
def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x - y


@sub.register_fake
def _(x, y):
    return x - y


def project(module, value):
    if type(module) is not torch.nn.Linear:
        raise ValueError("Pi0.5 legacy compilation requires unwrapped nn.Linear projections.")
    return linear(value, module.weight, module.bias)
