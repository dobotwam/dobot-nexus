"""FrameSample token projection, cross-attention and action-expert modulation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _rms(x: torch.Tensor, inference_ops: Any = None) -> torch.Tensor:
    # Preserve the reference FP32 normalization even inside AMP.
    with torch.autocast(x.device.type, enabled=False):
        value = x.float()
        inverse = (
            torch.reciprocal(torch.sqrt(value.square().mean(-1, keepdim=True) + 1e-6))
            if inference_ops is None
            else inference_ops.rms_inverse(x)
        )
        return value * inverse


class MemoryTokenEncoder(nn.Module):
    def __init__(self, compute_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.compute_dtype = compute_dtype
        self.pos_proj = nn.Linear(768, 768)
        self.encoder_static = nn.Linear(2816, 1024)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.pos_proj, self.encoder_static):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(
        self, image: torch.Tensor, position: torch.Tensor, inference_ops: Any = None
    ) -> torch.Tensor:
        dtype = self.compute_dtype
        linear = F.linear if inference_ops is None else inference_ops.linear
        silu = F.silu if inference_ops is None else inference_ops.silu
        pos = silu(
            linear(position.to(dtype), self.pos_proj.weight.to(dtype), self.pos_proj.bias.to(dtype))
        )
        return linear(
            torch.cat((image.to(dtype), pos), dim=-1),
            self.encoder_static.weight.to(dtype),
            self.encoder_static.bias.to(dtype),
        )


class MemoryAttention(nn.Module):
    def __init__(self, rope: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.rope = rope
        self.q_weight = nn.Parameter(torch.empty(4, 1024, 256))
        self.kv_weight = nn.Parameter(torch.empty(2, 1, 1024, 256))
        self.out_weight = nn.Parameter(torch.empty(4, 256, 1024))
        self.norm_scale = nn.Parameter(torch.zeros(1024))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Flax lecun_normal uses a variance-corrected normal truncated at 2 sigma.
        std = (1024**-0.5) / 0.87962566103423978
        for parameter in (self.q_weight, self.kv_weight, self.out_weight):
            nn.init.trunc_normal_(parameter, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.norm_scale)

    def prepare_memory(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project one request's history once, before the denoising loop."""
        mn = (_rms(memory) * (1 + self.norm_scale)).to(memory.dtype)
        k, v = torch.einsum("bsd,zkdh->zbskh", mn, self.kv_weight.to(memory.dtype)).unbind(0)
        kp = torch.arange(memory.shape[1], device=memory.device)[None].expand(memory.shape[0], -1)
        return self.rope(k, positions=kp), v

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        inference_ops: Any = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        s = memory.shape[1]
        dtype = x.dtype
        # One shared scale parameter normalizes both Q input and memory input.
        xn = (_rms(x, inference_ops) * (1 + self.norm_scale)).to(dtype)
        einsum = torch.einsum if inference_ops is None else inference_ops.einsum
        q = einsum("btd,ndh->btnh", xn, self.q_weight.to(dtype))
        k, v = self.prepare_memory(memory) if cached_kv is None else cached_kv
        qp = torch.arange(s, s + t, device=x.device)[None].expand(b, -1)
        q = (
            self.rope(q, positions=qp)
            if inference_ops is None
            else inference_ops.rope(q, positions=inference_ops.rope_angles(qp, 256))
        ) * (256**-0.5)
        q = q.reshape(b, t, 1, 4, 256)
        with torch.autocast(x.device.type, enabled=False):
            logits = einsum("btkgh,bskh->bkgts", q.float(), k.float())
            logits = torch.where(mask[:, None, None, None, :], logits, -2.3819763e38)
            softmax = torch.softmax if inference_ops is None else inference_ops.softmax
            probs = softmax(logits, dim=-1).to(dtype)
        encoded = einsum("bkgts,bskh->btkgh", probs, v.to(dtype)).reshape(b, t, 4, 256)
        return einsum("btnh,nhd->btd", encoded, self.out_weight.to(dtype))


class MemoryModulation(nn.Module):
    def __init__(self, rope: Callable[..., torch.Tensor]) -> None:
        super().__init__()
        self.attention = MemoryAttention(rope)
        self.modulation = nn.Linear(1024, 2048)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.attention.reset_parameters()
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        inference_ops: Any = None,
    ) -> torch.Tensor:
        context = self.attention(x, memory, mask, cached_kv=cached_kv, inference_ops=inference_ops)
        linear = F.linear if inference_ops is None else inference_ops.linear
        scale, shift = linear(
            context, self.modulation.weight.to(x.dtype), self.modulation.bias.to(x.dtype)
        ).chunk(2, dim=-1)
        normalized = _rms(x, inference_ops) * (1 + scale)
        return (
            normalized + shift if inference_ops is None else inference_ops.add(normalized, shift)
        ).to(x.dtype)
