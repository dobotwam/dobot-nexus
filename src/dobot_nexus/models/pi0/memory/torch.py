"""Only the new FrameSample operators; the Pi0 backbone remains shared."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _rms(x):
    # Preserve the reference FP32 normalization even inside AMP.
    with torch.autocast(x.device.type, enabled=False):
        value = x.float()
        inverse = torch.reciprocal(torch.sqrt(value.square().mean(-1, keepdim=True) + 1e-6))
        return value * inverse


class MemoryTokenEncoder(nn.Module):
    def __init__(self, compute_dtype=torch.bfloat16):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.pos_proj = nn.Linear(768, 768)
        self.encoder_static = nn.Linear(2816, 1024)
        self.reset_parameters()

    def reset_parameters(self):
        for module in (self.pos_proj, self.encoder_static):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, image, position):
        dtype = self.compute_dtype
        pos = F.silu(
            F.linear(
                position.to(dtype), self.pos_proj.weight.to(dtype), self.pos_proj.bias.to(dtype)
            )
        )
        return F.linear(
            torch.cat((image.to(dtype), pos), dim=-1),
            self.encoder_static.weight.to(dtype),
            self.encoder_static.bias.to(dtype),
        )


class MemoryAttention(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.rope = rope
        self.q_weight = nn.Parameter(torch.empty(4, 1024, 256))
        self.kv_weight = nn.Parameter(torch.empty(2, 1, 1024, 256))
        self.out_weight = nn.Parameter(torch.empty(4, 256, 1024))
        self.norm_scale = nn.Parameter(torch.zeros(1024))
        self.reset_parameters()

    def reset_parameters(self):
        # Flax lecun_normal uses a variance-corrected normal truncated at 2 sigma.
        std = (1024**-0.5) / 0.87962566103423978
        for parameter in (self.q_weight, self.kv_weight, self.out_weight):
            nn.init.trunc_normal_(parameter, std=std, a=-2 * std, b=2 * std)
        nn.init.zeros_(self.norm_scale)

    def prepare_memory(self, memory):
        """Project one request's history once, before the denoising loop."""
        mn = (_rms(memory) * (1 + self.norm_scale)).to(memory.dtype)
        k, v = torch.einsum("bsd,zkdh->zbskh", mn, self.kv_weight.to(memory.dtype)).unbind(0)
        kp = torch.arange(memory.shape[1], device=memory.device)[None].expand(memory.shape[0], -1)
        return self.rope(k, positions=kp), v

    def forward(self, x, memory, mask, cached_kv=None):
        b, t, _ = x.shape
        s = memory.shape[1]
        dtype = x.dtype
        # One shared scale parameter normalizes both Q input and memory input.
        xn = (_rms(x) * (1 + self.norm_scale)).to(dtype)
        q = torch.einsum("btd,ndh->btnh", xn, self.q_weight.to(dtype))
        k, v = self.prepare_memory(memory) if cached_kv is None else cached_kv
        qp = torch.arange(s, s + t, device=x.device)[None].expand(b, -1)
        q = self.rope(q, positions=qp) * (256**-0.5)
        q = q.reshape(b, t, 1, 4, 256)
        with torch.autocast(x.device.type, enabled=False):
            logits = torch.einsum("btkgh,bskh->bkgts", q.float(), k.float())
            logits = torch.where(mask[:, None, None, None, :], logits, -2.3819763e38)
            probs = torch.softmax(logits, dim=-1).to(dtype)
        encoded = torch.einsum("bkgts,bskh->btkgh", probs, v.to(dtype)).reshape(b, t, 4, 256)
        return torch.einsum("btnh,nhd->btd", encoded, self.out_weight.to(dtype))


class MemoryModulation(nn.Module):
    def __init__(self, rope):
        super().__init__()
        self.attention = MemoryAttention(rope)
        self.modulation = nn.Linear(1024, 2048)
        self.reset_parameters()

    def reset_parameters(self):
        self.attention.reset_parameters()
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, memory, mask, cached_kv=None):
        context = self.attention(x, memory, mask, cached_kv=cached_kv)
        scale, shift = F.linear(
            context, self.modulation.weight.to(x.dtype), self.modulation.bias.to(x.dtype)
        ).chunk(2, dim=-1)
        normalized = _rms(x) * (1 + scale)
        return (normalized + shift).to(x.dtype)


def install_memory(core, compute_dtype):
    """Called during meta construction, before loading or FSDP wrapping."""
    if core.llm.configs[-1].width != 1024:
        raise ValueError("FrameSample modulation requires the 1024-wide Pi0 action expert.")
    core.memory_encoder = MemoryTokenEncoder(compute_dtype)
    for block in core.llm.layers:
        # Reuse this model's eager/compiled RoPE selection, not a package global.
        block.memory_modulation = MemoryModulation(block.attn.rope)
