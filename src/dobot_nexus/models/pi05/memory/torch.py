from __future__ import annotations

from dobot_nexus.modules.memory import torch_layers as _shared

F = _shared.F
MemoryAttention = _shared.MemoryAttention
MemoryModulation = _shared.MemoryModulation
MemoryTokenEncoder = _shared.MemoryTokenEncoder
_rms = _shared._rms
nn = _shared.nn
torch = _shared.torch


def install_memory(core, compute_dtype):
    """Called during meta construction, before loading or FSDP wrapping."""
    if core.llm.configs[-1].width != 1024:
        raise ValueError("FrameSample modulation requires the 1024-wide Pi0.5 action expert.")
    core.memory_encoder = MemoryTokenEncoder(compute_dtype)
    for block in core.llm.layers:
        # Reuse this model's eager/compiled RoPE selection, not a package global.
        block.memory_modulation = MemoryModulation(block.attn.rope)
