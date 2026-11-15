"""Instance-local compilation of the three original Torch training helpers."""

from functools import cache, partial

import torch

from ..torch_core import gemma, lora
from ..torch_core.utils import gelu_glu


@cache
def _compiled_helpers():
    # Preserve the original compilation boundaries and default compiler options.
    return (
        torch.compile(gemma._apply_rope),
        torch.compile(gemma._fused_gated_residual),
        torch.compile(gelu_glu),
    )


def configure_training_compile(core: torch.nn.Module, *, enabled: bool) -> None:
    """Select helpers before FSDP wrapping; never mutate imports or other models.

    Only call before execution/capture. Inference acceleration is owned by
    ``openpi_torch.engine`` and starts from the eager helper implementations.
    Callables are ordinary attributes, so parameter and state-dict keys survive.
    """
    rope, residual, activation = (
        _compiled_helpers() if enabled
        else (gemma._apply_rope, gemma._fused_gated_residual, gelu_glu)
    )
    for module in core.modules():
        if isinstance(module, gemma.Attention):
            module.rope = rope
        elif isinstance(module, gemma.Block):
            module.gated_residual = partial(gemma._gated_residual, fused=residual)
        elif isinstance(module, (gemma.FeedForward, lora.FeedForward)):
            module.activation = activation
