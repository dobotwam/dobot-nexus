"""Native timestep projections shared by inference numerical profiles."""

import torch


def prepare_modulations(core, observation, noise, num_steps):
    """Cache native condition projections for one immutable inference graph.

    Use the sampler's repeated FP64 Python time additions and FP32 tensor
    construction exactly. Batching all timesteps into one GEMM could change
    accumulation order, so each step keeps the original inference batch size.
    """
    if core.llm.adarms != [False, True]:
        raise ValueError("Modulation caching requires the Pi0.5 action expert.")
    norms = [
        norm
        for layer in core.llm.layers
        for norm in (layer.pre_attention_norms[-1], layer.pre_ffw_norms[-1])
    ]
    norms.append(core.llm.final_norms[-1])
    outputs = []
    dt, time = -1.0 / num_steps, 1.0
    while time >= -dt / 2:
        timestep = torch.full((noise.shape[0],), time, device=noise.device, dtype=torch.float32)
        # Reuse embed_suffix so input casts, sin/cos and both time MLPs retain
        # their original operator ordering. The action tokens are discarded.
        _, _, _, condition = core.embed_suffix(observation, noise, timestep)
        outputs.append(
            tuple(
                norm.ada_modulation(
                    condition.to(norm.ada_modulation.weight.dtype),
                )
                for norm in norms
            )
        )
        time += dt
    return tuple(outputs)
