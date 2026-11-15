"""Decode categorical value predictions on uniformly spaced support points."""

import torch


def expected_value(logits, *, v_min=-1.0, v_max=0.0, num_bins=201):
    if logits.ndim != 2 or logits.shape[-1] != num_bins or num_bins < 2 or not v_min < v_max:
        raise ValueError("Expected logits[B,num_bins] and ordered value bounds.")
    centers = torch.linspace(v_min, v_max, num_bins, device=logits.device)
    return (logits.float().softmax(-1) * centers).sum(-1)
