"""Dobot adjacent-bin two-hot cross entropy, computed in float32."""

import torch
import torch.nn.functional as F


def categorical_loss(logits, target_values, *, v_min=-1.0, v_max=0.0, num_bins=201):
    targets = target_values.float().reshape(-1)
    if (
        num_bins < 2
        or not v_min < v_max
        or logits.shape != (targets.numel(), num_bins)
        or targets.numel() == 0
        or not torch.isfinite(targets).all()
        or torch.any((targets < v_min) | (targets > v_max))
    ):
        raise ValueError("Expected finite normalized targets[B] and logits[B,num_bins].")
    centers = torch.linspace(v_min, v_max, num_bins, device=logits.device)
    step = centers[1] - centers[0]
    coordinate = (targets - centers[0]) / step
    lower = coordinate.floor().long().clamp(0, num_bins - 1)
    upper = (lower + 1).clamp(max=num_bins - 1)
    weight = (coordinate - lower.float()).clamp(0, 1)
    probs = torch.zeros_like(logits, dtype=torch.float32)
    probs.scatter_add_(1, lower[:, None], (1 - weight)[:, None])
    probs.scatter_add_(1, upper[:, None], weight[:, None])
    loss = -(probs * F.log_softmax(logits.float(), dim=-1)).sum(-1)
    predicted = (logits.float().softmax(-1) * centers).sum(-1)
    return loss, {"mae": (predicted - targets).abs().mean()}
