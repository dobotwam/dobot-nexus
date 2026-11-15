"""Dobot positive/negative/base task conditioning and condition dropout."""

from __future__ import annotations

import re

import torch

from ._validation import nonempty, number

POSITIVE_TAG = "Advantage: positive"
NEGATIVE_TAG = "Advantage: negative"
_RESERVED = re.compile(r"\badvantage\s*:", re.IGNORECASE)


def condition_prompt(task: str, condition: str) -> str:
    nonempty(task, "task")
    if _RESERVED.search(task):
        raise ValueError("Task already contains the reserved Advantage: condition.")
    if condition == "base":
        return task
    if condition in {"positive", "negative"}:
        return task + "\n" + (POSITIVE_TAG if condition == "positive" else NEGATIVE_TAG)
    raise ValueError("RECAP condition must be positive, negative or base.")


def select_condition(positive, *, unconditional_prob: float, random_values=None):
    if not isinstance(positive, torch.Tensor) or positive.dtype != torch.bool or positive.ndim != 1:
        raise ValueError("RECAP positive labels must be bool[B].")
    q = number(unconditional_prob, "unconditional_prob")
    if not 0 <= q <= 1:
        raise ValueError("unconditional_prob must be in [0,1].")
    u = (
        torch.rand(positive.shape, device=positive.device)
        if random_values is None
        else random_values
    )
    if (
        u.shape != positive.shape
        or u.device != positive.device
        or not u.is_floating_point()
        or not torch.isfinite(u).all()
        or torch.any((u < 0) | (u > 1))
    ):
        raise ValueError("Condition random values must be finite [0,1] floats matching bool[B].")
    return (u >= q) if q < 1 else torch.zeros_like(positive)
