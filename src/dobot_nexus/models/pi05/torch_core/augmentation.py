# Copyright 2024 Konrad Heidler (augmax operations).
# Copyright 2026 Dobot Nexus contributors.
# Licensed under Apache-2.0. See LICENSE and NOTICE.md.
"""Torch equivalent of OpenPI's augmax 0.4.1 training image operations.

Parameters, rather than PRNG bit streams, can be shared with the JAX reference.
Geometry is composed before one bilinear resampling; color operates in HSV.
"""

from __future__ import annotations

import math

import torch


def sample_parameters(image: torch.Tensor, *, rng: torch.Generator | None = None) -> torch.Tensor:
    """Return per-image dy, dx, radians, brightness, contrast, hue, color gate."""
    _, h, w, _ = image.shape
    device = rng.device if rng is not None else image.device
    values = torch.rand((image.shape[0], 7), generator=rng, device=device).to(image.device)
    scales = image.new_tensor(
        [h - int(h * 0.95), w - int(w * 0.95), math.pi / 18, 0.6, 0.8, 0.2, 1.0]
    )
    parameters = (values - 0.5) * scales
    parameters[:, 6] = values[:, 6] < 0.5
    return parameters


def _geometry(image: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    b, h, w, _ = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=torch.float32) - (h - 1) / 2,
        torch.arange(w, device=image.device, dtype=torch.float32) - (w - 1) / 2,
        indexing="ij",
    )
    dy, dx, angle = (parameters[:, i] for i in range(3))
    cos, sin = angle.cos(), angle.sin()
    # augmax applies T_crop @ S_resize @ R_rotate in centered y/x coordinates.
    sy, sx = int(h * 0.95) / h, int(w * 0.95) / w
    matrix = torch.stack([cos * sy, sin * sy, dy, -sin * sx, cos * sx, dx], -1).reshape(b, 2, 3)
    grid = torch.stack([yy, xx, torch.ones_like(yy)]).reshape(3, -1)
    coordinates = (matrix @ grid).reshape(b, 2, h, w)
    y, x = coordinates[:, 0] + (h - 1) / 2, coordinates[:, 1] + (w - 1) / 2
    y0, x0 = y.floor().long(), x.floor().long()
    wy, wx = y - y0, x - x0
    batch = torch.arange(b, device=image.device)[:, None, None]
    result = torch.zeros_like(image)
    # Equivalent to jax.scipy.ndimage.map_coordinates(order=1, mode='constant').
    for oy, fy in ((0, 1 - wy), (1, wy)):
        for ox, fx in ((0, 1 - wx), (1, wx)):
            iy, ix = y0 + oy, x0 + ox
            valid = (iy >= 0) & (iy < h) & (ix >= 0) & (ix < w)
            pixels = image[batch, iy.clamp(0, h - 1), ix.clamp(0, w - 1)]
            result = result + pixels * (fy * fx * valid)[..., None]
    return result


def _color(image: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    value, index = image.max(dim=-1)
    extent = value - image.min(dim=-1).values
    safe_extent = torch.where(extent == 0, 1, extent)
    following = image.gather(-1, ((index + 1) % 3)[..., None])[..., 0]
    preceding = image.gather(-1, ((index + 2) % 3)[..., None])[..., 0]
    hue = torch.where(extent == 0, 0, (2 * index + (following - preceding) / safe_extent) / 6)
    saturation = extent / torch.where(value == 0, 1, value)
    brightness, contrast, shift = (parameters[:, i, None, None] for i in range(3, 6))
    value = torch.where(
        brightness < 0, value * (1 + brightness), value * (1 - brightness) + brightness
    )
    slant = torch.tan((contrast + 1) * (math.pi / 4))
    # Algebraically equivalent to augmax, also defined at contrast=0.
    p1 = slant / (2 * (1 + slant))
    value = torch.where(
        value < p1,
        value / slant,
        torch.where(value > 1 - p1, value / slant + 1 - 1 / slant, slant * (value - 0.5) + 0.5),
    )
    # augmax 0.4.1 ColorJitter defaults hue=.1, p=.5. Its saturation call
    # discards the returned value, so saturation remains unchanged in this recipe.
    k = (image.new_tensor([5, 3, 1]) + (hue + shift)[..., None] * 6) % 6
    color = value[..., None] * (1 - saturation[..., None] * torch.minimum(k, 4 - k).clamp(0, 1))
    return torch.where(parameters[:, 6, None, None, None].bool(), color, image)


def augment_image(
    image: torch.Tensor,
    *,
    geometry: bool,
    rng: torch.Generator | None = None,
    parameters: torch.Tensor | None = None,
) -> torch.Tensor:
    """Augment normalized float32 BHWC images; wrists skip geometry."""
    if parameters is None:
        parameters = sample_parameters(image, rng=rng)
    image01 = image.float() / 2 + 0.5
    if geometry:
        image01 = _geometry(image01, parameters)
    result = _color(image01, parameters) * 2 - 1
    if not geometry:
        result = torch.where(parameters[:, 6, None, None, None].bool(), result, image)
    return result
