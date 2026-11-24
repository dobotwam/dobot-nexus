"""Image layout and dtype conversion shared by dataset consumers."""

from __future__ import annotations

import numpy as np


def convert_image(value) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            value = value.numpy()
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Image must be 3-D, got {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Image must be RGB, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        maximum = float(np.nanmax(image)) if image.size else 0.0
        scale = 255.0 if maximum <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def resize_with_pad(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize an RGB image with PIL bilinear interpolation and centered black padding."""
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        image = convert_image(image)
    if image.shape[:2] == (size, size):
        return image
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Image resizing requires pip install Pillow.") from exc
    source = Image.fromarray(image)
    width, height = source.size
    ratio = max(width / size, height / size)
    resized_width = max(1, int(width / ratio))
    resized_height = max(1, int(height / ratio))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = source.resize((resized_width, resized_height), resample=resampling)
    padded = Image.new("RGB", (size, size), 0)
    padded.paste(resized, ((size - resized_width) // 2, (size - resized_height) // 2))
    return np.asarray(padded, dtype=np.uint8)
