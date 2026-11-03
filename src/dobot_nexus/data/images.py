"""Image layout and dtype conversion shared by dataset consumers."""

from __future__ import annotations

import sys

import numpy as np


def image_uint8(value, *, root=None):
    """Decode a dataset image, validating RGB channels and pixel ranges."""
    if isinstance(value, dict):
        import io
        from pathlib import Path
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Embedded image decoding requires Pillow.") from exc
        payload = value.get("bytes")
        if payload is None and value.get("path") and root is not None:
            payload = (Path(root) / value["path"]).read_bytes()
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("Embedded image requires bytes or a path with its dataset root.")
        with Image.open(io.BytesIO(payload)) as image:
            value = np.array(image.convert("RGB"))
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError("Image must be CHW or HWC.")
    if array.shape[-1] not in (1, 3, 4) and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError("Image must have three finite RGB channels.")
    if np.issubdtype(array.dtype, np.floating):
        if array.min() < 0 or array.max() > 1:
            raise ValueError("Floating dataset images must be in [0,1].")
        array = array * 255
    if array.min() < 0 or array.max() > 255:
        raise ValueError("Integer dataset images must be in [0,255].")
    return np.ascontiguousarray(array, dtype=np.uint8)


def convert_image(value) -> np.ndarray:
    """Convert RGB input to contiguous HWC uint8, scaling or clipping as needed."""
    # Passing a Torch tensor implies the caller has already loaded Torch.
    torch = sys.modules.get("torch")
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
