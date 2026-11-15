"""Image layout and dtype conversion shared by dataset consumers."""

import numpy as np
import torch


def convert_image(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
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
