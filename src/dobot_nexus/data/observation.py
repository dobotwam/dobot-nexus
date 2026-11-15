"""JSON observation decoding shared by offline and HTTP inference."""
from __future__ import annotations

from typing import Any

import numpy as np

from dobot_nexus.core.contracts import ObservationBatch


def observation_from_json(value: Any) -> ObservationBatch:
    """Restore image dtypes that JSON cannot represent before validation."""
    if not isinstance(value, dict):
        raise TypeError(f"observation must be an object, got {type(value)!r}.")
    payload = dict(value)
    raw_images = payload.get("images")
    if not isinstance(raw_images, dict):
        raise TypeError("observation.images must be an object.")
    images: dict[str, np.ndarray] = {}
    for name, raw_image in raw_images.items():
        image = np.asarray(raw_image)
        if not np.issubdtype(image.dtype, np.integer):
            raise TypeError(
                f"JSON image {name!r} must contain integer values in [0, 255], "
                f"got {image.dtype}."
            )
        if image.size and (np.min(image) < 0 or np.max(image) > 255):
            raise ValueError(f"JSON image {name!r} contains values outside [0, 255].")
        images[str(name)] = image.astype(np.uint8, copy=False)
    payload["images"] = images
    return ObservationBatch.from_value(payload)

