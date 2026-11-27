"""Content hashes independent of tensor storage IDs and filesystem locations."""

import dataclasses
import hashlib
import json

import numpy as np
import torch


def scalar_value(value):
    """Represent JSON-compatible NumPy scalars identically in storage and hashes."""
    result = value.item()
    if type(result) not in (bool, int, float, str):
        raise TypeError(f"Unsupported RLToken scalar dtype {value.dtype}.")
    return result


def content_digest(value):
    digest = hashlib.sha256()

    def visit(item):
        if dataclasses.is_dataclass(item):
            visit(
                {
                    field.name: getattr(item, field.name)
                    for field in dataclasses.fields(item)
                    if not (
                        field.metadata.get("digest_omit_none") and getattr(item, field.name) is None
                    )
                }
            )
        elif isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            visit({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.generic):
            visit(scalar_value(item))
        elif isinstance(item, np.ndarray):
            visit({"numpy_dtype": str(item.dtype), "shape": list(item.shape)})
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, sort_keys=True, separators=(",", ":")).encode())
            digest.update(b";")

    visit(value)
    return digest.hexdigest()
