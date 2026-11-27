"""Restricted on-disk representation for RLToken records and training state."""

import pickle
from collections import OrderedDict
from dataclasses import fields

import numpy as np
import torch

from dobot_nexus.core.contracts import EncodedActionBatch, RLFeatureBatch, RLFeatureSpec

from .identity import scalar_value
from .configuration import HeadConfig, LearningConfig
from .transitions import ObservationID, Transition

_RECORDS = {cls.__name__: cls for cls in (
    Transition, ObservationID, RLFeatureBatch, EncodedActionBatch, RLFeatureSpec,
    HeadConfig, LearningConfig,
)}
_TAG = "__rltoken_record_v1__"


def _encode(value):
    kind = type(value)
    if kind in _RECORDS.values():
        return {_TAG: kind.__name__, "fields": {
            f.name: _encode(getattr(value, f.name)) for f in fields(value)
        }}
    if isinstance(value, np.generic):
        return _encode(scalar_value(value))
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc":
            raise TypeError("RLToken array storage requires a numeric or boolean dtype.")
        return {_TAG: "ndarray", "dtype": value.dtype.str,
                "shape": value.shape, "data": value.tobytes()}
    if isinstance(value, dict):
        items = {_encode(key): _encode(item) for key, item in value.items()}
        if isinstance(value, OrderedDict) and hasattr(value, "_metadata"):
            # Module version metadata is consulted by load_state_dict migration hooks.
            return {_TAG: "state_dict", "items": items, "metadata": _encode(value._metadata)}
        return {_TAG: "mapping", "items": items} if _TAG in items else items
    if kind in (list, tuple):
        return kind(_encode(item) for item in value)
    if value is None or kind in (str, bool, int, float, bytes) or kind is torch.Tensor:
        return value
    raise TypeError(f"Unsupported RLToken storage type {kind.__name__}.")


def _decode(value):
    if isinstance(value, dict):
        tag = value.get(_TAG)
        if tag == "ndarray":
            dtype = np.dtype(value["dtype"])
            if dtype.kind not in "biufc" or not isinstance(value["data"], bytes):
                raise ValueError("Invalid RLToken numeric array.")
            return np.frombuffer(value["data"], dtype).reshape(value["shape"]).copy()
        if tag == "state_dict":
            state = OrderedDict((k, _decode(v)) for k, v in value["items"].items())
            state._metadata = _decode(value["metadata"])
            return state
        if tag == "mapping":
            value = value["items"]
        elif tag is not None:
            return _RECORDS[tag](**{k: _decode(v) for k, v in value["fields"].items()})
        items = {k: _decode(v) for k, v in value.items()}
        if isinstance(value, OrderedDict) and hasattr(value, "_metadata"):
            # Older tensor-only token resume files already use Torch's native state dict.
            state = OrderedDict(items)
            state._metadata = value._metadata
            return state
        return items
    if isinstance(value, (list, tuple)):
        return type(value)(_decode(item) for item in value)
    return value


def save_payload(value, destination):
    torch.save(_encode(value), destination)


def load_payload(source, *, trusted_checkpoint: bool = False, trust_setting="train.trust_resume_checkpoint"):
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        if trusted_checkpoint is not True:
            raise ValueError(
                "Restricted RLToken load failed. Rebuild the file or, only for a trusted "
                f"legacy pickle, pass trusted_checkpoint=True or set {trust_setting}=true for this input."
            ) from exc
        if hasattr(source, "seek"):
            source.seek(0)
        return torch.load(source, map_location="cpu", weights_only=False)
    return _decode(payload)
