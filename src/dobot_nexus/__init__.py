"""Dobot Nexus: a thin, algorithm-facing VLA mapping layer."""

from dobot_nexus.core.contracts import (
    ObservationBatch,
    PolicyOutput,
    PolicySpec,
    SamplingOptions,
    TrainOutput,
    TrainResult,
)

__all__ = [
    "ObservationBatch",
    "PolicyOutput",
    "PolicySpec",
    "SamplingOptions",
    "TrainOutput",
    "TrainResult",
]

__version__ = "1.0.0"

