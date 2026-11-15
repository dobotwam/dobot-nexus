"""Resolve the model-owned Torch implementation and record its provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

UPSTREAM_REVISION = "89b0cd5fce528180559bbd50d055fb2e21a25bb5"


@dataclass(frozen=True)
class CoreSource:
    """Implementation types and provenance used by the Nexus model adapter."""

    repo_root: Path
    core_dir: Path
    package_name: str
    Pi0Config: type[Any]
    Observation: type[Any]
    revision: str | None
    embedded: bool


def load_core_source() -> CoreSource:
    """Import the model-owned core and record its provenance.

    Built-in modules have no import-time compilation. Their training helpers are
    configured per model by ``training.compile.configure_training_compile``.
    """
    # Load optional model dependencies only when constructing the Torch core.
    from . import pi0  # noqa: F401 -- load the complete architecture lazily
    from .model import Observation
    from .pi0_config import Pi0Config

    return CoreSource(
        repo_root=Path(__file__).resolve().parents[5],
        core_dir=Path(__file__).resolve().parent,
        package_name=__package__,
        Pi0Config=Pi0Config,
        Observation=Observation,
        revision=UPSTREAM_REVISION[:12],
        embedded=True,
    )
