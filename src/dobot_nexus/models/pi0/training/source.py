"""Load the vendored aligned PyTorch Pi operator without RLinf runtime dependencies."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EXTERNAL_CORE_RELATIVE_PATH = Path(
    "rlinf/models/embodiment/openpi_rlinf/pi0_model"
)
_EMBEDDED_CORE_DIR = Path(__file__).resolve().parent / "vendor" / "rlinf_core"
_EMBEDDED_REVISION = "89b0cd5fce528180559bbd50d055fb2e21a25bb5"
_LOAD_LOCK = threading.RLock()
_CACHE: dict[tuple[Path, bool], CoreSource] = {}


@dataclass(frozen=True)
class CoreSource:
    """Resolved core package and the types required by the Pi0 adapter."""

    repo_root: Path
    core_dir: Path
    package_name: str
    Pi0Config: type[Any]
    Observation: type[Any]
    revision: str | None
    embedded: bool


def _revision(repo_root: Path) -> str | None:
    head = repo_root / ".git" / "HEAD"
    if not head.is_file():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if not value.startswith("ref: "):
        return value[:12]
    ref_file = repo_root / ".git" / value.removeprefix("ref: ")
    if ref_file.is_file():
        return ref_file.read_text(encoding="utf-8").strip()[:12]
    return None


def _external_core_dir(repo_root: str | Path) -> tuple[Path, Path]:
    root = Path(repo_root).expanduser().resolve()
    direct = (
        root
        if (root / "__init__.py").is_file()
        else root / _EXTERNAL_CORE_RELATIVE_PATH
    )
    if not (direct / "__init__.py").is_file():
        raise FileNotFoundError(f"External aligned Pi core was not found under {root}.")
    return root, direct.resolve()


def _resolve_source(
    config: Any | None,
    repo_root: str | Path | None,
) -> tuple[Path, Path, str | None, bool]:
    external = repo_root
    if external is None and config is not None:
        # Compatibility only for old resolved configs. New configs always use
        # the vendored module and need no RLinf path or environment variable.
        external = config.get("dependencies", {}).get("rlinf_path")
    if external is not None:
        root, core_dir = _external_core_dir(external)
        return root, core_dir, _revision(root), False

    if not (_EMBEDDED_CORE_DIR / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Vendored aligned Pi core is incomplete: {_EMBEDDED_CORE_DIR}."
        )
    repository_root = Path(__file__).resolve().parents[5]
    return repository_root, _EMBEDDED_CORE_DIR.resolve(), _EMBEDDED_REVISION[:12], True


@contextmanager
def _compile_mode(enabled: bool):
    import torch

    if enabled or not hasattr(torch, "compile"):
        yield
        return
    original = torch.compile

    def eager_compile(function=None, *_args, **_kwargs):
        return (lambda value: value) if function is None else function

    torch.compile = eager_compile
    try:
        yield
    finally:
        torch.compile = original


def load_core_source(
    config: Any | None = None,
    *,
    repo_root: str | Path | None = None,
    enable_compile: bool = False,
) -> CoreSource:
    """Load the vendored core; an explicit external checkout is compatibility-only.

    The core is loaded under a private package name so its relative imports remain
    self-contained. It cannot import ``rlinf``, Ray, workers, or schedulers.
    """
    root, core_dir, revision, embedded = _resolve_source(config, repo_root)
    cache_key = (core_dir, bool(enable_compile))
    with _LOAD_LOCK:
        cached = _CACHE.get(cache_key)
        if cached is not None:
            return cached

        digest = hashlib.sha256(str(core_dir).encode("utf-8")).hexdigest()[:12]
        package_name = f"_dobot_nexus_pi_ops_{digest}_{int(enable_compile)}"
        init_path = core_dir / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[str(core_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create aligned Pi core spec from {init_path}.")

        try:
            with _compile_mode(enable_compile):
                package = importlib.util.module_from_spec(spec)
                sys.modules[package_name] = package
                spec.loader.exec_module(package)
                config_module = importlib.import_module(f"{package_name}.pi0_config")
                model_module = importlib.import_module(f"{package_name}.model")
                importlib.import_module(f"{package_name}.pi0")
        except Exception:
            for module_name in tuple(sys.modules):
                if module_name == package_name or module_name.startswith(package_name + "."):
                    sys.modules.pop(module_name, None)
            raise

        source = CoreSource(
            repo_root=root,
            core_dir=core_dir,
            package_name=package_name,
            Pi0Config=config_module.Pi0Config,
            Observation=model_module.Observation,
            revision=revision,
            embedded=embedded,
        )
        _CACHE[cache_key] = source
        return source
