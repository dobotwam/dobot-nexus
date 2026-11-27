"""Self-contained v2 deployment composition through existing model export hooks."""

from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import torch
import yaml

from dobot_nexus.core.artifacts import (
    _deployment_files,
    artifact_weight_path,
    load_manifest,
    resolve_artifact,
)
from dobot_nexus.core.config import clone_config, load_config, load_config_bytes, save_config
from dobot_nexus.core.contracts import RLFeatureSpec

from .identity import content_digest


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seal_path(root):
    return root.with_name(root.name + ".sha256")


def _manifest_seal(root):
    path = _seal_path(root)
    try:
        mode = _file_identity(path).mode
    except FileNotFoundError:
        return ()
    digests = tuple(path.read_text(encoding="ascii").removesuffix("\n").split("\n"))
    if (mode & 0o222 or len(digests) not in (1, 2)
            or not all(re.fullmatch(r"[a-f0-9]{64}", d) for d in digests)):
        raise ValueError(f"Invalid retained feature manifest seal: {path}")
    return digests


def _read_feature_manifest(root, *, run_owned=False, enclosing=None):
    """Authenticate the bytes before interpreting the child file mappings."""
    path = root / "manifest.yaml"
    _file_identity(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    anchored = False
    if run_owned:
        digests = _manifest_seal(root)
        if digests:
            if digest not in digests:
                # A reader can span the entire old-to-new manifest transaction.
                _file_identity(path)
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
            if digest not in digests:
                raise ValueError(f"Retained feature manifest seal checksum mismatch: {path}")
            anchored = True
    if enclosing is not None and "feature_manifest_sha256" in enclosing:
        if enclosing["feature_manifest_sha256"] != digest:
            raise ValueError("Deployment feature manifest checksum mismatch.")
        anchored = True
    manifest = yaml.safe_load(data)
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in {1, 2}:
        raise ValueError(f"Unsupported feature manifest: {path}")
    if ((run_owned or enclosing is not None) and not anchored
            and "retained_files_sha256" in manifest):
        raise ValueError(f"Missing independent feature manifest checksum: {path}")
    return manifest, digest


def _enclosing_deployment_manifest(root):
    if root.parts[-2:] != ("deployment", "features"):
        return None
    outer = root.parents[1]
    if not stat.S_ISDIR(root.parent.lstat().st_mode) or not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError(f"Invalid deployment feature directory: {root}")
    try:
        _file_identity(outer / "manifest.yaml")
    except FileNotFoundError:
        return None
    enclosing, _ = _read_feature_manifest(outer)
    if enclosing.get("files", {}).get("feature_manifest") == "deployment/features/manifest.yaml":
        return enclosing
    return None


def _read_retained_source_manifest(root):
    enclosing = _enclosing_deployment_manifest(root)
    if enclosing is not None:
        return _read_feature_manifest(root, enclosing=enclosing)
    return _read_feature_manifest(root, run_owned=True)


def _feature_spec(manifest):
    if "feature_manifest" in manifest.get("files", {}):
        raise ValueError("A composed RLToken policy cannot itself be a feature provider.")
    metadata = manifest["model"]
    if "rl_features" not in metadata:
        raise ValueError("Feature Artifact does not declare the optional RL feature capability.")
    return RLFeatureSpec.from_dict(metadata["rl_features"])


def select_feature_manifest(path):
    root = resolve_artifact(path)
    _, manifest, digest = _source_manifest(path, root)
    return root, _feature_spec(manifest), manifest["model"], digest


def feature_metadata(path):
    return select_feature_manifest(path)[:3]


def _source_manifest(path, root):
    retained = _retained_source(path, root)
    if retained is not None:
        if retained.resolve() != root or not stat.S_ISDIR(retained.lstat().st_mode):
            raise ValueError(f"Invalid retained feature directory or redirect: {retained}")
        return retained, *_read_retained_source_manifest(retained)
    enclosing = _enclosing_deployment_manifest(root)
    if enclosing is not None:
        return root, *_read_feature_manifest(root, enclosing=enclosing)
    return None, *_read_feature_manifest(root)


def _load_artifact_config(path, *, expected_task="train"):
    _file_identity(path)
    return load_config(path, expected_task=expected_task, allow_extends=False)


def _config_strings(value, key, excluded):
    if key in excluded:
        return
    if isinstance(value, dict):
        for name, child in value.items():
            yield from _config_strings(child, f"{key}.{name}", excluded)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _config_strings(child, key, excluded)
    elif isinstance(value, str) and value:
        yield key, value


def _names_asset_path(value, root):
    path = Path(value)
    try:
        path = path.expanduser()
    except RuntimeError:
        pass  # A literal string can start with a nonexistent user's name.
    for candidate in ((path,) if path.is_absolute() else (root / path, path)):
        try:
            if candidate.is_symlink() or candidate.is_file() or candidate.is_dir():
                return True
        except OSError as exc:
            if exc.errno != errno.ENAMETOOLONG:
                raise
    return False


def _check_constructor_assets(config, allowed, root):
    data = config.get("data", {})
    backend = config.model.get("backend")
    legacy = backend.get("paligemma_tokenizer") if isinstance(backend, dict) else None
    for key, value in (
        ("data.norm_stats_path", data.get("norm_stats_path")),
        ("data.tokenizer_path", data.get("tokenizer_path") or legacy),
    ):
        if value:
            _checked_asset_path(key, value, allowed)

    skip = {
        "model.target", "model.name", "model.artifact", "model.checkpoint",
        "model.jax_checkpoint", "model.jax_export_base_checkpoint", "model.openpi_root",
        "model.precision", "model.parameter_precision", "model.inference_parameter_precision",
        "model.operator_numerics", "model.training_recipe", "model.io.cameras",
        "model.rltoken.action_codec", "model.rltoken.token",
        "data.default_prompt", "data.format", "data.format_version", "data.normalization",
        "data.field_mapping", "data.transforms", "data.path", "data.raw_path",
        "data.cache_path", "data.offline_cache", "data.norm_stats_cache_dir",
    }
    if data.get("tokenizer_path"):
        skip.add("model.backend.paligemma_tokenizer")
    for section in ("model", "data"):
        for key, value in _config_strings(config.get(section, {}), section, skip):
            if key in allowed:
                _checked_asset_path(key, value, allowed)
            elif _names_asset_path(value, root):
                raise ValueError(f"Undeclared external asset path at {key}: {value}")


def _check_loaded_provider_files(root, before, rehash):
    """Recheck constructor inputs without a second read of the weight files."""
    error = "Feature artifact file changed during provider load."
    for rel, expected in before.items():
        path = root / rel
        try:
            unchanged = (
                not any((root / p).is_symlink() for p in reversed(rel.parents))
                and _file_identity(path) == expected.identity
            )
            if unchanged and rel in rehash:
                unchanged = (
                    file_digest(path) == expected.sha256
                    and _file_identity(path) == expected.identity
                )
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            raise ValueError(error) from exc
        if not unchanged:
            raise ValueError(error)


def load_provider(path, *, device="cpu", _verified_files=None, expected_manifest_sha256=None):
    from dobot_nexus.inference.policy import InferencePolicy, _set_config_path

    root = resolve_artifact(path)
    retained, manifest, digest = _source_manifest(path, root)
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError("Selected feature source manifest checksum mismatch.")
    spec = _feature_spec(manifest)
    allowed = _validate_feature_inputs(root, manifest, spec, stage_one=True)
    cached = {}
    if _verified_files is not None and _verified_files[0] == root:
        cached = _verified_files[1]
    relative = Path("config_resolved.yaml")
    config_path = root / relative
    identity = _file_identity(config_path)
    config_bytes = config_path.read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    previous = cached.get(relative)
    if (identity != _file_identity(config_path)
            or (previous is not None and previous.sha256 != config_sha)):
        owner = "Retained" if retained is not None else "Source"
        raise ValueError(f"{owner} config_resolved.yaml changed during load.")
    cached = {**cached, relative: _VerifiedFile(config_sha, identity)}
    if retained is not None:
        declared = manifest.get("retained_files_sha256")
        if declared is not None and (
            not isinstance(declared, dict) or declared.get(str(relative)) != config_sha
        ):
            raise ValueError("Retained config_resolved.yaml changed during load.")
        verified = _verify_retained_feature_bundle(
            root, spec, manifest_digest=digest, cached=cached
        )
        if verified[relative].sha256 != config_sha:
            raise ValueError("Retained config_resolved.yaml changed during load.")
    else:
        verified = {**cached, **_verify_source_weights(root, manifest, cached)}
    bound_training_config = load_config_bytes(
        config_bytes, config_path, expected_task="train", allow_extends=False
    )
    for key, bundled in allowed.items():
        _set_config_path(bound_training_config, key, str(bundled))
    _check_constructor_assets(bound_training_config, allowed, root)
    config = clone_config(bound_training_config)
    config.task = "infer"
    config.setdefault("runtime", {}).update(device=str(device), prefer_bundled_assets=True)
    rehash = {path.relative_to(root) for path in allowed.values()} | {relative}
    inputs = rehash | {Path(_weight_path(manifest))}
    if "nexus_weights" in manifest.get("files", {}):
        inputs.add(Path(manifest["files"]["nexus_weights"]))
    # Keep the entries from the original verification, even if a path has since been replaced.
    before = {
        rel: verified[rel] if rel in verified else _verified_file(root / rel, None)
        for rel in inputs
    }
    policy = InferencePolicy(config, _preloaded_artifact=(root, manifest, bound_training_config))
    _check_loaded_provider_files(root, before, rehash)
    _export_assets(policy.model, policy.config, allowed)
    actual = RLFeatureSpec.from_dict(policy.model.artifact_metadata()["rl_features"])
    if actual != spec:
        raise ValueError("Loaded provider feature identity/spec differs from its Artifact.")
    return policy.model, policy.config, spec


def _copy_preserving_ceiling(src, dst, *, strip_write=False, ceiling=None):
    with src.open("rb") as source:
        src_mode = stat.S_IMODE(os.fstat(source.fileno()).st_mode)
        if ceiling is not None:
            src_mode &= ceiling
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        with os.fdopen(os.open(dst, flags, (src_mode & 0o777) | 0o200), "wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
            target.flush()
            created = stat.S_IMODE(os.fstat(target.fileno()).st_mode)
            os.fchmod(target.fileno(), created & src_mode & (0o555 if strip_write else 0o777))


def _own_file_mode(path, *, expected_digest=None):
    """Detach a hook-created hardlink before changing the inode's contents or mode."""
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"Expected a regular feature file: {path}")
    if st.st_nlink == 1:
        return stat.S_IMODE(st.st_mode)
    with tempfile.TemporaryDirectory(dir=path.parent) as private:
        copy = Path(private) / "file"
        _copy_preserving_ceiling(path, copy)
        if expected_digest is not None and file_digest(copy) != expected_digest:
            raise ValueError(f"Feature file changed while detaching hardlink: {path}")
        os.replace(copy, path)
    return _file_identity(path).mode


def _checked_asset_path(key, value, allowed):
    source = Path(value)
    if key == "data.norm_stats_path" and source.is_dir():
        source /= "norm_stats.json"
    if allowed is not None:
        absolute = source.absolute()
        if (absolute != allowed.get(key)
                or any(p.is_symlink() for p in (absolute, *absolute.parents))):
            raise ValueError(f"Undeclared or symlinked feature asset {key}: {source}")
    return source


def _export_assets(model, config, allowed_asset_files):
    assets = dict(model.artifact_assets())
    stats_key = "data.norm_stats_path"
    cfg_norm = config.get("data", {}).get("norm_stats_path")
    if cfg_norm:
        cfg_path = _checked_asset_path(stats_key, cfg_norm, allowed_asset_files)
        if stats_key in assets:
            model_path = _checked_asset_path(stats_key, assets[stats_key], allowed_asset_files)
            if model_path.resolve() != cfg_path.resolve():
                raise ValueError("Model and config disagree on data.norm_stats_path.")
        else:
            assets[stats_key] = str(cfg_path)
    result = {}
    for key, value in assets.items():
        source = _checked_asset_path(key, value, allowed_asset_files)
        if not source.is_file():
            raise FileNotFoundError(f"Required model asset {key}: {source}")
        result[key] = source
    if allowed_asset_files is not None:
        missing = allowed_asset_files.keys() - result.keys()
        if missing:
            raise ValueError(f"Export omitted declared feature assets: {sorted(missing)}")
    return result


def _bundle_asset_path(key, source):
    name = key.replace(".", "_") + source.suffix
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid model asset key: {key!r}")
    return Path("deployment/assets") / name


def _check_hook_outputs(root, produced, files, expected_digests, copied_asset_modes):
    manifest = Path("manifest.yaml")
    written = {manifest, *expected_digests}
    for key, relative in produced.items():
        rel = Path(relative)
        if key in files or rel in written:
            raise ValueError(f"Deployment output collision for {key}: {rel}")
        written.add(rel)

    if os.path.lexists(root / manifest):
        raise ValueError("Deployment output collision with manifest.yaml")
    for rel in written - {manifest}:
        path = root / rel
        if any((root / directory).is_symlink() for directory in reversed(rel.parents)):
            raise ValueError(f"Invalid model deployment file path: {path}")
        try:
            mode = _file_identity(path).mode
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError(f"Missing model deployment file: {path}") from exc
        if rel in copied_asset_modes and mode & ~copied_asset_modes[rel]:
            raise ValueError(f"Export hook broadened asset permissions: {rel}")
    for rel, digest in expected_digests.items():
        if file_digest(root / rel) != digest:
            raise ValueError(f"Model export changed generated file: {rel}")


def write_model_bundle(root, model, config, *, allowed_asset_files=None, mode_ceilings=None,
                       extra_mode_ceilings=None):
    """Export the live model privately until all file permissions are final."""
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(root, 0o700)
    probe = root / ".mode-probe"
    probe.mkdir()
    default_dir_mode = stat.S_IMODE(probe.lstat().st_mode)
    probe.rmdir()

    def output_path(relative):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode_ceilings is not None and relative in mode_ceilings:
            # Bound read access from creation, including while Torch writes the weights.
            path.touch(mode=(0o666 & mode_ceilings[relative]) | 0o200, exist_ok=False)
        return path

    config = clone_config(config)
    config.task = "train"  # Artifact contract records a train-shaped resolved config.
    config.setdefault("train", {})
    config.setdefault("data", {})
    config.get("runtime", {}).pop("prefer_bundled_assets", None)
    config.model.pop("artifact", None)
    config.model.pop("checkpoint", None)
    weights = output_path("deployment/pytorch/full_weights.pt")
    live_export = getattr(model, "export_live_state_dict", None)
    if live_export is not None:
        state = live_export()
    else:
        state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in model.state_dict().items()
        }
    torch.save(model.export_backend_state_dict(state), weights)
    files = {"weights": str(weights.relative_to(root)), "resolved_config": "config_resolved.yaml"}
    if model.save_nexus_weights():
        nexus = output_path("deployment/nexus/full_weights.pt")
        torch.save(state, nexus)
        files["nexus_weights"] = str(nexus.relative_to(root))
    copied_asset_modes = {}
    for key, source in _export_assets(model, config, allowed_asset_files).items():
        relative = str(_bundle_asset_path(key, source))
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _copy_preserving_ceiling(source, destination, ceiling=(mode_ceilings or {}).get(relative))
        copied_asset_modes[Path(relative)] = _file_identity(destination).mode
        files.setdefault("model_assets", {})[key] = relative
    save_config(config, output_path("config_resolved.yaml"))
    generated = [Path(rel) for key, rel in files.items() if key != "model_assets"]
    generated.extend(copied_asset_modes)
    expected_digests = {rel: file_digest(root / rel) for rel in generated}
    produced = _deployment_files(
        root, model.export_deployment_artifacts(source_weights=weights, artifact_dir=root)
    )
    _check_hook_outputs(root, produced, files, expected_digests, copied_asset_modes)
    for key, ceiling in (extra_mode_ceilings or {}).items():
        if key not in produced:
            raise ValueError(f"Export omitted declared deployment file: {key}")
        path = root / produced[key]
        path.chmod(_own_file_mode(path) & ceiling)
    files.update(produced)
    manifest = {
        "schema_version": 2,
        "model": {**model.artifact_metadata(), "target": config.model.target},
        "training": {},
        "components": {"deployment": {"status": "ready", "path": "deployment"}},
        "files": files,
        "weights_sha256": expected_digests[weights.relative_to(root)],
    }
    if "feature_manifest" in files:
        manifest["feature_manifest_sha256"] = file_digest(root / files["feature_manifest"])
    save_config(manifest, output_path("manifest.yaml"))
    if mode_ceilings is not None:
        regular_outputs = [str(weights.relative_to(root)), "config_resolved.yaml", "manifest.yaml",
                           *files.get("model_assets", {}).values()]
        if "nexus_weights" in files:
            regular_outputs.append(files["nexus_weights"])
        for relative in regular_outputs:
            if relative in mode_ceilings:
                file = root / relative
                file.chmod(_own_file_mode(file) & mode_ceilings[relative])
    special = stat.S_IMODE(root.lstat().st_mode) & 0o7000
    os.chmod(root, special | default_dir_mode)
    return root


def _bundle_files(root):
    """Walk without following links, including unlisted assets and directories."""
    if not stat.S_ISDIR(root.lstat().st_mode):
        raise ValueError(f"Expected a real feature bundle directory: {root}")
    for path in root.iterdir():
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            yield from _bundle_files(path)
        elif stat.S_ISREG(mode):
            yield path
        else:
            raise ValueError(f"Symlink or special file in feature bundle: {path}")


def _retained_source(configured, resolved):
    # Inspect a direct directory's *own* manifest before allowing pointer resolution.
    supplied = Path(configured).expanduser()
    direct = supplied if (supplied.parts[-2:] == ("deployment", "features")
                          or (supplied / "manifest.yaml").is_file()) else resolved
    if _enclosing_deployment_manifest(direct) is not None:
        return direct
    manifest, _ = _read_feature_manifest(direct)
    seal = _seal_path(direct)
    if "retained_files_sha256" in manifest or seal.exists() or seal.is_symlink():
        return direct
    # Legacy run-owned bundles were emitted by write_model_bundle. Ordinary
    # training artifacts can have the same pathname and include resume state.
    if (direct.parts[-2:] == ("rltoken", "features") and manifest.get("weights_sha256")
            and manifest.get("training") == {} and manifest.get("components") == {
                "deployment": {"status": "ready", "path": "deployment"}
            } and not (direct / "token_validation.json").exists()):
        return direct
    return None


class _FileIdentity(NamedTuple):
    device: int
    inode: int
    size: int
    mtime_ns: int
    mode: int


def _file_identity(path):
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"Expected a regular feature file: {path}")
    return _FileIdentity(st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, stat.S_IMODE(st.st_mode))


@dataclass(frozen=True)
class _VerifiedFile:
    sha256: str
    # Bytes identify a bundle across copies. Inodes identify a file during linking.
    identity: _FileIdentity = field(compare=False)


def _weight_path(manifest):
    default = ("deployment/pytorch/full_weights.pt" if manifest["schema_version"] == 2
               else "backend/full_weights.pt")
    return manifest.get("files", {}).get("weights", default)


def _verified_file(path, cached):
    identity = _file_identity(path)
    if cached is not None and cached.identity == identity:
        return cached
    entry = _VerifiedFile(file_digest(path), identity)
    if _file_identity(path) != identity:
        raise ValueError(f"Feature file changed during verification: {path}")
    return entry


def _verify_source_weights(root, manifest, verified=None):
    if "weights_sha256" not in manifest:
        return {}
    relative = Path(_weight_path(manifest))
    cached = verified.get(relative) if verified is not None else None
    entry = _verified_file(root / relative, cached)
    if entry.sha256 != manifest["weights_sha256"]:
        raise ValueError("Source feature weights checksum mismatch.")
    return {relative: entry}


def _declared_assets(files):
    assets = files.get("model_assets", {})
    if not isinstance(assets, dict) or not all(isinstance(key, str) for key in assets):
        raise ValueError("Invalid feature model_assets mapping.")
    assets = dict(assets)
    aliases = (("norm_stats", "data.norm_stats_path"), ("tokenizer", "data.tokenizer_path"))
    for alias, key in aliases:
        if alias in files:
            assets.setdefault(key, files[alias])
    return assets


def _model_deployment_files(files):
    standard = {"weights", "resolved_config", "nexus_weights", "trainer_state",
                "norm_stats", "tokenizer", "model_assets"}
    return {k: v for k, v in files.items() if k not in standard and isinstance(v, str)}


def _logical_file_map(manifest):
    files = manifest.get("files", {})
    paths = {
        "weights": Path(_weight_path(manifest)),
        "resolved_config": Path("config_resolved.yaml"),
        "manifest": Path("manifest.yaml"),
    }
    if "nexus_weights" in files:
        paths["nexus_weights"] = Path(files["nexus_weights"])
    paths.update({("asset", key): Path(rel) for key, rel in _declared_assets(files).items()})
    paths.update({("deployment_file", key): Path(rel)
                  for key, rel in _model_deployment_files(files).items()})
    return paths


def _validate_feature_inputs(root, manifest, expected_spec, *, stage_one=False):
    if _feature_spec(manifest) != expected_spec:
        raise ValueError("Run's frozen feature bundle disagrees with this policy.")
    paths = dict(manifest.get("files", {}))
    assets = _declared_assets(paths)
    paths.pop("model_assets", None)
    if stage_one:
        paths.pop("trainer_state", None)
    paths.setdefault("weights", _weight_path(manifest))
    real_root = root.resolve(strict=True)
    # InferencePolicy also opens config_resolved.yaml directly.
    for value in ("config_resolved.yaml", *paths.values(), *assets.values()):
        if not isinstance(value, str):
            raise ValueError(f"Invalid feature file path: {value!r}")
        relative = Path(value)
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid feature file path: {value!r}")
        current = root
        for part in relative.parts[:-1]:
            current = current / part
            if not stat.S_ISDIR(current.lstat().st_mode):
                raise ValueError(f"Invalid directory in feature file path: {current}")
        try:
            _file_identity(root / relative)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"Invalid feature file path: {value!r}") from exc
    return {key: real_root / relative for key, relative in assets.items()}


def _source_deployment_fingerprint(root, manifest, cached_digests):
    """Fingerprint the inputs consumed by provider inference independently of resume state."""
    files = manifest.get("files", {})

    def digest(relative):
        relative = Path(relative)
        entry = _verified_file(root / relative, cached_digests.get(relative))
        cached_digests[relative] = entry
        return entry.sha256

    inputs = {
        "schema_version": manifest["schema_version"],
        "model": manifest["model"],
        "resolved_config": digest("config_resolved.yaml"),
        "weights": digest(_weight_path(manifest)),
        "nexus_weights": digest(files["nexus_weights"]) if "nexus_weights" in files else None,
        "assets": {key: digest(value) for key, value in _declared_assets(files).items()},
    }
    extras = _model_deployment_files(files)
    if extras:
        inputs["deployment_files"] = {key: digest(path) for key, path in extras.items()}
    return content_digest(inputs)


def _source_export_restrictions(root, manifest, spec):
    allowed = _validate_feature_inputs(root, manifest, spec, stage_one=True)
    weights = _weight_path(manifest)
    nexus = manifest.get("files", {}).get("nexus_weights", weights)
    ceilings = {
        name: _file_identity(root / relative).mode
        for name, relative in (
            ("deployment/pytorch/full_weights.pt", weights),
            ("deployment/nexus/full_weights.pt", nexus),
            ("config_resolved.yaml", "config_resolved.yaml"),
            ("manifest.yaml", "manifest.yaml"),
        )
    }
    for key, source in allowed.items():
        mode = _file_identity(source).mode
        ceilings[str(_bundle_asset_path(key, source))] = mode
    extra_ceilings = {
        key: _file_identity(root / rel).mode
        for key, rel in _model_deployment_files(manifest.get("files", {})).items()
    }
    return allowed, ceilings, extra_ceilings


def _verify_retained_feature_bundle(root, expected_spec, require_sealed: bool = False,
                                    *, manifest_digest=None, cached=None):
    """Verify every retained file, including files outside the deployment input index."""
    root = Path(root)
    files = {path.relative_to(root) for path in _bundle_files(root)}
    redirects = {Path(name) for name in ("latest.txt", "latest_deployment.txt")}
    if files & (redirects | {Path("checkpoints") / p for p in redirects}):
        raise ValueError(f"Retained feature bundle redirects to a different artifact: {root}")
    manifest, digest = _read_feature_manifest(root)
    if manifest_digest is not None and digest != manifest_digest:
        raise ValueError("Retained feature manifest checksum mismatch.")
    if not {"weights", "resolved_config"} <= manifest["files"].keys():
        raise ValueError("Incomplete retained feature file index.")
    _validate_feature_inputs(root, manifest, expected_spec)
    cached = cached or {}
    verified = {p: _verified_file(root / p, cached.get(p)) for p in sorted(files)}
    if ("weights_sha256" in manifest and
            manifest["weights_sha256"] != verified[Path(_weight_path(manifest))].sha256):
        raise ValueError("Retained feature weights checksum mismatch.")
    if "retained_files_sha256" in manifest:
        if manifest["retained_files_sha256"] != _asset_digests(verified):
            raise ValueError("Retained feature file checksum mismatch.")
    if verified[Path("manifest.yaml")].sha256 != digest:
        raise ValueError("Retained feature manifest changed during verification.")
    if require_sealed and any(f.identity.mode & 0o222 for f in verified.values()):
        raise ValueError("Retained feature files must be sealed before hardlinking.")
    return verified


def _asset_digests(verified):
    return {p.as_posix(): f.sha256 for p, f in verified.items() if p != Path("manifest.yaml")}


def _link_feature_bundle(source, destination, verified):
    destination.mkdir(parents=True, exist_ok=False)
    for relative, expected in verified.items():
        src, dst = source / relative, destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        identity = _file_identity(src)
        if identity.mode & 0o222 or identity != expected.identity:
            raise ValueError(f"Retained feature file is unsealed or changed before hardlink: {src}")
        try:
            os.link(src, dst, follow_symlinks=False)
        except OSError as exc:
            if exc.errno not in {errno.EXDEV, errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise
            _copy_preserving_ceiling(src, dst, strip_write=True)
            if file_digest(dst) != expected.sha256:
                raise ValueError(f"Retained feature file changed during copy: {src}") from exc
        else:
            if _file_identity(dst) != expected.identity:
                raise ValueError(f"Retained feature file changed during hardlink: {src}")


def _write_manifest_seal(destination, digests, private):
    from .checkpoint import _sync_dir

    temporary = private / "manifest.sha256"
    with temporary.open("x", encoding="ascii") as handle:
        handle.write("\n".join(digests) + "\n")
        handle.flush()
        os.fchmod(handle.fileno(), 0o400)
        os.fsync(handle.fileno())
    os.replace(temporary, _seal_path(destination))
    _sync_dir(destination.parent)


def _sync_bundle_dirs(root):
    from .checkpoint import _sync_dir

    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        _sync_dir(directory)
    _sync_dir(root)


def _seal_staging(staging, verified, *, source_fingerprint=None):
    from .checkpoint import _fsync

    path = staging / "manifest.yaml"
    mode = _own_file_mode(path, expected_digest=verified[Path("manifest.yaml")].sha256)
    manifest, _ = _read_feature_manifest(staging)
    if "retained_files_sha256" not in manifest:
        path.chmod(mode | 0o200)
        if source_fingerprint is not None:
            manifest["source_deployment_sha256"] = source_fingerprint
        save_config({**manifest, "retained_files_sha256": _asset_digests(verified)}, path)
        path.chmod(mode & ~0o222)
    for relative, entry in verified.items():
        path = staging / relative
        path.chmod(_own_file_mode(path, expected_digest=entry.sha256) & ~0o222)
        _fsync(path)
    _sync_bundle_dirs(staging)
    return file_digest(staging / "manifest.yaml")


def _copy_verified_files(source, staging, verified):
    for relative, entry in verified.items():
        dst = staging / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        _copy_preserving_ceiling(source / relative, dst, strip_write=True)
        if file_digest(dst) != entry.sha256:
            raise ValueError(f"Retained feature file checksum changed during copy: {relative}")


def _migrate_retained_bundle(destination, verified, private):
    from .checkpoint import _sync_dir

    staging = private / "bundle"
    _copy_verified_files(destination, staging, verified)
    old_digest = verified[Path("manifest.yaml")].sha256
    new_digest = _seal_staging(staging, verified)
    seal = _manifest_seal(destination)
    if len(seal) == 2 and seal != (new_digest, old_digest):
        raise ValueError("Retained feature files changed during interrupted migration.")
    # Every replacement preserves the old bytes until the new index is published.
    # Replacing inodes also leaves any older hardlinked deployment's mode intact.
    manifest = Path("manifest.yaml")
    for relative in verified:
        if relative != manifest:
            os.replace(staging / relative, destination / relative)
    _sync_bundle_dirs(destination)
    _write_manifest_seal(destination, (new_digest, old_digest), private)
    os.replace(staging / manifest, destination / manifest)
    _sync_dir(destination)
    _write_manifest_seal(destination, (new_digest,), private)


def _reuse_mode_ceilings(source_root, source_manifest, destination_manifest, retained_files):
    source_paths = _logical_file_map(source_manifest)
    if retained_files is None:
        # A live export can produce separate Nexus weights from a backend-only source.
        source_paths.setdefault("nexus_weights", source_paths["weights"])
    ceilings = {}
    for key, destination in _logical_file_map(destination_manifest).items():
        if key not in source_paths:
            raise ValueError(f"Retained file has no configured feature source: {key!r}")
        mode = _file_identity(source_root / source_paths[key]).mode
        ceilings[destination] = ceilings.get(destination, mode) & mode
    if retained_files is not None:
        for relative, entry in retained_files.items():
            ceilings.setdefault(relative, entry.identity.mode)
    return ceilings


def _check_retained_modes(modes, ceilings, *, project_sealed=False):
    for relative, mode in modes:
        ceiling = ceilings.get(relative)
        if project_sealed:
            mode &= ~0o222
        if ceiling is None or mode & ~(ceiling & ~0o222):
            raise ValueError(
                "Existing retained feature bundle has broader permissions than source."
            )


def _check_bundle_against_source(source_root, source_manifest, manifest, verified,
                                 *, retained_files=None, project_sealed=False):
    ceilings = _reuse_mode_ceilings(source_root, source_manifest, manifest, retained_files)
    modes = ((p, f.identity.mode) for p, f in verified.items())
    _check_retained_modes(modes, ceilings, project_sealed=project_sealed)
    return ceilings


def _check_retained_copy(source, source_manifest, verified, staging):
    # Copy-time checksums cannot detect a chmod, including on an unlisted file.
    current = {p.relative_to(source): _file_identity(p) for p in _bundle_files(source)}
    if current != {rel: entry.identity for rel, entry in verified.items()}:
        raise ValueError("Retained feature source changed during copy.")
    staged_manifest, _ = _read_feature_manifest(staging)
    ceilings = _reuse_mode_ceilings(source, source_manifest, staged_manifest, verified)
    modes = {p.relative_to(staging): _file_identity(p).mode for p in _bundle_files(staging)}
    if modes.keys() != verified.keys():
        raise ValueError("Retained feature file set changed during copy.")
    _check_retained_modes(modes.items(), ceilings)


def retain_feature_bundle(source, destination, *, expected_manifest_sha256=None, source_sha256=None):
    """Verify provider inputs and publish a sealed run-owned bundle."""
    from .checkpoint import _publication_lock, _sync_dir

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _publication_lock(destination.parent):
        source_root = resolve_artifact(source)
        retained, source_manifest, source_digest = _source_manifest(source, source_root)
        if source_sha256 is not None and source_digest != source_sha256:
            raise ValueError("Selected feature source manifest checksum mismatch.")
        expected = _feature_spec(source_manifest)
        allowed, ceilings, extra_ceilings = _source_export_restrictions(
            source_root, source_manifest, expected
        )
        same_root = source_root == destination.resolve()
        source_verified = {}
        if retained is None:
            source_verified = _verify_source_weights(source_root, source_manifest)
            _load_artifact_config(source_root / "config_resolved.yaml")
        elif not same_root:
            source_verified = _verify_retained_feature_bundle(
                retained, expected, manifest_digest=source_digest
            )
        if not same_root:
            source_fp = _source_deployment_fingerprint(
                source_root, source_manifest, source_verified
            )

        if destination.exists() or destination.is_symlink():
            manifest, digest = _read_feature_manifest(destination, run_owned=True)
            if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
                raise ValueError(
                    "Existing retained feature bundle does not match checkpoint feature manifest."
                )
            verified = _verify_retained_feature_bundle(
                destination, expected, manifest_digest=digest
            )
            indexed = "retained_files_sha256"
            sealed = indexed in manifest and all(not f.identity.mode & 0o222
                                                  for f in verified.values())
            if not same_root:
                if retained is None:
                    matches = manifest.get("source_deployment_sha256") == source_fp
                else:
                    destination_fp = _source_deployment_fingerprint(destination, manifest, verified)
                    if indexed in manifest and indexed in source_manifest:
                        same_files = source_verified == verified  # Includes the manifest pin.
                    else:
                        same_files = _asset_digests(source_verified) == _asset_digests(verified)
                    matches = source_fp == destination_fp and same_files
                if not matches:
                    raise ValueError(
                        "Existing retained feature bundle disagrees with configured feature source."
                    )
                reuse_ceilings = _check_bundle_against_source(
                    source_root, source_manifest, manifest, verified,
                    retained_files=source_verified if retained is not None else None,
                    project_sealed=not sealed,
                )
            if retained is not None:
                _load_artifact_config(source_root / "config_resolved.yaml")
            if sealed:
                seal = _manifest_seal(destination)
                if seal == (digest,):
                    return destination
                if not seal or seal[0] != digest:
                    raise ValueError("Retained feature files changed during interrupted migration.")
            with tempfile.TemporaryDirectory(dir=destination.parent) as private:
                if sealed:
                    # Resume after manifest replacement without copying the weights again.
                    _sync_dir(destination)
                    _write_manifest_seal(destination, (digest,), Path(private))
                else:
                    _migrate_retained_bundle(destination, verified, Path(private))
                    if not same_root:
                        modes = ((p, _file_identity(destination / p).mode) for p in verified)
                        _check_retained_modes(modes, reuse_ceilings)
            return destination

        provider, config, _ = load_provider(
            source_root, _verified_files=(source_root, source_verified),
            expected_manifest_sha256=source_digest,
        )
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".features-") as private:
            staging = Path(private) / "bundle"
            if retained is not None:
                _copy_verified_files(source_root, staging, source_verified)
                digest = _seal_staging(staging, source_verified)
                _check_retained_copy(source_root, source_manifest, source_verified, staging)
            else:
                write_model_bundle(
                    staging, provider, config, allowed_asset_files=allowed, mode_ceilings=ceilings,
                    extra_mode_ceilings=extra_ceilings,
                )
                staged_manifest, staged_digest = _read_feature_manifest(staging)
                verified = _verify_retained_feature_bundle(
                    staging, expected, manifest_digest=staged_digest
                )
                current, _ = _read_feature_manifest(source_root)
                _validate_feature_inputs(source_root, current, expected, stage_one=True)
                current_fp = _source_deployment_fingerprint(source_root, current, source_verified)
                _verify_source_weights(source_root, current, source_verified)
                if current_fp != source_fp:
                    raise ValueError("Configured deployment inputs changed during retention.")
                _check_bundle_against_source(
                    source_root, current, staged_manifest, verified, project_sealed=True
                )
                digest = _seal_staging(staging, verified, source_fingerprint=source_fp)
            if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
                raise ValueError(
                    "Retained feature bundle does not match checkpoint feature manifest."
                )
            # Install the seal before making a new directory visible.
            _write_manifest_seal(destination, (digest,), Path(private))
            staging.rename(destination)
            _sync_dir(destination.parent)
        return destination


def restore_policy_config(config):
    if not config.model.get("artifact"):
        return config
    return _restore_deployment_config(config)[0]


def _restore_deployment_config(config):
    root = resolve_artifact(config.model.artifact)
    manifest = load_manifest(root)
    trained = _load_artifact_config(root / "config_resolved.yaml")
    files = manifest["files"]
    relative = Path(files.get("feature_manifest", ""))
    if relative.name != "manifest.yaml" or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("RLToken Artifact requires a relative feature_manifest file.")
    child_manifest = root / relative
    for part in (child_manifest, *child_manifest.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Symlink in bundled feature manifest path: {part}")
    if not child_manifest.is_file():
        raise FileNotFoundError(f"Missing bundled feature manifest: {child_manifest}")
    child, digest = _read_feature_manifest(child_manifest.parent, enclosing=manifest)
    if manifest.get("weights_sha256") != file_digest(artifact_weight_path(root)):
        raise ValueError("Actor deployment weight checksum mismatch.")
    restored = clone_config(config)
    restored.model["rltoken"] = trained.model.rltoken
    if isinstance(trained.get("algorithm"), dict):
        restored["algorithm"] = trained.algorithm
    restored.model["features"] = {"artifact": str(child_manifest.parent)}
    restored.model.checkpoint = str(artifact_weight_path(root))
    spec = _feature_spec(child)
    if spec.to_dict() != RLFeatureSpec.from_dict(manifest["model"]["rl_features"]).to_dict():
        raise ValueError("Inner and outer feature specifications disagree.")
    return clone_config(restored), digest
