"""Frozen-provider episode cache. Readers supply actual timestamp-aligned observations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import torch

from dobot_nexus.core.contracts import ObservationBatch, SamplingOptions, RLFeatureSpec
from .serialization import load_payload, save_payload
from .identity import content_digest
from .transitions import ObservationID, Transition
from .routing import PolicySwitch


CACHE_FORMAT = "rltoken_episode_cache_v2"


@dataclass
class Episode:
    episode_id: str
    observations: list  # T+1 ObservationBatch objects; final entry is never a reset observation.
    commands: torch.Tensor  # [T,A] in physical controller units
    rewards: torch.Tensor  # [T], finalized per-step reward
    takeover: torch.Tensor  # bool [T]
    origin: str  # demo or vla
    terminated: bool
    accepted: bool = True
    success: bool | None = None
    supervision_commands: torch.Tensor | None = None
    supervision_mask: torch.Tensor | None = None
    timestamps: torch.Tensor | None = None
    reference_seeds: dict[int, int] | None = None


def build_cache(provider, episodes, destination, *, chunk_size, dataset_version, reward_definition,
                gamma_step, seed=0, validation_fraction=0.2, stride=None, reference_num_steps=10,
                policy_switch=None):
    switch = PolicySwitch(**(policy_switch or {}))
    if not dataset_version or not reward_definition or not 0 <= validation_fraction < 1:
        raise ValueError("Cache requires data/reward versions and validation_fraction in [0,1).")
    if not 0 <= gamma_step <= 1 or chunk_size <= 0:
        raise ValueError("Invalid cache discount/chunk.")
    if type(reference_num_steps) is not int or reference_num_steps <= 0:
        raise ValueError("Cache reference_num_steps must be a positive integer.")
    stride = chunk_size if stride is None else stride
    if type(stride) is not int or not 0 < stride <= chunk_size:
        raise ValueError("Cache stride must be in [1,chunk_size].")
    provider.eval().requires_grad_(False)
    spec = RLFeatureSpec.from_dict(provider.artifact_metadata()["rl_features"])
    metadata = dict(format=CACHE_FORMAT, feature_spec=spec.to_dict(),
                    dataset_version=dataset_version, reward_definition=reward_definition,
                    chunk_size=chunk_size, stride=stride, gamma_step=gamma_step, seed=seed, reference_num_steps=reference_num_steps,
                    **switch.identity())
    identity = content_digest(metadata)
    all_rows, origins = {}, {}
    for episode in episodes:
        if not episode.accepted:
            continue
        if episode.episode_id in origins:
            raise ValueError("Duplicate offline episode identity.")
        origins[episode.episode_id] = episode.origin
        all_rows[episode.episode_id] = []
        total = len(episode.commands)
        if (episode.origin not in {"demo", "vla"} or total == 0
                or len(episode.observations) != total + 1
                or episode.commands.shape != (total, spec.action_dim)
                or episode.rewards.shape != (total,) or not torch.isfinite(episode.rewards).all()
                or episode.takeover.shape != (total,) or episode.takeover.dtype != torch.bool):
            raise ValueError(f"Invalid complete episode {episode.episode_id}.")
        if episode.timestamps is not None:
            times = torch.as_tensor(episode.timestamps).double()
            if (times.shape != (total + 1,) or not torch.isfinite(times).all()
                    or not torch.allclose(times.diff(), torch.full((total,), spec.control_dt, dtype=torch.float64),
                                          rtol=1e-3, atol=1e-5)):
                raise ValueError("Episode timestamps disagree with the provider control period.")
        if episode.supervision_commands is not None and episode.supervision_commands.shape != episode.commands.shape:
            raise ValueError("Supervision commands must describe the same execution timeline.")
        if episode.supervision_mask is not None and (episode.supervision_mask.shape != (total,)
                                                      or episode.supervision_mask.dtype != torch.bool):
            raise ValueError("Supervision mask must be bool [T].")
        success = episode.success
        # A pure-VLA MC label cannot contain human interventions.
        pure_vla = episode.origin == "vla" and not episode.takeover.any()
        returns = torch.zeros(total)
        running = 0.
        for index in reversed(range(total)):
            running = float(episode.rewards[index]) + gamma_step * running
            returns[index] = running
        packets = {}
        observations, selections = {}, {}
        def observation(index):
            if index not in observations:
                value = ObservationBatch.from_value(episode.observations[index])
                if value.batch_size != 1:
                    raise ValueError("Offline observations must describe one environment per episode.")
                observations[index] = value
            return observations[index]
        def selection(index):
            if index not in selections:
                selections[index] = switch.select(observation(index))
            return selections[index]
        def packet(index):
            if index not in packets:
                sample_seed = (episode.reference_seeds[index] if episode.reference_seeds is not None
                               else int(content_digest((seed, episode.episode_id, index))[:15], 16))
                packets[index] = provider.extract_rl_features(
                    observation(index), SamplingOptions(deterministic=True, seed=sample_seed, num_steps=reference_num_steps)
                ).cpu_copy().validate(spec, chunk_size=chunk_size)
            return packets[index]
        def observation_id(index):
            return ObservationID(identity, "offline", dataset_version, episode.episode_id, str(index))
        for start in range(0, total, stride):
            phase, _, record = selection(start)
            if not record[0]:
                continue
            end = min(start + chunk_size, total)
            count = end - start
            terminal = episode.terminated and end == total
            if count < chunk_size and not terminal:
                continue  # Unknown truncated suffix is never padded into a terminal return.
            current = packet(start)
            commands = torch.zeros(1, chunk_size, spec.action_dim)
            commands[0, :count] = episode.commands[start:end]
            encoded = provider.adapter.encode_executed_actions(commands.numpy(), aux={
                **current.action_context, "executed_steps": [count]})
            reward, takeover = torch.zeros(chunk_size), torch.zeros(chunk_size, dtype=torch.bool)
            reward[:count], takeover[:count] = episode.rewards[start:end], episode.takeover[start:end]
            supervision_mask = (encoded.executed_mask.clone() if episode.origin == "demo" and success
                                else takeover[None].clone())
            if episode.supervision_mask is not None:
                supervision_mask.zero_()
                supervision_mask[0, :count] = episode.supervision_mask[start:end]
            supervision = encoded.actions.clone()
            if episode.supervision_commands is not None:
                targets = commands.clone()
                targets[0, :count] = episode.supervision_commands[start:end]
                supervision = provider.adapter.encode_executed_actions(targets.numpy(), aux={
                    **current.action_context, "executed_steps": [count]}).actions
            row = Transition(observation_id(start), str(start), current, encoded, reward, takeover,
                next_features=None if terminal else packet(end),
                next_observation_id=None if terminal else observation_id(end),
                terminated=terminal, truncated=end == total and not terminal, bootstrap_truncation=True,
                origin=episode.origin, supervision=supervision, episode_success=success,
                supervision_mask=supervision_mask,
                critical_phase=None if phase is None else bool(phase[0]),
                record_transition=bool(record[0]),
                next_actor_enabled=None if terminal else bool(selection(end)[1][0]),
                terminal_candidate_actions=current.reference_actions[:, :chunk_size].clone() if count < chunk_size else None,
                mc_return=float(returns[start]) if episode.terminated else None,
                mc_source="vla" if pure_vla else "demo")
            status, reason = row.admission(spec, chunk_size, allow_terminal_chunks=True,
                                           training_scope=switch.training_scope)
            if status != "admitted":
                raise ValueError(f"Cache transition {row.key} rejected: {reason}")
            all_rows[episode.episode_id].append(row.cpu_copy())
    validation_ids = set()
    for origin in sorted(set(origins.values())):
        ordered = sorted([key for key, value in origins.items() if value == origin],
                         key=lambda value: content_digest((seed, value)))
        if validation_fraction == 0:
            continue
        if len(ordered) < 2:
            raise ValueError("Each offline origin requires at least two episodes for held-out validation.")
        count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_fraction)))
        validation_ids.update(ordered[:count])
    metadata.update(validation_episodes=sorted(validation_ids),
                    training_episodes=sorted(set(origins) - validation_ids))
    identity = content_digest(metadata)
    rows = {"train": [], "validation": []}
    for key, episode_rows in all_rows.items():
        rows["validation" if key in validation_ids else "train"].extend(episode_rows)
    if not rows["train"] or (validation_fraction > 0 and not rows["validation"]):
        raise ValueError("Cache split contains no admissible chunks.")
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=False)
    payload = dict(identity=identity, metadata=metadata, **rows)
    save_payload(payload, path / "cache.pt")
    (path / "manifest.json").write_text(json.dumps({**metadata, "identity": identity,
                                                  "payload_digest": content_digest(payload)}, indent=2))
    return path


def load_cache(path, spec, *, chunk_size, reward_definition, gamma_step=None, reference_num_steps=10,
               policy_switch=None, trusted_checkpoint=False):
    switch = PolicySwitch(**(policy_switch or {}))
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    payload = load_payload(path / "cache.pt", trusted_checkpoint=trusted_checkpoint,
                           trust_setting="data.trust_legacy_cache")
    metadata = payload["metadata"]
    if metadata.get("format") != CACHE_FORMAT:
        raise ValueError(
            f"Unsupported cache format {metadata.get('format')!r}; rebuild as {CACHE_FORMAT}."
        )
    if PolicySwitch(**metadata.get("policy_switch", {})) != switch:
        raise ValueError("Cache policy_switch identity mismatch; rebuild with the training phase controls.")
    if (content_digest(payload) != manifest["payload_digest"]
            or payload["identity"] != content_digest(metadata)
            or content_digest(metadata["feature_spec"]) != content_digest(spec.to_dict())
            or metadata["reference_num_steps"] != reference_num_steps
            or metadata["chunk_size"] != chunk_size or metadata["reward_definition"] != reward_definition
            or (gamma_step is not None and abs(metadata["gamma_step"] - gamma_step) > 1e-12)):
        raise ValueError("Cache content, feature, codec, chunk or reward/discount identity mismatch.")
    return payload
