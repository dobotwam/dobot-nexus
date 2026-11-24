from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

import torch

from dobot_nexus.core.contracts import EncodedActionBatch, RLFeatureBatch, RLFeatureSpec


@dataclass(frozen=True)
class ObservationID:
    run_id: str
    env_id: str
    session_epoch: str
    episode_id: str
    observation_id: str

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in vars(self).values()):
            raise ValueError("Observation identity fields must be nonempty strings.")

    @property
    def session(self):
        return (self.run_id, self.env_id, self.session_epoch, self.episode_id)


@dataclass
class Transition:
    observation_id: ObservationID
    chunk_id: str
    features: RLFeatureBatch
    executed: EncodedActionBatch
    rewards: torch.Tensor | None
    takeover: torch.Tensor
    next_features: RLFeatureBatch | None = None
    next_observation_id: ObservationID | None = None
    terminated: bool = False
    truncated: bool = False
    bootstrap_truncation: bool = False
    execution_confirmed: bool = True
    reward_finalized: bool = True
    accepted: bool | None = True
    source: str = "reference"
    actor_version: tuple[str, str, int] | None = None
    candidate_commands: object = None
    executed_commands: object = None
    command_id: str | None = None
    # A macro-action is selected before execution. On a true terminal prefix,
    # its unexecuted suffix cannot affect the return; it is not execution evidence.
    terminal_candidate_actions: torch.Tensor | None = field(
        default=None, metadata={"digest_omit_none": True}
    )

    episode_success: bool | None = field(default=None, metadata={"digest_omit_none": True})
    origin: str | None = field(default=None, metadata={"digest_omit_none": True})
    supervision: torch.Tensor | None = field(default=None, metadata={"digest_omit_none": True})
    supervision_mask: torch.Tensor | None = field(default=None, metadata={"digest_omit_none": True})
    mc_return: float | None = field(default=None, metadata={"digest_omit_none": True})
    mc_source: str | None = field(default=None, metadata={"digest_omit_none": True})
    critical_phase: bool | None = field(default=None, metadata={"digest_omit_none": True})
    record_transition: bool | None = field(default=None, metadata={"digest_omit_none": True})
    next_actor_enabled: bool | None = field(default=None, metadata={"digest_omit_none": True})

    @property
    def key(self):
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be a nonempty string.")
        return (*self.observation_id.session, self.chunk_id)

    def cpu_copy(self):
        result = copy.deepcopy(self)
        result.features = self.features.cpu_copy()
        result.next_features = (
            self.next_features.cpu_copy() if self.next_features is not None else None
        )
        result.executed = EncodedActionBatch(
            self.executed.actions.detach().cpu().float().contiguous().clone(),
            self.executed.executed_mask.detach().cpu().contiguous().clone(),
            self.executed.encodable_mask.detach().cpu().contiguous().clone(),
            self.executed.action_codec,
        )
        for name in ("supervision", "supervision_mask"):
            value = getattr(self, name)
            if value is not None:
                setattr(result, name, value.detach().cpu().contiguous().clone())
        result.takeover = self.takeover.detach().cpu().contiguous().clone()
        if self.terminal_candidate_actions is not None:
            result.terminal_candidate_actions = (
                self.terminal_candidate_actions.detach().cpu().float().contiguous().clone()
            )
        result.rewards = (
            self.rewards.detach().cpu().float().contiguous().clone()
            if self.rewards is not None
            else None
        )
        return result

    def admission(self, spec: RLFeatureSpec, chunk_size: int, *, allow_terminal_chunks=False,
                  training_scope="all"):
        for value in (self.critical_phase, self.record_transition, self.next_actor_enabled):
            if value is not None and type(value) is not bool:
                return "rejected", "invalid_phase_control"
        if not self.execution_confirmed:
            return "rejected", "unconfirmed_execution"
        if training_scope == "critical_phase" and self.critical_phase is None:
            return "rejected", "missing_critical_phase"
        if self.record_transition is False or (training_scope == "critical_phase" and not self.critical_phase):
            return "rejected", "outside_training_scope"
        if not self.reward_finalized or self.accepted is None:
            return "pending", "awaiting_reward_or_gate"
        if self.accepted is not True:
            return "rejected", "record_gate"
        try:
            if self.episode_success is not None and type(self.episode_success) is not bool:
                return "rejected", "invalid_episode_outcome"
            if self.origin not in {None, "demo", "vla", "online"}:
                return "rejected", "invalid_data_origin"
            if (self.supervision is None) != (self.supervision_mask is None):
                return "rejected", "incomplete_supervision"
            if self.supervision is not None:
                if (self.supervision.shape != (1, chunk_size, spec.action_dim)
                        or self.supervision_mask.shape != (1, chunk_size)
                        or self.supervision_mask.dtype != torch.bool
                        or not torch.isfinite(self.supervision[self.supervision_mask]).all()
                        or (self.supervision[self.supervision_mask].abs() > 1.000001).any()
                        or (self.supervision_mask & ~self.executed.executed_mask).any()):
                    return "rejected", "invalid_supervision"
            if self.mc_return is not None:
                import math
                if not math.isfinite(self.mc_return) or self.mc_source not in {"vla", "demo", "online"}:
                    return "rejected", "invalid_mc_label"
            self.features.validate(spec, chunk_size=chunk_size)
            self.executed.validate(spec)
            if self.features.latent.shape[0] != 1 or self.executed.actions.shape != (
                1,
                chunk_size,
                spec.action_dim,
            ):
                return "rejected", "fixed_chunk_shape"
            mask = self.executed.executed_mask
            short = not bool(mask.all())
            if short:
                if not allow_terminal_chunks or not self.terminated:
                    return "rejected", "short_chunk"
                count = int(mask.sum())
                if count == 0 or not torch.equal(
                    mask, (torch.arange(chunk_size, device=mask.device) < count)[None]
                ):
                    return "rejected", "invalid_terminal_prefix"
                candidate = self.terminal_candidate_actions
                if (not isinstance(candidate, torch.Tensor)
                        or candidate.shape != self.executed.actions.shape
                        or not candidate.is_floating_point()
                        or not torch.isfinite(candidate).all()
                        or candidate.requires_grad or (candidate.abs() > 1.000001).any()):
                    return "rejected", "missing_or_invalid_terminal_candidate"
            if not self.executed.encodable_mask[mask].all():
                return "rejected", "unencodable_command"
            if (
                self.rewards is None
                or self.rewards.shape != (chunk_size,)
                or not torch.isfinite(self.rewards).all()
            ):
                return "rejected", "missing_or_invalid_reward"
            if self.takeover.shape != (chunk_size,) or self.takeover.dtype != torch.bool:
                return "rejected", "takeover_must_describe_executed_steps"
            if short and (self.rewards[~mask[0]].ne(0).any() or self.takeover[~mask[0]].any()):
                return "rejected", "terminal_padding_has_feedback"
            if self.source not in {"actor", "reference", "human"}:
                return "rejected", "unknown_action_source"
            if (self.source == "actor") != (self.actor_version is not None):
                return "rejected", "invalid_actor_provenance"
            if not self.terminated:
                if self.truncated and not self.bootstrap_truncation:
                    return "rejected", "undeclared_truncation_semantics"
                if self.next_features is None or self.next_observation_id is None:
                    return "rejected", "missing_final_observation"
                if self.next_observation_id.session != self.observation_id.session:
                    return "rejected", "reset_observation"
                if self.next_observation_id == self.observation_id:
                    return "rejected", "unchanged_observation_id"
                self.next_features.validate(spec, chunk_size=chunk_size)
                if self.next_features.latent.shape[0] != 1:
                    return "rejected", "next_batch_size"
        except (ValueError, TypeError) as exc:
            return "rejected", str(exc)
        return "admitted", "terminal_prefix" if short else "complete_fixed_chunk"

    def finalize(self, *, rewards, accepted=True):
        if self.reward_finalized and self.accepted is not None:
            raise ValueError("Only pending transitions can be finalized.")
        return replace(self, rewards=rewards, reward_finalized=True, accepted=accepted)


def training_batch(rows, chunk_size, device):
    def cat(name, items):
        return torch.cat([getattr(row, name) for row in items]).to(device)

    current = [row.features for row in rows]
    reference = cat("reference_actions", current)[:, :chunk_size]
    executed = torch.cat([row.executed.actions for row in rows]).to(device)
    critic_actions = torch.cat([
        torch.where(row.executed.executed_mask[..., None], row.executed.actions,
                    row.terminal_candidate_actions)
        if row.terminated and not row.executed.executed_mask.all()
        else row.executed.actions
        for row in rows
    ]).to(device)
    takeover = torch.stack([row.takeover for row in rows]).to(device)
    # Clone through torch.where; neither raw reference nor next reference is modified.
    bc_target = torch.where(takeover[..., None], executed, reference)
    bootstrap = torch.tensor([not row.terminated for row in rows], device=device, dtype=torch.bool)
    next_packets = [row.next_features for row in rows if not row.terminated]
    return {
        "latent": cat("latent", current),
        "state": cat("state", current),
        "executed": executed,
        "critic_actions": critic_actions,
        "reference_condition": bc_target.clone(),
        "bc_target": bc_target,
        "rewards": torch.stack([row.rewards for row in rows]).to(device),
        "bootstrap": bootstrap,
        "next_latent": cat("latent", next_packets) if next_packets else None,
        "next_state": cat("state", next_packets) if next_packets else None,
        "next_reference": cat("reference_actions", next_packets) if next_packets else None,
        "next_actor_enabled": torch.tensor([row.next_actor_enabled is not False
                                            for row in rows if not row.terminated],
                                           device=device, dtype=torch.bool),
    }


def residual_batch(rows, chunk_size, device):
    batch = training_batch(rows, chunk_size, device)
    reference = torch.cat([r.features.reference_actions[:, :chunk_size] for r in rows]).to(device)
    batch["reference_condition"] = reference
    batch["valid_mask"] = torch.cat([r.executed.executed_mask for r in rows]).to(device)
    batch["episode_success"] = torch.tensor(
        [bool(r.episode_success) for r in rows], device=device, dtype=torch.bool
    )
    batch["episode_success_known"] = torch.tensor(
        [r.episode_success is not None for r in rows], device=device, dtype=torch.bool
    )
    batch["terminated"] = ~batch["bootstrap"]
    batch["truncated"] = torch.tensor([bool(r.truncated) for r in rows], device=device)
    batch["intervene_flags"] = torch.stack([r.takeover for r in rows]).to(device)
    return batch


@dataclass(frozen=True)
class Finalization:
    key: tuple[str, ...]
    rewards: torch.Tensor
    accepted: bool
    episode_success: bool | None = None
