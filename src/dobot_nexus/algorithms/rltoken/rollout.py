from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch

from dobot_nexus.core.contracts import ObservationBatch, SamplingOptions

from .batching import observation_row, stack_observations
from .identity import content_digest
from .policy import sampling_seed
from .routing import PolicySwitch, routed_actions
from .weight_sync import WeightSlots


@dataclass
class Decision:
    features: object
    commands: object
    source: str
    actor_version: tuple[str, str, int] | None
    candidate_actions: torch.Tensor | None = None
    critical_phase: bool | None = None
    record_transition: bool | None = None


class RolloutEngine:
    """Bounded environment caches and one complete Actor version per inference batch."""

    def __init__(self, policy, *, run_id, learner_epoch, seed=0, max_batch_size=1, policy_switch=None):
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer.")
        if policy.mode != "infer":
            raise ValueError("Rollout requires an inference policy with a frozen provider.")
        self.policy = policy.eval().requires_grad_(False)
        self.weights = WeightSlots(
            policy.actor, policy.feature_spec.feature_id, run_id, learner_epoch
        )
        self.max_batch_size = max_batch_size
        self.policy_switch = (PolicySwitch(**policy_switch) if policy_switch is not None else
                              PolicySwitch.from_config(getattr(policy, "config", {})))
        self.environments = {}
        self.seed = seed
        self.generator = torch.Generator(device=next(policy.actor.parameters()).device).manual_seed(
            sampling_seed(seed, "actor")
        )

    def packet(self, identity, observation):
        return self.packets([identity], observation)[0]

    @torch.no_grad()
    def packets(self, identities, observation):
        observation = ObservationBatch.from_value(observation)
        phase, enabled, record = self.policy_switch.select(observation)
        if not 0 < len(identities) == observation.batch_size <= self.max_batch_size:
            raise ValueError("Rollout batch size must match identities and the configured limit.")
        env_ids = [identity.env_id for identity in identities]
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("A rollout batch must contain distinct environment IDs.")
        if len(set(self.environments) | set(env_ids)) > self.max_batch_size:
            raise ValueError("Rollout environment count exceeds the configured limit.")
        if observation.env_ids is not None and observation.env_ids != env_ids:
            raise ValueError("Observation env_ids disagree with rollout identities.")
        packets, missing = [None] * len(identities), []
        for row, identity in enumerate(identities):
            if (identity.run_id, identity.session_epoch) != (
                self.weights.run_id,
                self.weights.learner_epoch,
            ):
                raise ValueError("Observation belongs to another run or session epoch.")
            single = observation_row(observation, row)
            fingerprint = content_digest(
                {"images": single.images, "state": single.state, "prompt": single.prompt,
                 "control": single.control}
            )
            state = self.environments.get(identity.env_id)
            index = int(identity.observation_id)
            if state is None or identity.session != state["session"]:
                if state is not None and identity.session in state["closed"]:
                    raise ValueError("Observation belongs to a closed episode.")
                if index != 0:
                    raise ValueError("A new observation session must start at zero.")
            elif identity in state["cache"]:
                old_hash, packet = state["cache"][identity]
                if old_hash != fingerprint:
                    raise ValueError("Conflicting observation payload for an existing ID.")
                packets[row] = packet.cpu_copy()
                continue
            elif index <= state["last"]:
                raise ValueError("Expired observation ID cannot regenerate a new reference packet.")
            missing.append((row, identity, single, fingerprint))
        if missing:
            observations = stack_observations(
                [item[2] for item in missing], [item[1].env_id for item in missing]
            )
            # Fixed request ordering is reproducible. Cached next observations keep their noise
            # and reference actions even when other environments reset or weights change.
            seed_keys = []
            for _, identity, _, _ in missing:
                current = self.environments.get(identity.env_id)
                episode = (
                    0
                    if current is None
                    else current["episode_index"] + int(current["session"] != identity.session)
                )
                seed_keys.append((identity.env_id, episode, identity.observation_id))
            offset = (
                int(missing[0][1].observation_id)
                if self.max_batch_size == 1
                else int(content_digest(seed_keys)[:16], 16)
            )
            seed = sampling_seed(self.seed + offset, "reference")
            batch = (
                self.policy.provider.extract_rl_features(observations, SamplingOptions(seed=seed))
                .cpu_copy()
                .validate(self.policy.feature_spec, chunk_size=self.policy.head_config.chunk_size)
            )
            if batch.latent.shape[0] != len(missing):
                raise ValueError("Feature provider returned the wrong batch size.")
            prepared = [batch.row(index) for index in range(len(missing))]
            for (row, identity, _, fingerprint), packet in zip(missing, prepared, strict=True):
                packet.action_context["observation_id"] = content_digest(identity)
                packet.action_context["rltoken_routing"] = {
                    "critical_phase": None if phase is None else bool(phase[row]),
                    "actor_enabled": bool(enabled[row]), "record_transition": bool(record[row]),
                }
                state = self.environments.get(identity.env_id)
                if state is None or state["session"] != identity.session:
                    closed = set() if state is None else state["closed"] | {state["session"]}
                    state = {
                        "session": identity.session,
                        "closed": closed,
                        "cache": OrderedDict(),
                        "episode_index": 0 if state is None else state["episode_index"] + 1,
                    }
                    self.environments[identity.env_id] = state
                state["cache"][identity] = fingerprint, packet
                state["last"] = int(identity.observation_id)
                while len(state["cache"]) > 2:
                    state["cache"].popitem(last=False)
                packets[row] = packet.cpu_copy()
        return packets

    @torch.no_grad()
    def decide(self, identity, observation, *, actor_ready=False, deterministic=False):
        return self.decide_batch(
            [identity], observation, actor_ready=actor_ready, deterministic=deterministic
        )[0]

    @torch.no_grad()
    def decide_batch(self, identities, observation, *, actor_ready=False, deterministic=False):
        packets = self.packets(identities, observation)
        latent, state, reference = (
            torch.cat([getattr(packet, name) for packet in packets])
            for name in ("latent", "state", "reference_actions")
        )
        enabled = [actor_ready and packet.action_context["rltoken_routing"]["actor_enabled"]
                   for packet in packets]
        if any(enabled):
            with self.weights.request() as (actor, version):
                device = next(actor.parameters()).device
                actions = (
                    routed_actions(actor,
                        latent.to(device),
                        state.to(device),
                        reference.to(device),
                        enabled, self.policy.head_config.chunk_size,
                        generator=self.generator,
                        deterministic=deterministic,
                    )
                    .detach()
                    .cpu()
                    .clone()
                )
                actor_version = version.as_tuple()
        else:
            # Even reference routing starts only after the initial synchronization barrier.
            with self.weights.request():
                pass
            actions = reference[:, : self.policy.head_config.chunk_size].clone()
            actor_version = None
        decisions = []
        for row, packet in enumerate(packets):
            output = self.policy.adapter.decode_actions(
                actions[row : row + 1], aux=packet.action_context
            )
            decisions.append(
                Decision(
                    packet,
                    output.actions.copy(),
                    "actor" if enabled[row] else "reference",
                    actor_version if enabled[row] else None,
                    actions[row : row + 1].clone(),
                    packet.action_context["rltoken_routing"]["critical_phase"],
                    packet.action_context["rltoken_routing"]["record_transition"],
                )
            )
        return decisions

    def encode(self, commands, packet, executed_steps):
        return self.policy.adapter.encode_executed_actions(
            commands,
            aux={**packet.action_context, "executed_steps": [executed_steps]},
        ).validate(self.policy.feature_spec)

    def process(self, request):
        operation = request["operation"]
        if operation == "decide_batch":
            return self.decide_batch(
                request["identities"],
                request["observation"],
                actor_ready=request["actor_ready"],
                deterministic=request.get("deterministic", False),
            )
        if operation == "features_batch":
            return self.packets(request["identities"], request["observation"])
        if operation == "encode_batch":
            rows = request["rows"]
            if not 0 < len(rows) <= self.max_batch_size:
                raise ValueError("Executed action batch exceeds the configured limit.")
            return [self.encode(*row) for row in rows]
        if operation == "decide":
            return self.decide(
                request["identity"], request["observation"], actor_ready=request["actor_ready"],
                deterministic=request.get("deterministic", False)
            )
        if operation == "features":
            return self.packet(request["identity"], request["observation"])
        if operation == "encode":
            return self.encode(request["commands"], request["features"], request["executed_steps"])
        raise ValueError(f"Unknown Rollout operation: {operation}")
