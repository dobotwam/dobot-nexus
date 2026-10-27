"""Robot/simulator session. The external agent owns command execution and deadlines."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace

import numpy as np
import torch

from dobot_nexus.core.contracts import ObservationBatch

from .transitions import ObservationID, Transition


@dataclass
class ExecutionFeedback:
    command_id: str
    commands: np.ndarray
    executed_steps: int
    rewards: np.ndarray | None
    takeover: np.ndarray
    final_observation: ObservationBatch | None
    terminated: bool = False
    truncated: bool = False
    confirmed: bool = True
    reward_finalized: bool = True
    accepted: bool | None = True


class EnvSession:
    """One in-flight motor command; retries query its status, never replay it."""

    def __init__(
        self,
        agent,
        *,
        run_id,
        env_id,
        session_epoch,
        chunk_size,
        action_dim,
        timeout=30.0,
        bootstrap_truncation=False,
    ):
        if timeout <= 0:
            raise ValueError("Environment timeout must be positive.")
        self.agent = agent
        self.run_id, self.env_id, self.session_epoch = run_id, env_id, session_epoch
        self.chunk_size, self.action_dim = chunk_size, action_dim
        self.timeout, self.bootstrap_truncation = timeout, bootstrap_truncation
        self.observation = self.identity = None
        self.closed = self.faulted = False
        self.chunks = self.executed_steps = 0

    def collect_chunk(self, rollout, commit, *, actor_ready=False):
        if self.closed or self.faulted:
            raise RuntimeError(
                "Environment session is closed or requires execution reconciliation."
            )
        if self.observation is None:
            self.reset_to(self.agent.reset(timeout=self.timeout))
        decision = rollout(
            {
                "operation": "decide",
                "identity": self.identity,
                "observation": self.observation,
                "actor_ready": actor_ready,
            }
        )
        command_id = self.command_id
        unknown_execution = False
        try:
            feedback = self.agent.execute(
                command_id, decision.commands.copy(), timeout=self.timeout
            )
        except TimeoutError:
            try:
                feedback = self.agent.query(command_id, timeout=self.timeout)
            except Exception:
                # Values below are transport placeholders, not a claim that no motion occurred.
                # Preserve the command ID and candidate; admission rejects unconfirmed execution.
                unknown_execution = True
                feedback = ExecutionFeedback(
                    command_id,
                    np.zeros_like(decision.commands),
                    0,
                    None,
                    np.zeros(self.chunk_size, dtype=np.bool_),
                    None,
                    confirmed=False,
                )
        row = self.build_transition(
            decision, feedback, rollout, unknown_execution=unknown_execution
        )
        ack = commit(row)  # durable ACK precedes the next motor command
        return self.acknowledge(row, feedback, ack)

    @property
    def command_id(self):
        return ":".join((*self.identity.session, self.identity.observation_id))

    def reset_to(self, observation):
        observation = ObservationBatch.from_value(observation)
        if observation.batch_size != 1:
            raise ValueError("An environment session requires exactly one observation.")
        if observation.env_ids is not None and observation.env_ids != [self.env_id]:
            raise ValueError("Reset observation belongs to another environment.")
        self.observation = observation
        self.identity = ObservationID(
            self.run_id, self.env_id, self.session_epoch, uuid.uuid4().hex, "0"
        )

    def validate_feedback(self, feedback):
        if not isinstance(feedback, ExecutionFeedback) or feedback.command_id != self.command_id:
            self.faulted = True
            raise ValueError("Robot Agent returned a mismatched execution confirmation.")
        if (
            type(feedback.executed_steps) is not int
            or not 0 <= feedback.executed_steps <= self.chunk_size
        ):
            self.faulted = True
            raise ValueError("Invalid controller-confirmed executed_steps.")
        expected = (1, self.chunk_size, self.action_dim)
        if np.asarray(feedback.commands).shape != expected:
            raise ValueError(f"Confirmed commands must have padded shape {expected}.")
        if feedback.final_observation is not None:
            final = ObservationBatch.from_value(feedback.final_observation)
            if final.batch_size != 1 or (
                final.env_ids is not None and final.env_ids != [self.env_id]
            ):
                raise ValueError("Final observation belongs to another environment.")

    def next_identity(self, feedback):
        if (
            not feedback.terminated
            and feedback.final_observation is not None
            and feedback.confirmed
        ):
            return replace(self.identity, observation_id=str(int(self.identity.observation_id) + 1))
        return None

    def build_transition(self, decision, feedback, rollout, *, unknown_execution=False):
        self.validate_feedback(feedback)
        encoded = rollout(
            {
                "operation": "encode",
                "commands": feedback.commands,
                "features": decision.features,
                "executed_steps": feedback.executed_steps,
            }
        )
        next_packet = None
        next_identity = self.next_identity(feedback)
        if next_identity is not None:
            next_packet = rollout(
                {
                    "operation": "features",
                    "identity": next_identity,
                    "observation": feedback.final_observation,
                }
            )
        return Transition(
            self.identity,
            self.identity.observation_id,
            decision.features,
            encoded,
            torch.as_tensor(feedback.rewards).float() if feedback.rewards is not None else None,
            torch.as_tensor(feedback.takeover),
            next_packet,
            next_identity,
            feedback.terminated,
            feedback.truncated,
            self.bootstrap_truncation,
            feedback.confirmed,
            feedback.reward_finalized,
            feedback.accepted,
            decision.source,
            decision.actor_version,
            decision.commands.copy(),
            None if unknown_execution else np.asarray(feedback.commands).copy(),
            self.command_id,
            terminal_candidate_actions=(
                decision.candidate_actions
                if feedback.terminated and 0 < feedback.executed_steps < self.chunk_size
                else None
            ),
        )

    def acknowledge(self, row, feedback, ack):
        if ack.key != row.key:
            self.faulted = True
            raise ValueError("Durable ACK does not match the executed command.")
        self.chunks += 1
        if feedback.confirmed:
            self.executed_steps += feedback.executed_steps
        if not feedback.confirmed:
            self.faulted = True
        elif feedback.terminated or feedback.truncated:
            self.observation = self.identity = None
        elif row.next_observation_id is None:
            self.faulted = True
        else:
            self.observation, self.identity = feedback.final_observation, row.next_observation_id
        return {
            "chunks": self.chunks,
            "env_steps": self.executed_steps,
            "admission": ack.status,
            "reason": ack.reason,
            "faulted": self.faulted,
        }

    def close(self):
        self.closed = True
        self.agent.close()
