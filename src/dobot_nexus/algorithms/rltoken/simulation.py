"""Vector collection protocol; simulator SDKs remain in the external environment adapter."""

from __future__ import annotations

import numpy as np

from dobot_nexus.core.contracts import ObservationBatch

from .batching import observation_row, stack_observations
from .env import EnvSession, ExecutionFeedback


class SimulationSession:
    """One vector simulator call per chunk, with independent episode and replay identities."""

    def __init__(self, agent, *, batch_size, env_id="sim", **session_options):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Simulation batch_size must be positive.")
        for method in ("reset", "execute_batch", "query_batch", "close"):
            if not callable(getattr(agent, method, None)):
                raise TypeError(f"Vector simulator adapter must provide {method}().")
        self.agent = agent
        self.sessions = [
            EnvSession(agent, env_id=f"{env_id}-{index}", **session_options)
            for index in range(batch_size)
        ]
        self.timeout = self.sessions[0].timeout
        self.closed = self.faulted = False

    def collect_chunk(self, rollout, commit, *, actor_ready=False):
        if self.closed or self.faulted:
            raise RuntimeError("Simulation session is closed or requires execution reconciliation.")
        try:
            return self._collect(rollout, commit, actor_ready=actor_ready)
        except Exception:
            # An executed vector step must never be repeated after a transport/validation failure.
            self.faulted = True
            raise

    def _collect(self, rollout, commit, *, actor_ready):
        reset_indices = [
            index for index, session in enumerate(self.sessions) if session.identity is None
        ]
        if reset_indices:
            reset = ObservationBatch.from_value(
                self.agent.reset(env_indices=reset_indices, timeout=self.timeout)
            )
            expected_ids = [self.sessions[index].env_id for index in reset_indices]
            if reset.batch_size != len(reset_indices) or reset.env_ids != expected_ids:
                raise ValueError("Vector reset must return the requested env_ids in order.")
            for row, index in enumerate(reset_indices):
                self.sessions[index].reset_to(observation_row(reset, row))
        identities = [session.identity for session in self.sessions]
        observation = stack_observations(
            [session.observation for session in self.sessions],
            [session.env_id for session in self.sessions],
        )
        decisions = rollout(
            {
                "operation": "decide_batch",
                "identities": identities,
                "observation": observation,
                "actor_ready": actor_ready,
            }
        )
        if len(decisions) != len(self.sessions):
            raise ValueError("Rollout returned the wrong number of decisions.")
        commands = np.concatenate([decision.commands for decision in decisions])
        chunk_size, action_dim = self.sessions[0].chunk_size, self.sessions[0].action_dim
        if (
            commands.shape != (len(self.sessions), chunk_size, action_dim)
            or not np.isfinite(commands).all()
        ):
            raise ValueError("Simulator commands must have finite shape [B,C,A].")
        command_ids = [session.command_id for session in self.sessions]
        unknown = False
        try:
            feedbacks = self.agent.execute_batch(command_ids, commands.copy(), timeout=self.timeout)
        except TimeoutError:
            try:
                feedbacks = self.agent.query_batch(command_ids, timeout=self.timeout)
            except Exception:
                unknown = True
                feedbacks = [
                    ExecutionFeedback(
                        command_id,
                        np.zeros_like(decision.commands),
                        0,
                        None,
                        np.zeros(chunk_size, dtype=np.bool_),
                        None,
                        confirmed=False,
                    )
                    for command_id, decision in zip(command_ids, decisions, strict=True)
                ]
        if not isinstance(feedbacks, (list, tuple)) or len(feedbacks) != len(self.sessions):
            raise ValueError("Simulator feedback must contain one row per environment.")
        next_rows = []
        for index, (session, feedback) in enumerate(zip(self.sessions, feedbacks, strict=True)):
            session.validate_feedback(feedback)
            identity = session.next_identity(feedback)
            if identity is not None:
                next_rows.append((index, identity, feedback.final_observation))
        encoded = rollout(
            {
                "operation": "encode_batch",
                "rows": [
                    (feedback.commands, decision.features, feedback.executed_steps)
                    for feedback, decision in zip(feedbacks, decisions, strict=True)
                ],
            }
        )
        if len(encoded) != len(self.sessions):
            raise ValueError("Rollout returned the wrong number of encoded actions.")
        next_packets = {}
        if next_rows:
            packets = rollout(
                {
                    "operation": "features_batch",
                    "identities": [item[1] for item in next_rows],
                    "observation": stack_observations(
                        [item[2] for item in next_rows], [item[1].env_id for item in next_rows]
                    ),
                }
            )
            if len(packets) != len(next_rows):
                raise ValueError("Rollout returned the wrong number of next observations.")
            next_packets = {
                item[0]: packet for item, packet in zip(next_rows, packets, strict=True)
            }
        rows = []
        for index, (session, decision, feedback) in enumerate(
            zip(self.sessions, decisions, feedbacks, strict=True)
        ):

            def prepared(request, index=index):
                if request["operation"] == "encode":
                    return encoded[index]
                if request["operation"] == "features":
                    return next_packets[index]
                raise ValueError("Unexpected per-row rollout operation.")

            rows.append(
                session.build_transition(decision, feedback, prepared, unknown_execution=unknown)
            )
        # One bounded message and one vector ACK; Replay still admits each transition separately.
        acks = commit(rows)
        if len(acks) != len(rows) or any(
            ack.key != row.key for ack, row in zip(acks, rows, strict=True)
        ):
            raise ValueError("Durable ACKs do not match the simulation batch.")
        statuses = [
            session.acknowledge(row, feedback, ack)
            for session, row, feedback, ack in zip(
                self.sessions, rows, feedbacks, acks, strict=True
            )
        ]
        self.faulted = any(status["faulted"] for status in statuses)
        return {
            "chunks": sum(status["chunks"] for status in statuses),
            "env_steps": sum(status["env_steps"] for status in statuses),
            "faulted": self.faulted,
            "admission": [status["admission"] for status in statuses],
            "reason": [status["reason"] for status in statuses],
        }

    def close(self):
        if not self.closed:
            self.closed = True
            self.agent.close()
