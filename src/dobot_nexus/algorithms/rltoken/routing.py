"""Phase routing shared by collection, frozen caches and deployment.

As in RLinf's RLT route, readiness gates Actor execution, not critical-phase
recording. The environment owns task predicates, switch latching and reset.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch

from dobot_nexus.core.contracts import ObservationBatch


@dataclass(frozen=True)
class PolicySwitch:
    execution_scope: str = "all"
    training_scope: str = "all"
    task_mode: str = "full_task"

    def __post_init__(self):
        if self.execution_scope not in {"all", "critical_phase"}:
            raise ValueError("policy_switch.execution_scope must be all or critical_phase.")
        if self.training_scope not in {"all", "critical_phase"}:
            raise ValueError("policy_switch.training_scope must be all or critical_phase.")
        if self.task_mode not in {"full_task", "critical_phase"}:
            raise ValueError("policy_switch.task_mode must be full_task or critical_phase.")

    @classmethod
    def from_config(cls, config):
        algorithm = config.get("algorithm", {})
        return cls(**algorithm.get("policy_switch", {})) if isinstance(algorithm, dict) else cls()

    def identity(self):
        # Preserve identities of existing ungated runs/caches.
        return {} if self == PolicySwitch() else {"policy_switch": asdict(self)}

    def select(self, observation):
        observation = ObservationBatch.from_value(observation)
        phase = observation.control.get("rlt_switch_flags")
        ones = np.ones(observation.batch_size, dtype=np.bool_)
        if self.task_mode == "critical_phase":
            phase = ones  # The adapter resets directly into the critical task.
        if phase is None and "critical_phase" in {self.execution_scope, self.training_scope}:
            raise ValueError("Full-task phase routing requires observation.control.rlt_switch_flags "
                             "(bool [B]) on reset and each post-action observation; the environment "
                             "adapter owns the trigger, latch and episode reset.")
        execute = phase if self.execution_scope == "critical_phase" else ones
        record = phase if self.training_scope == "critical_phase" else ones
        return phase, execute, record


def routed_actions(actor, latent, state, reference, enabled, chunk_size, **kwargs):
    """Run only selected rows; inactive environments do not consume Actor noise."""
    mask = torch.as_tensor(enabled, dtype=torch.bool, device=reference.device)
    actions = reference[:, :chunk_size].clone()
    if mask.any():
        actions[mask] = actor(latent[mask], state[mask], reference[mask], **kwargs)
    return actions
