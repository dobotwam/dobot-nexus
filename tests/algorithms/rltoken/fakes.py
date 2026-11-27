import numpy as np
import torch

from dobot_nexus.core.contracts import RLFeatureBatch, RLFeatureSpec
from tests.fakes import DummyAdapter, DummyVLAModule


def feature_spec(latent=4, state=3, action=2, horizon=3):
    return RLFeatureSpec(
        latent,
        state,
        action,
        horizon,
        "test-features-v1",
        "absolute-affine-v1",
        "test-state-v1",
        "radians",
        0.02,
        (-2.0,) * action,
        (2.0,) * action,
    )


def features(spec=None, batch=1):
    spec = spec or feature_spec()
    return RLFeatureBatch(
        torch.ones(batch, spec.latent_dim),
        torch.ones(batch, spec.proprio_dim),
        torch.zeros(batch, spec.reference_horizon, spec.action_dim),
        torch.ones(batch, spec.reference_horizon, dtype=torch.bool),
        spec.feature_id,
    )


def transition(index=0, spec=None, chunk_size=2, **overrides):
    from dobot_nexus.algorithms.rltoken.transitions import ObservationID, Transition
    from dobot_nexus.core.contracts import EncodedActionBatch

    spec = spec or feature_spec()
    args = dict(
        observation_id=ObservationID("run", "env", "session", "episode", str(index)),
        chunk_id=str(index),
        features=features(spec),
        executed=EncodedActionBatch(
            torch.full((1, chunk_size, spec.action_dim), 0.25),
            torch.ones(1, chunk_size, dtype=torch.bool),
            torch.ones(1, chunk_size, dtype=torch.bool),
            spec.action_codec,
        ),
        rewards=torch.ones(chunk_size),
        takeover=torch.zeros(chunk_size, dtype=torch.bool),
        next_features=features(spec),
        next_observation_id=ObservationID("run", "env", "session", "episode", str(index + 1)),
    )
    args.update(overrides)
    return Transition(**args)


class TinyRLAdapter(DummyAdapter):
    def encode_executed_actions(self, actions, *, aux=None):
        from dobot_nexus.core.contracts import EncodedActionBatch

        actions = torch.as_tensor(actions).float().clone()
        mask = torch.arange(actions.shape[1])[None] < torch.tensor(aux["executed_steps"])[:, None]
        return EncodedActionBatch(actions, mask, actions.abs().le(1).all(-1), "absolute-affine-v1")


class TinyFeatureModel(DummyVLAModule):
    def __init__(self, config, *, mode):
        super().__init__(config, mode=mode)
        self.adapter = TinyRLAdapter()
        self.register_buffer("running", torch.zeros(1))
        self.calls = 0

    def spec_rl(self):
        from dataclasses import replace

        from dobot_nexus.algorithms.rltoken.identity import content_digest

        spec = feature_spec(
            int(self.config.model.get("latent_dim", 4)),
            int(self.config.model.get("proprio_dim", 3)),
            self.config.model.action_dim,
            self.config.model.action_horizon,
        )
        return replace(spec, feature_id=content_digest(self.state_dict()))

    def artifact_metadata(self):
        return {**super().artifact_metadata(), "rl_features": self.spec_rl().to_dict()}

    @torch.no_grad()
    def extract_rl_features(self, observation, options):
        self.calls += 1
        if self.training:
            self.running.add_(1)
        packet = features(self.spec_rl(), observation.batch_size)
        packet.latent *= self.scale.cpu()
        packet.action_context = {"raw_state": observation.state.tolist(), "observation_id": "test"}
        packet.action_contexts = [
            {"raw_state": [state], "observation_id": "test"} for state in observation.state.tolist()
        ]
        return packet


def observation(state_dim=2):
    from dobot_nexus.core.contracts import ObservationBatch

    return ObservationBatch(
        {"base": np.zeros((1, 8, 8, 3), dtype=np.uint8)},
        np.zeros((1, state_dim), dtype=np.float32),
        ["pick"],
    )
