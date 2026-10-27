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
