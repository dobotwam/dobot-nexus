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
