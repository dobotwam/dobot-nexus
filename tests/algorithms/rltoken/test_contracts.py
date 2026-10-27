import unittest

import torch

from dobot_nexus.core.config import Config
from dobot_nexus.core.contracts import EncodedActionBatch, SamplingOptions
from tests.fakes import DummyVLAModule

from .fakes import feature_spec, features


class ContractsTest(unittest.TestCase):
    def test_batched_feature_contexts_split_without_aliasing(self):
        spec = feature_spec()
        packet = features(spec, batch=2)
        with self.assertRaisesRegex(ValueError, "action_contexts"):
            packet.row(0)
        packet.action_contexts = [{"raw_state": [[1, 2]]}, {"raw_state": [[3, 4]]}]
        packet.validate(spec)
        row = packet.row(1)
        packet.action_contexts[1]["raw_state"][0][0] = 99
        packet.latent.zero_()
        self.assertEqual(row.action_context["raw_state"], [[3, 4]])
        self.assertTrue(row.latent.eq(1).all())
        packet.action_contexts = [{}]
        with self.assertRaisesRegex(ValueError, "one dictionary"):
            packet.validate(spec)
        with self.assertRaisesRegex(ValueError, "index"):
            row.row(1)

    def test_optional_capabilities_preserve_existing_models(self):
        model = DummyVLAModule(Config({"model": Config({})}), mode="train")
        with self.assertRaisesRegex(NotImplementedError, "DummyVLAModule.*RL feature"):
            model.extract_rl_features(None, SamplingOptions())
        with self.assertRaisesRegex(NotImplementedError, "DummyAdapter.*RL action"):
            model.adapter.encode_executed_actions(None)

    def test_feature_storage_is_independent(self):
        spec = feature_spec()
        packet = features(spec).validate(spec, chunk_size=2)
        packet.action_context = {"raw_state": [[1, 2]]}
        stored = packet.cpu_copy()
        packet.latent.zero_()
        packet.action_context["raw_state"][0][0] = 5
        self.assertTrue(stored.latent.eq(1).all())
        self.assertEqual(stored.action_context["raw_state"][0][0], 1)
        packet.feature_id = "different"
        with self.assertRaisesRegex(ValueError, "feature_id"):
            packet.validate(spec)
        packet = features(spec)
        packet.latent.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "detached"):
            packet.validate(spec)

    def test_action_mask_and_domain_validation(self):
        spec = feature_spec()
        action = EncodedActionBatch(
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
            spec.action_codec,
        )
        action.validate(spec)
        action.actions[0, 0] = 2
        with self.assertRaisesRegex(ValueError, "within"):
            action.validate(spec)
        action.encodable_mask[0, 0] = False
        action.validate(spec)
