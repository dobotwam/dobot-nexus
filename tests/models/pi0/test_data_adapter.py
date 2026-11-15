"""CPU contract tests for the Pi0 data adapter."""

from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

import torch

from dobot_nexus.models.pi0.data_adapter_pi0 import Pi0DataAdapter


@dataclasses.dataclass
class _Observation:
    images: dict
    image_masks: dict
    state: object
    tokenized_prompt: object
    tokenized_prompt_mask: object
    token_ar_mask: object = None
    token_loss_mask: object = None
    pcd_xyz: object = None

    @classmethod
    def from_dict(cls, value):
        return cls(
            images=value["image"],
            image_masks=value["image_mask"],
            state=value["state"],
            tokenized_prompt=value["tokenized_prompt"],
            tokenized_prompt_mask=value["tokenized_prompt_mask"],
        )


def _tensor_mock():
    value = mock.Mock(spec=torch.Tensor)
    value.to.return_value = value
    return value


class Pi0DataAdapterTest(unittest.TestCase):
    def test_train_batch_uses_non_blocking_device_copies(self):
        tensors = {
            "image": {"base": _tensor_mock()},
            "image_mask": {"base": _tensor_mock()},
            "state": _tensor_mock(),
            "tokenized_prompt": _tensor_mock(),
            "tokenized_prompt_mask": _tensor_mock(),
            "actions": _tensor_mock(),
        }
        adapter = Pi0DataAdapter.__new__(Pi0DataAdapter)
        adapter.observation_type = _Observation
        device = torch.device("cuda", 0)

        batch = adapter.encode_train_batch(tensors, device)

        for tensor in (
            tensors["image"]["base"],
            tensors["image_mask"]["base"],
            tensors["state"],
            tensors["tokenized_prompt"],
            tensors["tokenized_prompt_mask"],
            tensors["actions"],
        ):
            tensor.to.assert_called_once_with(device, non_blocking=True)
        self.assertIs(batch["actions"], tensors["actions"])


if __name__ == "__main__":
    unittest.main()
