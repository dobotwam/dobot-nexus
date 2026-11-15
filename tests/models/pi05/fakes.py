"""CPU fixtures shared by Pi0.5 adapter and RECAP policy tests."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class FakeObservation:
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
