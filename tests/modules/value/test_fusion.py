import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dobot_nexus.modules.value.fusion.configuration import ValueModelConfig
from dobot_nexus.modules.value.losses import categorical_loss


class FusionTest(unittest.TestCase):
    def test_fusion_learns_from_images_and_state_tokens_and_restores_strictly(self):
        from dobot_nexus.modules.value.fusion.modeling_fusion import DobotValueFusion

        class VisionEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(3, 4)

            def forward(self, pixel_values, return_dict=True):
                return SimpleNamespace(
                    last_hidden_state=self.proj(pixel_values.mean((2, 3)))[:, None]
                )

        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=4))
                self.vision_model = VisionEncoder()

        class Language(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=6)
                self.embedding = torch.nn.Embedding(10, 6)

            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        with tempfile.TemporaryDirectory() as directory:
            config = ValueModelConfig(Path(directory), Path(directory), Path(directory))
            model = DobotValueFusion(config, vision=Vision(), language=Language())
            batch = {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "attention_mask": torch.ones(2, 2, dtype=torch.long),
                "image_tensor": torch.randn(2, 3, 3, 8, 8),
                "image_attention_mask": torch.ones(2, 3, dtype=torch.long),
            }
            logits = model(batch)
            categorical_loss(logits, torch.tensor([-0.2, -0.7]))[0].mean().backward()
            for parameter in (
                model.vision.vision_model.proj.weight,
                model.language.embedding.weight,
                model.value_queries,
            ):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
            restored = DobotValueFusion(config, vision=Vision(), language=Language())
            restored.load_state_dict(model.state_dict(), strict=True)
            model.eval()
            restored.eval()
            torch.testing.assert_close(model(batch), restored(batch), rtol=0, atol=0)
