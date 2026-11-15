from __future__ import annotations

import ast
import unittest
from pathlib import Path

from dobot_nexus.core.config import Config
from dobot_nexus.core.import_utils import build_model
from dobot_nexus.models import BaseDataAdapter, BaseVLAModule

ROOT = Path(__file__).resolve().parents[1]


def absolute_imports(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module or ""
            yield from (f"{node.module}.{alias.name}" for alias in node.names)


class RepositoryStructureTest(unittest.TestCase):
    def test_public_model_interfaces_are_exported(self) -> None:
        self.assertTrue(hasattr(BaseVLAModule, "training_step"))
        self.assertTrue(hasattr(BaseVLAModule, "predict_action_chunk"))
        self.assertTrue(hasattr(BaseDataAdapter, "encode_train_batch"))
        self.assertTrue(hasattr(BaseDataAdapter, "decode_actions"))

    def test_model_target_must_implement_the_framework_interface(self) -> None:
        config = Config({"model": Config({"target": "tests.fakes:InvalidModel"})})
        with self.assertRaisesRegex(TypeError, "must return a BaseVLAModule"):
            build_model(config, mode="train")

    def test_public_entries_do_not_embed_machine_specific_paths(self) -> None:
        roots = (
            ROOT / "docs",
            ROOT / "examples",
            ROOT / "scripts",
            ROOT / "src",
        )
        paths = [
            path
            for root in roots
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".md", ".py", ".sh", ".yaml", ".yml"}
        ]
        paths.append(ROOT / "README.md")
        content = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for prefix in ("/data/trainer/", "/share/models/", "/share/datasets/"):
            self.assertNotIn(prefix, content)

    def test_production_is_independent_of_reference_tooling(self) -> None:
        forbidden = {"jax", "jaxlib", "flax", "orbax", "openpi", "local_script", "benchmark"}
        for directory in (ROOT / "src", ROOT / "tests"):
            for path in directory.rglob("*.py"):
                for name in absolute_imports(path):
                    self.assertNotIn(name.split(".")[0], forbidden, str(path))
                    self.assertNotIn(".openpi_jax", name, str(path))

    def test_pi_operator_sources_are_owned_by_each_model(self) -> None:
        for model_name in ("pi0", "pi05"):
            model_root = ROOT / "src/dobot_nexus/models" / model_name
            local_core = model_root / (
                "torch_core" if model_name == "pi05" else "training/vendor/rlinf_core"
            )
            self.assertTrue((local_core / "pi0.py").is_file())
            self.assertTrue((local_core / "LICENSE").is_file())
            source = (local_core if model_name == "pi05" else model_root / "training") / "source.py"
            self.assertTrue(source.is_file())
            self.assertTrue((model_root / "training/checkpoint.py").is_file())
            self.assertTrue((model_root / "data/pipeline.py").is_file())
            other = "pi0" if model_name == "pi05" else "pi05"
            for path in model_root.rglob("*.py"):
                for name in absolute_imports(path):
                    self.assertNotIn(name.split(".")[0], {"rlinf", "ray", "omegaconf"}, str(path))
                    self.assertFalse(name.startswith(f"dobot_nexus.models.{other}."), str(path))
                    self.assertNotEqual(name, f"dobot_nexus.models.{other}", str(path))


if __name__ == "__main__":
    unittest.main()
