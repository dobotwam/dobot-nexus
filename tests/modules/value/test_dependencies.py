import subprocess
import sys
import tempfile
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest import mock

from dobot_nexus.modules.value import fusion
from dobot_nexus.modules.value.fusion import require_value_dependencies


class ValueDependenciesTest(unittest.TestCase):
    def test_config_and_factory_import_without_torch_or_transformers(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
sys.modules['torch'] = None
sys.modules['transformers'] = None
from dobot_nexus.modules.value.fusion import build_value_model
from dobot_nexus.modules.value.fusion.configuration import ValueModelConfig
""",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_network_processing_and_losses_do_not_import_model_or_job_layers(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
for name in ('dobot_nexus.models', 'dobot_nexus.training', 'dobot_nexus.data',
             'dobot_nexus.algorithms',
             'transformers', 'lerobot'):
    sys.modules[name] = None
from dobot_nexus.modules.value.fusion.modeling_fusion import DobotValueFusion
from dobot_nexus.modules.value.fusion.processing import FusionProcessor
from dobot_nexus.modules.value.losses import categorical_loss
from dobot_nexus.modules.value.distributions import expected_value
""",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_dependency_error_identifies_the_shared_installation_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            installed_module = Path(directory) / "site-packages" / "fusion" / "__init__.py"
            for installed in ("0.0", None):
                with (
                    self.subTest(installed=installed),
                    mock.patch.object(fusion, "__file__", str(installed_module)),
                    mock.patch(
                        "importlib.metadata.version",
                        return_value=installed,
                        side_effect=PackageNotFoundError("transformers")
                        if installed is None
                        else None,
                    ),
                    self.assertRaises(ImportError) as error,
                ):
                    require_value_dependencies()
                self.assertIn(
                    str(installed_module.with_name("requirements.txt")), str(error.exception)
                )
                self.assertIn(f"found {installed or 'missing'}", str(error.exception))
