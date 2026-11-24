import subprocess
import sys
import unittest


class ReCapDependenciesTest(unittest.TestCase):
    def run_isolated(self, code):
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_recipe_and_cli_do_not_load_torch_or_optional_backbones(self):
        self.run_isolated("""
import sys
for name in ('torch', 'transformers', 'pyarrow', 'lerobot', 'dobot_nexus.models'):
    sys.modules[name] = None
from dobot_nexus.algorithms.recap.configuration import parse_offline_config
from dobot_nexus.algorithms.recap.cli import main
""")

    def test_offline_jobs_do_not_import_concrete_policy_models(self):
        self.run_isolated("""
import sys
for name in ('dobot_nexus.models.pi05', 'dobot_nexus.models.pi0',
             'transformers', 'lerobot', 'pyarrow', 'ray', 'rlinf'):
    sys.modules[name] = None
from dobot_nexus.algorithms.recap import annotations, conditioning, data, labeling, targets
from dobot_nexus.algorithms.recap.value import checkpoint, dataset, processing, training
""")

    def test_policy_does_not_import_value_training_or_distributed_execution(self):
        self.run_isolated("""
import sys
for name in ('dobot_nexus.algorithms.recap.value.training',
             'dobot_nexus.algorithms.recap.value.distributed', 'transformers', 'ray', 'rlinf'):
    sys.modules[name] = None
from dobot_nexus.models.pi0.modeling_recap import Pi0ReCapModel
from dobot_nexus.models.pi05.modeling_recap import Pi05ReCapModel
from dobot_nexus.models.pi05.recap.modeling_recap import Pi05ReCapModel as saved_target
assert saved_target is Pi05ReCapModel
""")
