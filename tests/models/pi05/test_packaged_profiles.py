import unittest
from importlib.resources import as_file, files
from pathlib import Path

from dobot_nexus.core.config import load_config


class PackagedProfilesTest(unittest.TestCase):
    def test_inference_profiles_are_self_contained_and_match_examples(self):
        root = Path(__file__).resolve().parents[3]
        for name in ('pi05.yaml', 'pi05_recap.yaml', 'pi05_memory.yaml', 'pi05_rltoken.yaml'):
            resource = files('dobot_nexus.models.pi05.deploy.profiles').joinpath(name)
            with as_file(resource) as path:
                config = load_config(path, expected_task='infer')
                self.assertEqual(config.task, 'infer')
                self.assertEqual(path.read_bytes(), (root/'examples/configs/infer'/name).read_bytes())
