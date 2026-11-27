import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dobot_nexus.algorithms.rltoken.artifacts import write_model_bundle
from dobot_nexus.algorithms.rltoken.policy import RLTokenPolicy
from dobot_nexus.core.config import clone_config, save_config
from dobot_nexus.inference.policy import InferencePolicy
from dobot_nexus.models.pi05.deploy import server
from dobot_nexus.models.pi05.rltoken.modeling_stage1 import Pi05RLTokenModel
from dobot_nexus.modules.serving.openpi import NexusOpenPiPolicy
from examples.real_robot.shared.resolve_model import resolve_model_path
from tests.models.pi05.rltoken import test_stage1


class ComposedOpenPiTest(unittest.TestCase):
    def setUp(self):
        fixture = test_stage1.Stage1Test()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        cfg = fixture.config
        cfg.model.io = {"cameras": {"gripper": "base_0_rgb"}}
        cfg.data.field_mapping.images = {"base": "observation.images.left_wrist"}
        stage = Pi05RLTokenModel(cfg, mode="train")
        provider = write_model_bundle(fixture.root / "provider", stage, stage.root_config)
        cfg = clone_config({
            "schema_version": 1, "task": "train", "train": {},
            "data": {"field_mapping": {"images": {"base": "observation.images.head"}}},
            "algorithm": {"target": "dobot_nexus.algorithms.rltoken.training:RLTokenTrainer",
                          "reference_num_steps": 4,
                          "policy_switch": {"execution_scope": "critical_phase"}},
            "runtime": {"device": "cpu"},
            "model": {"target": "dobot_nexus.algorithms.rltoken.policy:RLTokenPolicy",
                      "features": {"artifact": str(provider)}, "rltoken": {"chunk_size": 2, "residual": {"hidden_dims": [8]}},
                      "action_horizon": 2, "state_dim": 2, "action_dim": 2},
        })
        deployment = write_model_bundle(fixture.root / "policy", RLTokenPolicy(cfg, mode="infer"), cfg)
        self.root, self.deployment, self.policy_config = fixture.root, deployment, cfg

    def test_shared_adapter_uses_trained_steps_when_inference_override_is_null(self):
        policy = InferencePolicy({
            "model": {"artifact": str(self.deployment)}, "runtime": {"device": "cpu"},
            "inference": {"num_steps": None},
        })
        cameras = {"left_camera": "gripper"}
        wire = {"state": np.zeros(2), "prompt": "pick", "control": {"rlt_switch_flags": [True]},
                "images": {"left_camera": np.zeros((3, 224, 224), dtype=np.uint8)}}
        provider = policy.model.provider
        with patch.object(provider, "extract_rl_features", wraps=provider.extract_rl_features) as extract:
            for configured, kwargs, expected in ((None, {}, 4), (7, {}, 7),
                                                 (7, {"num_steps": None}, 4), (7, {"num_steps": 6}, 6)):
                with self.subTest(configured=configured, kwargs=kwargs):
                    policy.config.inference.num_steps = configured
                    NexusOpenPiPolicy(policy, camera_mapping=cameras, **kwargs).infer(wire)
                    self.assertEqual(extract.call_args.args[1].num_steps, expected)

    def test_packaged_profile_warmup_and_phase_requests_use_saved_camera_columns(self):
        args = ["--artifact", str(self.deployment), "--device", "cpu"]
        with patch.dict("os.environ"):
            policy = server._load_policy(server.parse_args(args + ["--num-steps", "4"]))
        provider = policy.model.provider
        with patch.object(provider, "extract_rl_features", wraps=provider.extract_rl_features) as extract:
            self.assertTrue(policy.warmup().aux["actor_switch"][0])
            self.assertEqual(extract.call_args.args[1].num_steps, 4)
        self.assertEqual(policy.create_session().options.execute_steps, 2)
        metadata = server.build_openpi_metadata(policy)
        self.assertEqual(metadata["input_schema"]["images"], {"left_camera": [3, 224, 224]})
        with patch.dict(policy.config, server={"camera_mapping": {"top_camera": "gripper"}}):
            self.assertEqual(server.NexusOpenPiPolicy(policy).camera_mapping, {"top_camera": "gripper"})
            explicit = server.NexusOpenPiPolicy(policy, camera_mapping={"right_camera": "gripper"})
            self.assertEqual(explicit.camera_mapping, {"right_camera": "gripper"})
        adapter = server.NexusOpenPiPolicy(policy, num_steps=4)
        wire = {"state": np.zeros(2), "prompt": "pick",
                "images": {"left_camera": np.zeros((3, 224, 224), dtype=np.uint8)}}
        calls = []
        hook = policy.model.actor.register_forward_pre_hook(lambda *_: calls.append(True))
        self.addCleanup(hook.remove)
        for phase in (False, True):
            output = adapter.infer(dict(wire, control={"rlt_switch_flags": [phase]}))
            self.assertEqual(output["actions"].shape, (2, 2))
            self.assertEqual(len(calls), int(phase))
        with self.assertRaisesRegex(TypeError, "mapping"):
            adapter.infer(dict(wire, control=[True]))
        with self.assertRaisesRegex(ValueError, "bool"):
            adapter.infer(dict(wire, control={"rlt_switch_flags": [1]}))

        custom = self.root / "infer.yaml"
        config = clone_config(policy.config)
        config.inference.update(execute_steps=2, num_steps=7)
        save_config(config, custom)
        with patch.dict("os.environ"):
            loaded = server._load_policy(server.parse_args(args + ["--config", str(custom)]))
        self.assertEqual(loaded.config.inference.num_steps, 7)
        self.assertEqual(loaded.config.inference.execute_steps, 2)

    def test_launcher_and_server_use_trained_reference_steps_unless_overridden(self):
        resolved = resolve_model_path(self.deployment)
        self.assertTrue(resolved.norm_stats.is_relative_to(self.deployment / "deployment/features"))
        self.assertTrue(resolved.norm_stats.is_file())
        # Capture the real launcher's arguments, then run the server on CPU with
        # the fixture operators and an in-process WebSocket connection handler.
        probe = self.root / "launcher-python"
        probe.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "if sys.argv[1:3] == ['-m', 'dobot_nexus.models.pi05.deploy.server']:\n"
            "    print(json.dumps(sys.argv[3:]))\n"
            "else:\n"
            "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        )
        probe.chmod(0o755)
        launcher = Path(__file__).resolve().parents[1] / "examples/real_robot/start_policy_server.sh"
        env = dict(os.environ, POLICY_PYTHON=str(probe), NUM_STEPS="", WARMUP="1")
        wire = {"state": np.zeros(2), "prompt": "pick", "control": {"rlt_switch_flags": [True]},
                "images": {"left_camera": np.zeros((3, 224, 224), dtype=np.uint8)}}
        ac = clone_config(self.policy_config)
        ac.algorithm.recipe = "rlt_ac_v1"
        del ac.model.rltoken["residual"]
        ac_artifact = write_model_bundle(self.root / "policy-ac", RLTokenPolicy(ac, mode="infer"), ac)
        for artifact, overrides, steps in ((self.deployment, [], 4),
                                           (self.deployment, ["--num-steps", "6"], 6),
                                           (ac_artifact, [], 4)):
            with self.subTest(artifact=artifact.name, overrides=overrides):
                result = subprocess.run(["bash", str(launcher), str(artifact), "--device", "cpu",
                                         *overrides], env=env, capture_output=True, text=True, check=True)
                args = server.parse_args(json.loads(result.stdout))
                self.assertEqual(args.num_steps, 6 if overrides else None)
                extract = Pi05RLTokenModel.extract_rl_features
                with (patch.dict("os.environ"), patch.object(server, "WebsocketPolicyServer") as transport,
                      patch.object(Pi05RLTokenModel, "extract_rl_features", autospec=True,
                                   side_effect=extract) as calls):
                    server.run_server(args)
                    transport.call_args.kwargs["prepare"]()
                    with transport.call_args.kwargs["open_connection"](None) as infer:
                        self.assertEqual(infer(wire)["actions"].shape, (2, 2))
                    self.assertEqual([c.args[2].num_steps for c in calls.call_args_list], [steps, steps])
