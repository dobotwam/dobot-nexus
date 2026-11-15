from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml

from dobot_nexus.core.contracts import PolicyOutput, PolicySpec
from dobot_nexus.models.pi05.deploy import openpi_msgpack
from dobot_nexus.models.pi05.deploy.server import (
    NexusOpenPiPolicy,
    _load_policy,
    _policy_identity,
    build_nexus_observation,
    parse_args,
    run_server,
)


def openpi_observation() -> dict:
    top = np.zeros((3, 224, 224), dtype=np.uint8)
    left = np.full((3, 224, 224), 1, dtype=np.uint8)
    right = np.full((3, 224, 224), 2, dtype=np.uint8)
    return {
        "state": np.arange(14, dtype=np.float32),
        "images": {
            "top_camera": top,
            "left_camera": left,
            "right_camera": right,
        },
        "prompt": "Pick up the red cube.",
    }


class FakePolicy:
    def __init__(self, state_dim=14, action_dim=14) -> None:
        self.spec = PolicySpec(
            model_name="pi05",
            implementation="pi05-pytorch",
            camera_names=("base", "left_wrist", "right_wrist"),
            state_dim=state_dim,
            action_dim=action_dim,
            action_horizon=50,
        )
        self.artifact_dir = None
        self.manifest = None
        self.calls: list[tuple[dict, object]] = []

    def predict_chunk(self, observation, options):
        self.calls.append((observation, options))
        actions = np.full((1, 50, self.spec.action_dim), float(options.seed), dtype=np.float32)
        return PolicyOutput(actions=actions, timing={"total_ms": 3.5})


class Pi05OpenPiDeploymentTest(unittest.TestCase):
    def test_artifact_selects_acceleration_profile_and_recap_condition(self):
        targets = {
            "modeling_pi05:Pi05Model": "legacy",
            "modeling_recap:Pi05ReCapModel": "legacy",
            "recap.modeling_recap:Pi05ReCapModel": "legacy",
            "memory.modeling_memory:Pi05MemoryModel": "openpi",
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {}):
            root = Path(directory)
            for suffix, mode in targets.items():
                with self.subTest(target=suffix):
                    target = f"dobot_nexus.models.pi05.{suffix}"
                    (root / "manifest.yaml").write_text(yaml.safe_dump({
                        "schema_version": 2, "model": {
                            "target": target, "state_dim": 14, "action_dim": 14,
                        },
                    }))
                    args = parse_args(["--artifact", directory])
                    if "ReCap" in suffix:
                        args.recap_condition = "positive"
                    with mock.patch("dobot_nexus.inference.load_policy", side_effect=lambda c: c):
                        config = _load_policy(args)
                    self.assertEqual(config.model.target, target)
                    self.assertEqual(config.model.artifact, str(root))
                    self.assertTrue(config.runtime.compile)
                    self.assertTrue(config.runtime.cuda_graph)
                    self.assertEqual(config.runtime.compile_mode, mode)
                    if "ReCap" in suffix:
                        self.assertEqual(config.inference.recap.condition, "positive")
                    args.torch_compile = args.cuda_graph = False
                    with mock.patch("dobot_nexus.inference.load_policy", side_effect=lambda c: c):
                        config = _load_policy(args)
                    self.assertFalse(config.runtime.compile)
                    self.assertFalse(config.runtime.cuda_graph)

    def test_policy_entry_supports_configured_robot_dimensions(self):
        for state_dim, action_dim in ((14, 14), (34, 34), (24, 24), (12, 7)):
            with self.subTest(state=state_dim, action=action_dim):
                wrapper = NexusOpenPiPolicy(FakePolicy(state_dim, action_dim))
                observation = openpi_observation()
                observation["state"] = np.zeros(state_dim, np.float32)
                result = wrapper.infer(observation)
                self.assertEqual(result["actions"].shape, (50, action_dim))
                with tempfile.TemporaryDirectory() as directory, mock.patch.dict("os.environ", {}):
                    (Path(directory) / "manifest.yaml").write_text(yaml.safe_dump({
                        "schema_version": 2, "model": {
                            "state_dim": state_dim, "action_dim": action_dim,
                            "target": "dobot_nexus.models.pi05.modeling_pi05:Pi05Model",
                        },
                    }))
                    with mock.patch("dobot_nexus.inference.load_policy", side_effect=lambda c: c):
                        config = _load_policy(parse_args(["--artifact", directory]))
                    self.assertIsNone(config.model.state_dim)  # Restored by InferencePolicy.

    def test_memory_entry_warms_disposable_session_and_preserves_seed(self):
        from dobot_nexus.models.pi05.deploy.memory_server import MemoryWebsocketServer
        from dobot_nexus.models.pi05.memory.configuration import MemoryConfig

        fake = FakePolicy()
        fake.metadata = {}
        fake.model = SimpleNamespace(
            memory_config=MemoryConfig(budget=16, max_frames=4),
            inference_metadata={"backend": "torch", "torch_compile": True, "cuda_graph": True},
        )
        fake.config = SimpleNamespace(inference=SimpleNamespace(execute_steps=8))
        captured = []
        encoder = lambda frames: np.zeros((len(frames), 16, 2048), np.float32)
        with (mock.patch("dobot_nexus.models.pi05.deploy.server._load_policy", return_value=fake),
              mock.patch("dobot_nexus.models.pi05.deploy.memory_server.build_frame_encoder",
                         return_value=encoder),
              mock.patch.object(MemoryWebsocketServer, "serve_forever", autospec=True,
                                side_effect=lambda server: (server.prepare(), captured.append(server)))):
            run_server(parse_args(["--artifact", "/test", "--seed", "42"]))
        server = captured[0]
        self.assertTrue(server.metadata["history_requires_dense_frames"])
        self.assertTrue(server.metadata["torch_compile"])
        self.assertEqual(len(fake.calls), 1)  # warmup uses its own session
        self.assertEqual(fake.calls[0][1].seed, 0)
        first, second = server._connection_policy(), server._connection_policy()
        self.assertFalse(first.session.buffers)
        self.assertEqual(first.infer(openpi_observation())["seed"], 42)
        self.assertEqual(first.infer(openpi_observation())["seed"], 43)
        self.assertFalse(second.session.buffers)
        self.assertEqual(second.infer(openpi_observation())["seed"], 42)
        first.infer({"reset": True, "seed": 80})
        self.assertFalse(first.session.buffers)
        self.assertEqual(first.infer(openpi_observation())["seed"], 80)
        server.vary_seed = False
        fixed = server._connection_policy()
        self.assertEqual(fixed.infer(openpi_observation())["seed"], 42)
        self.assertEqual(fixed.infer(openpi_observation())["seed"], 42)

    def test_both_memory_entrypoints_apply_configured_session_quotas(self):
        from dobot_nexus.core.config import Config
        from dobot_nexus.models.pi05.deploy import memory_server
        from dobot_nexus.models.pi05.memory.configuration import MemoryConfig

        for limits in ({}, {"memory_max_environments": 2, "memory_max_history_bytes": 1024**3}):
            with self.subTest(limits=limits):
                fake = FakePolicy()
                fake.metadata = {}
                fake.model = SimpleNamespace(memory_config=MemoryConfig(budget=16, max_frames=4))
                fake.config = Config(inference=Config(execute_steps=8, num_steps=10, warmup=False),
                                     server=Config(limits))
                encoder = lambda frames: np.zeros((len(frames), 16, 2048), np.float32)
                captured = []
                with (
                    mock.patch("dobot_nexus.core.config.load_config", return_value=fake.config),
                    mock.patch("dobot_nexus.inference.policy.load_policy", return_value=fake),
                    mock.patch("dobot_nexus.models.pi05.deploy.server._load_policy", return_value=fake),
                    mock.patch.object(memory_server, "build_frame_encoder", return_value=encoder),
                    mock.patch.object(memory_server.MemoryWebsocketServer, "serve_forever", autospec=True,
                                      side_effect=lambda server: captured.append(server)),
                    mock.patch("sys.argv", ["memory_server", "config.yaml"]),
                ):
                    memory_server.main()
                    run_server(parse_args(["--artifact", "/test", "--no-warmup"]))
                self.assertEqual(len(captured), 2)
                for server in captured:
                    session = server._connection_policy().session
                    self.assertEqual(session.max_environments, limits.get("memory_max_environments", 4))
                    self.assertEqual(session.max_history_bytes, limits.get("memory_max_history_bytes", 256*1024**2))

    def test_memory_connection_context_isolates_history_and_cleans_up(self):
        from dobot_nexus.models.pi05.deploy.memory_server import MemoryWebsocketServer
        from dobot_nexus.models.pi05.memory.configuration import MemoryConfig

        fake = FakePolicy()
        fake.metadata = {}
        encoder = mock.Mock(side_effect=lambda frames: np.zeros((len(frames), 16, 2048), np.float32))
        deployment = MemoryWebsocketServer(
            fake, encoder, MemoryConfig(budget=16, max_frames=4),
            execute_steps=8, num_steps=10, host="127.0.0.1", port=0, seed=42,
        )
        with deployment.open_connection(None) as first, deployment.open_connection(None) as second:
            a, b = first.__self__, second.__self__
            # Identical environment identifiers still belong to different connections.
            first({"add_buffer": True, "history_frames": [np.zeros((224, 224, 3), np.uint8)],
                   "step_indices": [0], "prompt": "Pick up the red cube."})
            self.assertFalse(b.session.buffers)
            first({**openpi_observation(), "step_index": 0})
            self.assertEqual(encoder.call_count, 1)  # control fields survive wire adaptation
            second(openpi_observation())
            first({"reset": True, "seed": 70})
            self.assertFalse(a.session.buffers)
            self.assertEqual(b.session.buffers["default"].last_step, 0)
            self.assertEqual(second({**openpi_observation(), "step_index": 1})["seed"], 43)
        self.assertFalse(a.session.buffers)
        self.assertFalse(b.session.buffers)
        self.assertIs(a.session.encoder, b.session.encoder)
        self.assertIs(a.session.policy, b.session.policy)

    def test_torch_acceleration_cli_inherits_profile_unless_overridden(self) -> None:
        args = parse_args(["--artifact", "/test", "--backend", "torch",
                           "--torch-compile", "--cuda-graph"])
        self.assertTrue(args.torch_compile)
        self.assertTrue(args.cuda_graph)
        defaults = parse_args(["--artifact", "/test"])
        self.assertEqual(defaults.backend, "torch")
        self.assertIsNone(defaults.torch_compile)
        self.assertIsNone(defaults.cuda_graph)
        self.assertEqual(parse_args(["--artifact", "/test", "--compile-mode", "legacy"]).compile_mode,
                         "legacy")

    def test_server_identity_uses_runtime_torch_source(self) -> None:
        fake = FakePolicy()
        fake.model = type(
            "RuntimeModel",
            (),
            {
                "inference_metadata": {
                    "backend": "torch",
                    "operator_source": "/opt/nexus/torch_core",
                    "operator_revision": "abc123",
                }
            },
        )()
        identity = _policy_identity(fake)
        self.assertEqual(identity["backend"], "torch")
        self.assertEqual(identity["operator_source"], "/opt/nexus/torch_core")
        self.assertEqual(identity["operator_revision"], "abc123")

    def test_production_cli_rejects_reference_backend_and_paths(self) -> None:
        import contextlib
        import io

        defaults = parse_args(["--artifact", "/test"])
        self.assertFalse(hasattr(defaults, "openpi_root"))
        self.assertFalse(hasattr(defaults, "jax_checkpoint"))
        for option in (["--backend", "jax"], ["--openpi-root", "/old"],
                       ["--jax-checkpoint", "/old/weights"]):
            error = io.StringIO()
            with self.subTest(option=option), contextlib.redirect_stderr(error):
                with self.assertRaises(SystemExit) as raised:
                    parse_args(["--artifact", "/test", *option])
                self.assertEqual(raised.exception.code, 2)
                if option[0] == "--backend":
                    self.assertIn("reference_jax", error.getvalue())

    def test_openpi_chw_observation_maps_to_nexus_hwc_contract(self) -> None:
        canonical = build_nexus_observation(openpi_observation())
        self.assertEqual(set(canonical["images"]), {"base", "left_wrist", "right_wrist"})
        self.assertEqual(canonical["images"]["base"].shape, (224, 224, 3))
        self.assertTrue(canonical["images"]["base"].flags.c_contiguous)
        self.assertEqual(canonical["state"].shape, (14,))
        self.assertEqual(canonical["prompt"], ["Pick up the red cube."])

    def test_openpi_observation_rejects_missing_or_wrong_image(self) -> None:
        missing = openpi_observation()
        del missing["images"]["left_camera"]
        with self.assertRaisesRegex(KeyError, "left_camera"):
            build_nexus_observation(missing)

        wrong_dtype = openpi_observation()
        wrong_dtype["images"]["top_camera"] = np.zeros((3, 224, 224), dtype=np.float32)
        with self.assertRaisesRegex(TypeError, "uint8"):
            build_nexus_observation(wrong_dtype)

    def test_openpi_default_prompt_is_supported(self) -> None:
        observation = openpi_observation()
        del observation["prompt"]
        canonical = build_nexus_observation(observation, default_prompt="Pick up the red cube.")
        self.assertEqual(canonical["prompt"], ["Pick up the red cube."])

    def test_openpi_policy_returns_unbatched_action_chunk_and_advances_seed(self) -> None:
        fake = FakePolicy()
        adapter = NexusOpenPiPolicy(fake, seed=42, num_steps=10, vary_seed=True)
        first = adapter.infer(openpi_observation())
        second = adapter.infer(openpi_observation())
        self.assertEqual(first["actions"].shape, (50, 14))
        self.assertEqual(first["actions"].dtype, np.float32)
        self.assertEqual(first["state"].shape, (14,))
        self.assertEqual(first["seed"], 42)
        self.assertEqual(second["seed"], 43)
        self.assertEqual(fake.calls[0][1].num_steps, 10)
        self.assertEqual(set(fake.calls[0][0]["images"]), {"base", "left_wrist", "right_wrist"})

    def test_openpi_msgpack_numpy_round_trip_preserves_arrays(self) -> None:
        observation = openpi_observation()
        decoded = openpi_msgpack.unpackb(openpi_msgpack.packb(observation))
        canonical = build_nexus_observation(decoded)
        np.testing.assert_array_equal(canonical["state"], observation["state"])
        np.testing.assert_array_equal(
            canonical["images"]["left_wrist"],
            np.transpose(observation["images"]["left_camera"], (1, 2, 0)),
        )


if __name__ == "__main__":
    unittest.main()
