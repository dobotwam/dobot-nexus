"""Import paths, store schemas and sampling semantics of the two FrameSample integrations."""

from __future__ import annotations

import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dobot_nexus.core.config import Config
from dobot_nexus.core.contracts import PolicyOutput, PolicySpec
from dobot_nexus.inference.session import SessionOptions
from dobot_nexus.modules.memory.feature_store import DEFAULT_MAX_CACHED_BYTES, FrameFeatureStore
from dobot_nexus.modules.memory.position import position_embeddings
from dobot_nexus.modules.memory.types import MemoryConfig


def memory_module(model, name):
    return import_module(f"dobot_nexus.models.{model}.memory.{name}")


class ModelMemoryCompatibilityTest(unittest.TestCase):
    def test_both_training_pipelines_apply_the_configured_snapshot_byte_limit(self):
        memory = MemoryConfig(budget=16, vision_sha256="a" * 64)
        model_config = SimpleNamespace(camera_mapping={"base": "base_0_rgb"})
        columns = {"image_field": "images.top", "language_field": "task"}

        def initialize(pipeline, config, model):
            pipeline.model_config = model
            pipeline.image_fields = {"base_0_rgb": columns["image_field"]}
            pipeline.language_key = columns["language_field"]

        for model in ("pi0", "pi05"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                memory_module(model, "feature_store").write_episode(
                    directory, 0, np.zeros((1, 16, 2048)), [0], config=memory,
                    dataset_id="test", **columns,
                )
                config = Config(task="train", data=Config(memory_features=directory,
                                                          memory_dataset_id="test"))
                pipeline = memory_module(model, "data").MemoryPipeline
                with patch.object(pipeline.__base__, "__init__", initialize):
                    for options, expected in (({}, DEFAULT_MAX_CACHED_BYTES),
                                              ({"memory_cache_bytes": 500}, 500)):
                        with patch.dict(config.data, options):
                            store = pipeline(config, model_config, memory).store
                            self.assertEqual(store.max_cached_bytes, expected)
                            store.close()
                    for invalid in (0, -1, True, 1.5, None):
                        with (
                            self.subTest(invalid=invalid),
                            patch.dict(config.data, memory_cache_bytes=invalid),
                            self.assertRaisesRegex(ValueError, "memory_cache_bytes.*positive"),
                        ):
                            pipeline(config, model_config, memory)

    def test_legacy_direct_imports(self):
        # Include imported helpers as well as classes/functions defined at the old paths.
        common = {
            "configuration": "asdict dataclass fields Path clone_config MODEL_TARGET "
            "MemoryConfig resolve_config",
            "feature_store": "hashlib json os tempfile OrderedDict Path np assemble_memory "
            "select_frames content_sha256 feature_spec write_episode FrameFeatureStore",
            "position": "np position_embeddings",
            "runtime": "OrderedDict replace RLock np ObservationBatch PolicySession SessionOptions "
            "assemble_memory select_frames MemoryObservationBatch MemorySnapshot "
            "_step_index HistoryBuffer MemorySession",
            "sampling": "np position_embeddings MemorySnapshot select_frames assemble_memory",
            "torch": "torch nn F _rms MemoryTokenEncoder MemoryAttention "
            "MemoryModulation install_memory",
            "types": "dataclass np ObservationBatch MemorySnapshot MemoryObservationBatch",
        }
        modules = {
            f"dobot_nexus.models.{model}.memory.{name}": names
            for model in ("pi0", "pi05")
            for name, names in common.items()
        }
        pi0 = "dobot_nexus.models.pi0.memory."
        modules[pi0 + "configuration"] += " restore_token_limit"
        modules[pi0 + "feature_store"] += " _episode_digest _file_identity _map_episode"
        modules[pi0 + "runtime"] += " Sized contextmanager copy islice"
        deploy = "dobot_nexus.models.pi05.deploy."
        modules[deploy + "memory_server"] = (
            "argparse io Mapping contextmanager np SamplingOptions SessionOptions "
            "WebsocketPolicyServer OPENPI_TO_NEXUS_IMAGES _policy_identity build_nexus_observation "
            "MemorySession MemoryConnection MemoryWebsocketServer configured_memory_limits "
            "build_frame_encoder main"
        )
        modules[deploy + "server"] = (
            "argparse logging os time Iterable Mapping nullcontext as_file files Path Any np "
            "PredictionHandler WebsocketPolicyServer OPENPI_TO_NEXUS_IMAGES OPENPI_PROTOCOL_NAME "
            "_image_to_hwc_uint8 build_nexus_observation _policy_identity NexusOpenPiPolicy "
            "_load_policy run_server _torch_backend parse_args main"
        )
        for path, names in modules.items():
            with self.subTest(module=path):
                module = import_module(path)
                legacy = set(names.split())
                self.assertFalse(legacy.difference(vars(module)))
                public = {name for name in legacy if not name.startswith("_")}
                self.assertFalse(public.difference(getattr(module, "__all__", vars(module))))

    def test_model_readers_and_writers_are_locked_to_their_store_schema(self):
        legacy = memory_module("pi05", "feature_store")
        current = memory_module("pi0", "feature_store")
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        features = np.ones((1, 16, 2048), np.float32)
        write_args = dict(config=config, dataset_id="set")
        columns = dict(image_field="rgb", language_field=None)
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "v1", Path(directory) / "v2"
            legacy.write_episode(first, 0, features, [0], **write_args)
            current.write_episode(second, 0, features, [0], **write_args, **columns)
            with self.assertRaisesRegex(ValueError, "confirm its original"):
                legacy.FrameFeatureStore(first, config)
            v1 = legacy.FrameFeatureStore(first, config, confirmed_legacy_columns=columns).manifest[
                "feature_spec"
            ]
            self.assertEqual(v1, legacy.feature_spec(config))
            self.assertEqual(v1["schema_version"], 1)
            self.assertTrue(v1.keys().isdisjoint(columns))
            v2 = current.FrameFeatureStore(second, config, **columns).manifest["feature_spec"]
            self.assertEqual(v2, current.feature_spec(config, **columns))
            self.assertEqual(v2["schema_version"], 2)
            self.assertEqual({key: v2[key] for key in columns}, columns)
            with self.assertRaises(TypeError):
                current.FrameFeatureStore(first, config)
            with self.assertRaisesRegex(ValueError, "schema v2"):
                current.FrameFeatureStore(first, config, **columns)
            with self.assertRaisesRegex(ValueError, "schema v1"):
                legacy.FrameFeatureStore(second, config)
            with self.assertRaisesRegex(ValueError, "identity"):
                current.write_episode(first, 1, features, [0], **write_args, **columns)
            with self.assertRaisesRegex(ValueError, "identity"):
                legacy.write_episode(second, 1, features, [0], **write_args)
            with self.assertRaisesRegex(ValueError, "image_field"):
                current.feature_spec(config, image_field=None, language_field=None)

    def test_one_frame_offline_and_online_share_each_models_reference_rule(self):
        frames = np.stack([np.full((16, 2048), i, np.float32) for i in range(5)])
        policy = SimpleNamespace(spec=SimpleNamespace(action_horizon=2))
        for model in ("pi0", "pi05"):
            store_module = memory_module(model, "feature_store")
            runtime = memory_module(model, "runtime")
            columns = dict(image_field="rgb", language_field=None) if model == "pi0" else {}
            for window, start in ((None, 1), (2, 3)):
                with self.subTest(model=model, window=window), tempfile.TemporaryDirectory() as tmp:
                    config = MemoryConfig(budget=16, vision_sha256="a" * 64, window_frames=window)
                    store_module.write_episode(
                        tmp, 0, frames, [0, 1, 1, 1, 1], config=config, dataset_id="set", **columns
                    )
                    read_args = columns if model == "pi0" else {
                        "confirmed_legacy_columns": {"image_field": "rgb", "language_field": None}
                    }
                    offline = store_module.FrameFeatureStore(tmp, config, **read_args).lookup(0, 4)
                    rule = "current" if model == "pi0" else "start"
                    selected = 4 if model == "pi0" else start
                    for module in (memory_module(model, "sampling"), runtime, store_module):
                        indices, offsets = module.select_frames(
                            4, 1, segment_start=1, window_frames=window
                        )
                        np.testing.assert_array_equal(indices, [selected])
                        np.testing.assert_array_equal(offsets, [selected - start])
                    shared_reader = FrameFeatureStore(tmp, config, single_frame=rule, **columns)
                    self.addCleanup(shared_reader.close)
                    direct = shared_reader.lookup(0, 4)
                    np.testing.assert_array_equal(direct.features, offline.features)
                    np.testing.assert_array_equal(direct.positions, offline.positions)
                    override = "start" if model == "pi0" else "current"
                    for reader, args in ((FrameFeatureStore, columns),
                                         (store_module.FrameFeatureStore, read_args)):
                        other = reader(tmp, config, single_frame=override, **args).lookup(0, 4)
                        expected_frame = start if model == "pi0" else 4
                        np.testing.assert_array_equal(other.features, expected_frame)
                    buffer = runtime.HistoryBuffer(config)
                    for step in range(1, 5):
                        buffer.append(frames[step], step)
                    session = runtime.MemorySession(policy, np.stack, config, SessionOptions(1))
                    session.push_frames(frames[1:], range(1, 5), prompt="pick")
                    obs = {
                        "images": {"base": np.zeros((2, 4, 3), np.uint8)},
                        "state": [0.0],
                        "prompt": "pick",
                    }
                    prepared = session.prepare(obs, step_indices=[4])
                    chosen = 4 if model == "pi0" else start
                    offsets = position_embeddings([chosen - start])[0]
                    np.testing.assert_array_equal(offline.features, chosen)
                    np.testing.assert_array_equal(offline.positions, offsets)
                    np.testing.assert_array_equal(buffer.snapshot().features, offline.features)
                    np.testing.assert_array_equal(prepared.memory.features[0], offline.features)
                    np.testing.assert_array_equal(prepared.memory.positions[0], offsets)

    def test_pi05_websocket_connection_uses_legacy_one_frame_rule(self):
        deploy = import_module("dobot_nexus.models.pi05.deploy.memory_server")
        policy = Mock(spec=["spec", "metadata", "predict_chunk"])
        policy.spec = SimpleNamespace(action_horizon=2, camera_names=("base",))
        policy.metadata = {}
        policy.predict_chunk.return_value = PolicyOutput(np.ones((1, 2, 1), np.float32))

        def encode(frames):
            return np.stack([np.full((16, 2048), frame.mean(), np.float32) for frame in frames])

        server = deploy.MemoryWebsocketServer(
            policy,
            encode,
            MemoryConfig(budget=16),
            camera_mapping={"overhead": "base"},
            host="localhost",
            port=0,
            execute_steps=1,
            num_steps=1,
        )
        with server.open_connection(None) as infer:
            for value in (3, 9):
                infer(
                    {
                        "images": {"base": np.full((2, 4, 3), value, np.uint8)},
                        "state": [0.0],
                        "prompt": "pick",
                    }
                )
        np.testing.assert_array_equal(policy.predict_chunk.call_args.args[0].memory.features, 3)

    def test_pi05_memory_launchers_serve_renamed_artifact_cameras_in_both_protocols(self):
        bc = import_module("dobot_nexus.models.pi05.deploy.server")
        deploy = import_module("dobot_nexus.models.pi05.deploy.memory_server")
        slots = {
            "right_wrist": "right_wrist_0_rgb",
            "front": "base_0_rgb",
            "left_wrist": "left_wrist_0_rgb",
        }
        spec = PolicySpec("pi05", "stub", tuple(slots), 1, 1, 2, image_size=(2, 4))
        config = Config(inference=Config(execute_steps=1, num_steps=1, warmup=True, seed=4))
        config.data = {"field_mapping": {"images": {
            "base": "observation.images.top_camera", "left_wrist": "observation.images.left_camera",
            "right_wrist": "observation.images.right_camera",
        }}}
        policy = SimpleNamespace(
            config=config, spec=spec, metadata=spec.to_dict(), artifact_dir=None,
            model=SimpleNamespace(pi05_config=SimpleNamespace(camera_mapping=slots),
                                  memory_config=MemoryConfig(camera="front", budget=16)),
            predict_chunk=Mock(return_value=PolicyOutput(np.ones((1, 2, 1)))),
        )

        def encode(frames):
            return np.stack([np.full((16, 2048), frame.mean(), np.float32) for frame in frames])

        wire = {"top_camera": "front", "left_camera": "left_wrist", "right_camera": "right_wrist"}
        native = {name: np.full((2, 4, 3), i, np.uint8) for i, name in enumerate(slots, 1)}
        openpi = {name: native[public].transpose(2, 0, 1) for name, public in wire.items()}
        with (
            patch.object(bc, "_load_policy", return_value=policy),
            patch("dobot_nexus.core.config.load_config", return_value=config),
            patch("dobot_nexus.inference.policy.load_policy", return_value=policy),
            patch.object(deploy, "build_frame_encoder", return_value=encode),
            patch.object(deploy.MemoryWebsocketServer, "serve_forever", autospec=True) as serve,
            patch("sys.argv", ["memory_server", "config.yaml"]),
        ):
            bc.run_server(bc.parse_args(["--artifact", "fixture"]))
            deploy.main()
        self.assertEqual(serve.call_count, 2)
        for call in serve.call_args_list:
            server = call.args[0]
            self.assertEqual(server.camera_mapping, wire)
            self.assertEqual(server.prepare()["history_camera"], "front")
            for images, shape in ((native, (1, 2, 1)), (openpi, (2, 1))):
                with server.open_connection(None) as infer:
                    out = infer({"images": images, "state": [0.0], "prompt": "pick"})
                self.assertEqual(out["actions"].shape, shape)
                batch = policy.predict_chunk.call_args.args[0]
                for name in native:
                    np.testing.assert_array_equal(batch.images[name][0], native[name])
                np.testing.assert_array_equal(batch.memory.features[0], native["front"].mean())
