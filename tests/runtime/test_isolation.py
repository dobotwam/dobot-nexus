import asyncio
import copy
import importlib
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from dobot_nexus.runtime.contracts import (
    ChannelEmpty,
    ChannelFull,
    ChannelSpec,
    PlacementSpec,
    RoleSpec,
    TopologySpec,
)


class Probe:
    rpc_methods = ("ping",)

    def __init__(self, token):
        self.token = token

    def initialize(self, ctx):
        return (ctx.role_name, ctx.rank, ctx.world_size, ctx.device, ctx.channels)

    def ping(self, value):
        return self.token + value

    def close(self):
        self.closed = True


class IsolationTest(unittest.TestCase):
    def test_public_imports_in_clean_interpreter(self):
        modules = (
            "runtime",
            "training",
            "inference",
            "models.base",
            "modules.value",
            "algorithms.rltoken.roles",
            "algorithms.rltoken.topology",
            "algorithms.rltoken.training",
            "algorithms.rltoken.token_pretraining",
            "algorithms.rltoken.pretraining",
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import importlib, sys
for name in sys.argv[1:]:
    importlib.import_module("dobot_nexus." + name)
    assert not {"ray", "rlinf"}.intersection(sys.modules), name
""",
                *modules,
            ],
            check=True,
        )

    def test_packaged_pins_and_license(self):
        backend = files("dobot_nexus.runtime.ray")
        algorithm = files("dobot_nexus.algorithms.rltoken")
        for requirement in (backend / "requirements.txt", algorithm / "requirements-rlinf.txt"):
            pins = [s for s in requirement.read_text().splitlines() if s and not s.startswith("#")]
            self.assertCountEqual(
                pins,
                [
                    "rlinf @ git+https://github.com/RLinf/RLinf.git@7e90641c4135229db6d3426950cdb45cee6259a6",
                    "ray[default]==2.56.1",
                ],
            )
        self.assertEqual((backend / "LICENSE").read_bytes(), (algorithm / "LICENSE").read_bytes())
        self.assertIn("7e90641c", (backend / "NOTICE.md").read_text())


class RayAdapterTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("RAY_ADDRESS", None)
        ray = ModuleType("ray")
        self.node = SimpleNamespace(_global_node=None)
        ray._private = SimpleNamespace(worker=self.node)
        ray.is_initialized = Mock(return_value=False)
        ray.shutdown = Mock()
        ray.kill = Mock()
        ray.get = Mock(side_effect=lambda ref, *, timeout: ref())
        ray_constants = ModuleType("ray._common.constants")
        ray_constants.HEAD_NODE_RESOURCE_NAME = "node:__internal_head__"
        usage_constants = ModuleType("ray._common.usage.usage_constants")
        usage_constants.CLUSTER_METADATA_KEY = b"CLUSTER_METADATA"
        private_constants = ModuleType("ray._private.ray_constants")
        private_constants.KV_NAMESPACE_CLUSTER = b"cluster"
        kv = ModuleType("ray.experimental.internal_kv")
        kv._internal_kv_get = self.kv_get = Mock(return_value=b'{"ray_init_cluster": false}')
        ray.nodes = Mock(return_value=[{"Alive": True, "Resources": {"node:__internal_head__": 1}}])
        cluster_module = ModuleType("rlinf.scheduler.cluster.cluster")
        cluster_module.ray = ray
        cluster_package = ModuleType("rlinf.scheduler.cluster")
        cluster_package.cluster = cluster_module
        self.ray = ray
        self.scheduler = ModuleType("rlinf.scheduler")
        self.scheduler.Worker = object
        self.start_managers = Mock()
        self.allocate = Mock()
        case = self

        class ClusterStub:
            NAMESPACE = "RLinf"
            NamespaceConflictError = type("NamespaceConflictError", (Exception,), {})
            has_initialized = Mock(return_value=False)
            allocate = self.allocate

            def __init__(self, **kwargs):
                self._distributed_log_collector = None
                try:
                    self._init_and_launch_managers(**kwargs)
                except self.NamespaceConflictError:
                    # Match the pinned scheduler's first retry into a shared namespace.
                    type(self).NAMESPACE = "RLinf_1"
                    self._init_and_launch_managers(**kwargs)

            def _init_and_launch_managers(self, **kwargs):
                try:
                    cluster_module.ray.init(address="auto", namespace=self.NAMESPACE)
                except ConnectionError:
                    cluster_module.ray.init(namespace=self.NAMESPACE)
                case.start_managers(self, **kwargs)

        def connect(*, address, **kwargs):
            ray.is_initialized.return_value = True
            self.node._global_node = SimpleNamespace(ray_init_cluster=(address == "local"))

        def disconnect():
            ray.is_initialized.return_value = False
            self.node._global_node = None

        self.scheduler.Cluster = ClusterStub
        ray.init = Mock(side_effect=connect)
        ray.shutdown.side_effect = disconnect
        for name in ("Channel", "NodePlacementStrategy", "PackedPlacementStrategy"):
            setattr(self.scheduler, name, Mock())
        modules = {"ray": ray, "rlinf": ModuleType("rlinf"), "rlinf.scheduler": self.scheduler}
        modules[ray_constants.__name__] = ray_constants
        for module in (usage_constants, private_constants, kv):
            modules[module.__name__] = module
        modules[cluster_package.__name__] = cluster_package
        modules[cluster_module.__name__] = cluster_module
        worker_module = ModuleType("rlinf.scheduler.channel.channel_worker")

        class ChannelWorkerStub:
            create_group = Mock()

            def __init__(self, channel):
                self.channel = channel

            async def get_via_ray(self, *, nowait):
                assert nowait
                return self.channel.get_nowait()

        worker_module.ChannelWorker = self.channel_worker = ChannelWorkerStub
        modules[worker_module.__name__] = worker_module
        stubbed = patch.dict(sys.modules, modules)
        stubbed.start()
        self.addCleanup(stubbed.stop)
        self.backend = importlib.import_module("dobot_nexus.runtime.ray.backend")
        self.worker_cls = importlib.import_module(
            "dobot_nexus.runtime.ray.channel"
        ).DeadlineChannelWorker
        self.channel_worker.create_group.return_value = self.channel_group(
            self.scheduler.Channel.connect.return_value
        )
        self.spec = RoleSpec(
            "probe", f"{__name__}:Probe", PlacementSpec("cpu"), init_kwargs={"token": "ok"}
        )

    def test_worker_visibility_rank_and_allowlist(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            worker = self.backend.ManagedRoleWorker(self.spec, "bus")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
            worker._rank, worker._world_size = 3, 4
            self.assertEqual(worker.initialize(), ("probe", 3, 4, "cpu", "bus"))
            self.assertEqual(worker.invoke("ping", "!"), "ok!")
            with self.assertRaisesRegex(ValueError, "not allowed"):
                worker.invoke("initialize")
            worker.close()
            self.assertTrue(worker.role.closed)

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            worker = self.backend.ManagedRoleWorker(
                replace(self.spec, placement=PlacementSpec("gpu", 7)), None
            )
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")
            with patch("torch.cuda.device_count", return_value=2), self.assertRaises(RuntimeError):
                worker.initialize()
            with (
                patch("torch.cuda.device_count", return_value=1),
                patch("torch.cuda.set_device") as use,
            ):
                self.assertEqual(worker.initialize(), ("probe", 0, 1, "cuda:0", None))
                use.assert_called_once_with(0)

    def make_group(self, result):
        group = Mock(worker_info_list=[SimpleNamespace(worker=object())])
        group.maxsize.return_value.wait.return_value = [1]
        group.initialize.return_value.done.return_value = True
        group.initialize.return_value.wait.return_value = result
        group.close.return_value.done.return_value = True
        group.close.return_value.wait.return_value = [None]
        return group

    def channel_group(self, channel):
        worker = self.worker_cls(channel)
        actor = Mock()
        actor.take_before.remote.side_effect = lambda deadline: (
            lambda: asyncio.run(worker.take_before(deadline))
        )
        group = self.make_group(None)
        group.worker_info_list = [SimpleNamespace(worker=actor)]
        return group

    def test_partial_launch_releases_tracked_workers_and_only_owned_channels(self):
        channel_group = self.make_group(None)
        self.channel_worker.create_group.return_value = channel_group
        role = self.make_group(["ok"])
        role.launch.side_effect = RuntimeError("partial launch")
        role.close.side_effect = AttributeError("no worker RPCs bound yet")
        with patch.object(
            self.backend.ManagedRoleWorker, "create_group", return_value=role, create=True
        ):
            topology = TopologySpec("demo", (self.spec,), (ChannelSpec("arbitrary"),))
            with self.assertRaisesRegex(RuntimeError, "partial launch"):
                self.backend.RLinfRayRuntime().start(topology)
        self.ray.kill.assert_called_once_with(role.worker_info_list[0].worker, no_restart=True)
        role._close.assert_called_once()
        channel_group._close.assert_called_once()
        self.assertIn("arbitrary", self.scheduler.Channel.connect.call_args.args[0])
        self.channel_worker.create_group.assert_called_once_with(maxsize=1)

        # A collision in launch must not connect to (or close) the existing actor.
        self.scheduler.Channel.connect.reset_mock()
        channel_group.reset_mock()
        channel_group.worker_info_list = []
        channel_group.launch.side_effect = ValueError("duplicate actor name")
        with patch.object(self.backend, "_driver_claim", threading.Lock()):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.backend.RLinfRayRuntime().start(topology)
        self.scheduler.Channel.connect.assert_not_called()
        channel_group._close.assert_called_once()

    def test_channel_backpressure_and_receive_deadline(self):
        bus = self.backend.RayChannelBus()
        bus.create(
            (ChannelSpec("custom", max_message_bytes=32),),
            "run",
            object(),
            [],
            deadline=time.monotonic() + 2,
        )
        channel = self.scheduler.Channel.connect.return_value
        with self.assertRaisesRegex(ValueError, "message exceeds"):
            bus.put_nowait("custom", bytes(32))
        channel.put_nowait.assert_not_called()
        channel.put_nowait.side_effect = asyncio.QueueFull()
        with self.assertRaises(ChannelFull):
            bus.put_nowait("custom", 7)
        channel.get_nowait.side_effect = asyncio.QueueEmpty()
        with self.assertRaises(ChannelEmpty):
            bus.try_get("custom")
        with self.assertRaises(TimeoutError):
            bus.recv("custom", timeout=0.01)
        channel.get_nowait.side_effect = None
        channel.get_nowait.return_value = "reply"
        self.assertEqual(bus.recv("custom", timeout=1), "reply")

    def test_all_roles_initialize_before_wait_and_rpcs_use_submission_deadlines(self):
        first = self.make_group(["cpu"])
        second = self.make_group(["gpu"])
        first.initialize.return_value.done.side_effect = lambda: second.initialize.called
        second.invoke.return_value.done.return_value = True
        second.invoke.return_value.wait.return_value = [42]
        gpu = replace(self.spec, name="other", placement=PlacementSpec("gpu", 2))
        with patch.object(
            self.backend.ManagedRoleWorker, "create_group", side_effect=[first, second], create=True
        ):
            session = self.backend.RLinfRayRuntime().start(
                TopologySpec("run", (self.spec, gpu), ())
            )
            self.addCleanup(session.close)
            self.assertEqual(session.init_results, {"probe": "cpu", "other": "gpu"})
            self.scheduler.NodePlacementStrategy.assert_called_once_with([0])
            self.scheduler.PackedPlacementStrategy.assert_called_once_with(2, 2)
            for group in (first, second):
                self.assertEqual(group.launch.call_args.kwargs["max_concurrency"], 1)
            slow = session.role("probe").call("ping", timeout=0.02)
            fast = session.role("other").call("ping", "!")
            self.assertEqual(fast.wait(), 42)
            second.invoke.assert_called_once_with("ping", "!")
            time.sleep(0.03)
            with self.assertRaisesRegex(TimeoutError, "slow"):
                session.wait_any({"fast": fast, "slow": slow})
            with self.assertRaises(TimeoutError):
                slow.wait(timeout=10)

    def test_malformed_initialize_and_unresponsive_group_close_are_bounded(self):
        role = self.make_group((1,))
        unblock = threading.Event()
        self.addCleanup(unblock.set)
        role._close.side_effect = lambda: unblock.wait(5)
        with patch.object(
            self.backend.ManagedRoleWorker, "create_group", return_value=role, create=True
        ):
            topology = TopologySpec("demo", (self.spec,), (), rpc_timeout=0.2)
            start = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                self.backend.RLinfRayRuntime().start(topology)
            self.assertLess(time.monotonic() - start, 1)
        self.ray.kill.assert_called_once_with(role.worker_info_list[0].worker, no_restart=True)

    def test_session_retains_unique_tmpdirs_and_logs_on_every_shutdown(self):
        logger = Mock()
        directories = set()
        process = Mock()
        process.poll.return_value = 0
        collector = Mock()
        connect = self.ray.init.side_effect

        def init(*, address, **kwargs):
            if owned and address == "auto":
                raise ConnectionError("no external head")
            connect(address=address, **kwargs)

        def start(instance, **kwargs):
            directory = Path(os.environ["RAY_TMPDIR"])
            self.assertEqual(directory.parent, Path(parent))
            self.assertTrue(directory.name.startswith("r-"))
            self.assertNotIn(directory, directories)
            directories.add(directory)
            session_dir = directory / "ray/session"
            logs = session_dir / "logs"
            logs.mkdir(parents=True)
            (logs / "worker.log").write_text("source log")
            instance._distributed_log_collector = collector
            self.node._global_node.all_processes = {"raylet": [SimpleNamespace(process=process)]}
            self.node._global_node.get_session_dir_path = lambda: str(session_dir)

        self.ray.init.side_effect = init
        self.start_managers.side_effect = start
        with tempfile.TemporaryDirectory(dir="/tmp") as parent:
            topology = TopologySpec("demo", (), (), ray_tmpdir=parent)
            for owned, failure in ((False, False), (True, False), (True, True)):
                with (
                    self.subTest(owned=owned, failure=failure),
                    patch.object(self.backend, "_driver_claim", threading.Lock()),
                    patch.dict(os.environ, {"RAY_TMPDIR": "existing"}),
                ):
                    collector.stop.side_effect = OSError("flush failed") if failure else None
                    session = self.backend.RLinfRayRuntime(logger=logger).start(topology)
                    self.assertEqual(os.environ["RAY_TMPDIR"], "existing")
                    session.close()
                    session.close()
                    logger.info.assert_called_with(
                        "Retained Ray session directory for diagnostics: %s", session._tmpdir
                    )
                    if failure:
                        self.assertIn("RLinf log collector flush", str(logger.warning.call_args))
                        self.assertIn("flush failed", str(logger.warning.call_args))
                    self.ray.shutdown.assert_called_once_with()
                    self.ray.shutdown.reset_mock()
            self.assertEqual(len(directories), 3)
            for directory in directories:
                self.assertEqual(
                    (directory / "ray/session/logs/worker.log").read_text(), "source log"
                )

    def test_socket_budget_rejected_before_creating_tmpdir(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as parent:
            oversized = Path(parent) / ("é" * 30)
            with self.assertRaisesRegex(ValueError, "AF_UNIX"):
                self.backend.RLinfRayRuntime().start(
                    TopologySpec("demo", (), (), ray_tmpdir=str(oversized))
                )
            self.assertFalse(oversized.exists())
            self.ray.init.assert_not_called()

    def test_preinitialized_driver_is_rejected_without_shutdown(self):
        self.ray.is_initialized.return_value = True
        with self.assertRaisesRegex(RuntimeError, "exclusive RLinf namespace"):
            self.backend.RLinfRayRuntime().start(TopologySpec("demo", (), ()))
        self.start_managers.assert_not_called()
        self.ray.shutdown.assert_not_called()

    def test_delayed_channel_registration_precedes_connect(self):
        group = self.channel_worker.create_group.return_value
        waiting, registered = threading.Event(), threading.Event()

        def capacity():
            waiting.set()
            self.assertTrue(registered.wait(2))
            return [1]

        group.maxsize.return_value.wait.side_effect = capacity
        bus = self.backend.RayChannelBus()
        with ThreadPoolExecutor(1) as pool:
            startup = pool.submit(
                bus.create,
                (ChannelSpec("lane"),),
                "private",
                object(),
                [],
                deadline=time.monotonic() + 2,
            )
            try:
                self.assertTrue(waiting.wait(1))
                self.scheduler.Channel.connect.assert_not_called()
            finally:
                registered.set()
            startup.result(timeout=1)
        self.scheduler.Channel.connect.assert_called_once_with("private-lane", current_worker=None)

        registered.clear()
        self.scheduler.Channel.connect.reset_mock()
        try:
            with self.assertRaisesRegex(TimeoutError, "startup"):
                bus.create(
                    (ChannelSpec("other"),),
                    "private",
                    object(),
                    [],
                    deadline=time.monotonic() + 0.03,
                )
            self.scheduler.Channel.connect.assert_not_called()
        finally:
            registered.set()

    def test_all_channel_groups_launch_before_readiness_and_connect(self):
        specs = [ChannelSpec(f"lane{i}") for i in range(6)]
        groups = [self.make_group(None) for _ in specs]
        tracked, events = [], []
        for i, group in enumerate(groups):

            def launch(*args, name, owner=group, **kwargs):
                self.assertIs(tracked[-1], owner)
                events.append(("launch", name.removeprefix("test-")))

            def ready(name=specs[i].name):
                events.append(("ready", name))
                return [1]

            group.launch.side_effect = launch
            group.maxsize.return_value.wait.side_effect = ready
        self.channel_worker.create_group.side_effect = groups
        self.scheduler.Channel.connect.side_effect = lambda name, **kw: events.append(
            ("connect", name.removeprefix("test-"))
        )
        bus = self.backend.RayChannelBus()
        bus.create(specs, "test", object(), tracked, deadline=time.monotonic() + 2)
        self.assertEqual(tracked, groups)
        self.assertEqual(
            events, [(phase, s.name) for phase in ("launch", "ready", "connect") for s in specs]
        )

    def test_receive_deadline_bounds_remote_get_and_rejects_late_ack(self):
        bus = self.backend.RayChannelBus()
        bus.create((ChannelSpec("acks"),), "test", object(), [], deadline=time.monotonic() + 1)
        channel = self.scheduler.Channel.connect.return_value
        returned = threading.Event()

        def late_ack():
            time.sleep(0.06)
            returned.set()
            return "ACK"

        channel.get_nowait.side_effect = late_ack
        with self.assertRaisesRegex(TimeoutError, "receive deadline"):
            bus.recv("acks", timeout=0.010)
        self.assertFalse(returned.is_set())
        self.assertTrue(returned.wait(1))
        channel.get_nowait.assert_called_once()

    def test_indeterminate_receive_permanently_poisons_only_its_lane(self):
        started, release, finished = (threading.Event() for _ in range(3))
        transport, healthy = Mock(), Mock()

        def stuck():
            started.set()
            release.wait(3)
            finished.set()
            return "late"

        transport.get_nowait.side_effect = stuck
        healthy.get_nowait.return_value = "healthy"
        self.channel_worker.create_group.side_effect = [
            self.channel_group(transport),
            self.channel_group(healthy),
        ]
        self.scheduler.Channel.connect.side_effect = [transport, healthy]
        bus = self.backend.RayChannelBus()
        bus.create(
            [ChannelSpec("acks"), ChannelSpec("other")],
            "run",
            object(),
            [],
            deadline=time.monotonic() + 2,
        )
        with ThreadPoolExecutor(1) as pool:
            receiving = pool.submit(bus.recv, "acks", timeout=0.1)
            try:
                self.assertTrue(started.wait(1))
                with self.assertRaises(TimeoutError):
                    receiving.result(timeout=1)
                message = "Channel 'acks' is unusable after an indeterminate receive timeout."
                with self.assertRaisesRegex(RuntimeError, message):
                    bus.recv("acks", timeout=2)
                with self.assertRaisesRegex(RuntimeError, message):
                    bus.try_get("acks")
                self.assertEqual(bus.recv("other", timeout=1), "healthy")
            finally:
                release.set()
        self.assertTrue(finished.wait(1))
        with self.assertRaisesRegex(RuntimeError, message):
            bus.try_get("acks")
        transport.get_nowait.assert_called_once()

    def test_ephemeral_head_is_marked_at_start_and_rejects_a_second_driver(self):
        head = {}
        owner = SimpleNamespace(ray_init_cluster=True, all_processes={}, kill_all_processes=Mock())
        attached = SimpleNamespace(ray_init_cluster=False, kill_all_processes=Mock())

        def init(*, address, **kwargs):
            if address == "auto" and not head:
                raise ConnectionError("no head")
            if address == "local":
                head.update({"node:__internal_head__": 1, **kwargs["resources"]})
                self.node._global_node = owner
            else:
                self.node._global_node = attached
            self.ray.is_initialized.return_value = True

        def disconnect():
            if self.node._global_node is owner:
                head.clear()
            self.node._global_node = None
            self.ray.is_initialized.return_value = False

        self.ray.init.side_effect = init
        self.ray.nodes.side_effect = lambda: [{"Alive": True, "Resources": dict(head)}]
        self.ray.shutdown.side_effect = disconnect
        topology = TopologySpec("test", (), ())
        first = self.backend.RLinfRayRuntime().start(topology)
        self.assertIn(self.backend._EPHEMERAL_HEAD_SENTINEL, head)
        # Model a fresh driver's process-local state on the very same head.
        self.node._global_node = None
        self.ray.is_initialized.return_value = False
        with patch.object(self.backend, "_driver_claim", threading.Lock()):
            with self.assertRaisesRegex(RuntimeError, "another session's ephemeral Ray head"):
                self.backend.RLinfRayRuntime().start(topology)
        self.ray.shutdown.assert_called_once()
        self.assertIn(self.backend._EPHEMERAL_HEAD_SENTINEL, head)
        self.start_managers.assert_called_once()
        self.channel_worker.create_group.assert_not_called()
        owner.kill_all_processes.assert_not_called()
        attached.kill_all_processes.assert_not_called()
        self.ray.kill.assert_not_called()
        self.node._global_node = owner
        self.ray.is_initialized.return_value = True
        first.close()
        self.assertFalse(head)

        head["node:__internal_head__"] = 1
        with patch.object(self.backend, "_driver_claim", threading.Lock()):
            external = self.backend.RLinfRayRuntime().start(topology)
            external.close()
        self.assertEqual(head, {"node:__internal_head__": 1})
        attached.kill_all_processes.assert_not_called()

    def test_explicit_ray_address_propagates_original_error_without_local_fallback(self):
        address = "ray-head.example:6379"
        os.environ["RAY_ADDRESS"] = address
        error = ConnectionError("configured head unavailable")
        cause = error.__cause__ = OSError("connection refused")
        self.ray.init.side_effect = error
        with self.assertRaises(ConnectionError) as caught:
            self.backend.RLinfRayRuntime().start(TopologySpec("test", (), ()))
        self.assertIs(caught.exception, error)
        self.assertIs(caught.exception.__cause__, cause)
        self.ray.init.assert_called_once()
        self.assertEqual(self.ray.init.call_args.kwargs["address"], address)
        self.ray.shutdown.assert_not_called()
        self.start_managers.assert_not_called()

    def test_unmarked_head_requires_explicit_false_in_cluster_metadata(self):
        self.check_head_ownership()

    def test_ray_client_without_local_node_checks_remote_head_ownership(self):
        address = "ray://ray-head.example:10001"
        os.environ["RAY_ADDRESS"] = address

        def connect(**kwargs):
            self.assertEqual(kwargs["address"], address)
            self.assertIsNone(self.node._global_node)
            self.ray.is_initialized.return_value = True

        self.ray.init.side_effect = connect
        self.check_head_ownership()

    def check_head_ownership(self):
        topology = TopologySpec("test", (), ())
        for metadata in (b'{"ray_init_cluster": true}', None, b"{}", b'{"ray_init_cluster": 0}'):
            with (
                self.subTest(metadata=metadata),
                patch.object(self.backend, "_driver_claim", threading.Lock()),
            ):
                self.kv_get.return_value = metadata
                with self.assertRaisesRegex(RuntimeError, "confirmed externally managed"):
                    self.backend.RLinfRayRuntime().start(topology)
                self.kv_get.assert_called_once_with(b"CLUSTER_METADATA", namespace=b"cluster")
                self.start_managers.assert_not_called()
                self.ray.shutdown.assert_called_once()
                self.ray.kill.assert_not_called()
                self.kv_get.reset_mock()
                self.ray.shutdown.reset_mock()
        self.kv_get.return_value = b'{"ray_init_cluster": false}'
        session = self.backend.RLinfRayRuntime().start(topology)
        self.kv_get.assert_called_once_with(b"CLUSTER_METADATA", namespace=b"cluster")
        self.start_managers.assert_called_once()
        session.close()
        self.ray.shutdown.assert_called_once()

    def test_metadata_read_error_disconnects_without_reentering_discovery(self):
        self.kv_get.side_effect = ConnectionError("lost GCS connection")
        with self.assertRaisesRegex(RuntimeError, "verify ownership") as caught:
            self.backend.RLinfRayRuntime().start(TopologySpec("test", (), ()))
        self.assertIsInstance(caught.exception.__cause__, ConnectionError)
        self.start_managers.assert_not_called()
        self.ray.init.assert_called_once()
        self.ray.shutdown.assert_called_once()

    def test_expired_worker_read_cannot_dequeue_a_second_clients_new_ack(self):
        messages = queue.Queue(maxsize=1)
        channel = self.scheduler.Channel.connect.return_value
        channel.put_nowait.side_effect = messages.put_nowait
        channel.get_nowait.side_effect = messages.get_nowait
        consumer = self.backend.RayChannelBus()
        consumer.create([ChannelSpec("acks")], "run", object(), [], deadline=time.monotonic() + 1)
        actor = consumer._readers["acks"]
        # Serialized clients have independent state and handles to the same actor.
        producer = copy.deepcopy(consumer, {id(actor): actor, id(channel): channel})
        submitted, release, processed = (threading.Event() for _ in range(3))

        def delayed_worker_execution(ref, *, timeout):
            submitted.set()
            release.wait(3)
            try:
                return ref()
            finally:
                processed.set()

        self.ray.get.side_effect = delayed_worker_execution
        with ThreadPoolExecutor(1) as pool:
            receive = pool.submit(consumer.recv, "acks", timeout=0.1)
            try:
                self.assertTrue(submitted.wait(1))
                with self.assertRaisesRegex(TimeoutError, "receive deadline"):
                    receive.result(timeout=1)
                producer.put_nowait("acks", "next ACK")
            finally:
                release.set()
        self.assertTrue(processed.wait(1))
        channel.get_nowait.assert_not_called()
        self.assertEqual(producer.try_get("acks"), "next ACK")

    def test_inherited_namespace_cannot_redirect_channel_manager(self):
        namespaces = []

        def connect(name, **kwargs):
            # Pinned WorkerManager.get_proxy reads this again on Channel.connect.
            self.scheduler.Cluster.NAMESPACE = os.environ["CLUSTER_NAMESPACE"]
            namespaces.append(self.scheduler.Cluster.NAMESPACE)
            return Mock()

        os.environ["CLUSTER_NAMESPACE"] = "another-job"
        self.scheduler.Channel.connect.side_effect = connect
        topology = TopologySpec("test", (), (ChannelSpec("lane"),))
        session = self.backend.RLinfRayRuntime().start(topology)
        self.addCleanup(session.close)
        namespace = self.ray.init.call_args.kwargs["namespace"]
        self.assertNotEqual(namespace, "another-job")
        self.assertEqual(namespaces, [namespace])
        self.assertEqual(self.scheduler.Cluster.NAMESPACE, namespace)

    def test_auto_attach_uses_session_namespace_and_rejects_conflict_retry(self):
        actors = {"RLinf": {"other-job"}}
        seen = []

        def attach(instance, **kwargs):
            namespace = self.scheduler.Cluster.NAMESPACE
            seen.append(namespace)
            self.assertNotIn(namespace, actors)
            actors[namespace] = {"this-job"}

        self.start_managers.side_effect = attach
        topology = TopologySpec("test", (), ())
        session = self.backend.RLinfRayRuntime().start(topology)
        session.close()
        self.assertEqual(actors["RLinf"], {"other-job"})
        # The installed signal handler resolves NAMESPACE even after close.
        self.assertEqual(actors[self.scheduler.Cluster.NAMESPACE], {"this-job"})

        self.start_managers.side_effect = self.scheduler.Cluster.NamespaceConflictError()
        with patch.object(self.backend, "_driver_claim", threading.Lock()):
            with self.assertRaisesRegex(RuntimeError, "claim its exclusive namespace"):
                self.backend.RLinfRayRuntime().start(topology)
        self.assertNotIn(self.scheduler.Cluster.NAMESPACE, seen + ["RLinf", "RLinf_1"])

    def test_cpu_actor_launch_overrides_gpu_env_before_remote_construction(self):
        cpu, gpu = self.make_group([None]), self.make_group([None])

        def launch(cluster, **kwargs):
            cluster.allocate(
                cls=self.backend.ManagedRoleWorker,
                env_vars={"CUDA_VISIBLE_DEVICES": "7", "GROUP_NAME": kwargs["name"]},
            )

        cpu.launch.side_effect = gpu.launch.side_effect = launch
        roles = (self.spec, replace(self.spec, name="gpu", placement=PlacementSpec("gpu", 7)))
        with patch.object(
            self.backend.ManagedRoleWorker, "create_group", side_effect=[cpu, gpu], create=True
        ):
            session = self.backend.RLinfRayRuntime().start(TopologySpec("test", roles, ()))
            self.addCleanup(session.close)
        cpu_env, gpu_env = [call.kwargs["env_vars"] for call in self.allocate.call_args_list]
        self.assertEqual(cpu_env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(cpu_env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(cpu_env["LOCAL_ACCELERATOR_RANK"], "-1")
        self.assertTrue(cpu_env["GROUP_NAME"].endswith("-probe"))
        self.assertEqual(gpu_env["CUDA_VISIBLE_DEVICES"], "7")

    def test_attach_flushes_logs_before_disconnect_including_partial_start(self):
        events = []

        def flush(*, join_timeout_s):
            self.assertGreater(join_timeout_s, 0)
            self.assertLessEqual(join_timeout_s, 2)
            events.append("flushed")

        disconnect = self.ray.shutdown.side_effect

        def shutdown():
            events.append("disconnected")
            disconnect()

        def attach(instance, **kwargs):
            instance._distributed_log_collector = SimpleNamespace(stop=flush)
            if fail:
                raise ValueError("partial manager launch")

        self.ray.shutdown.side_effect = shutdown
        self.start_managers.side_effect = attach
        topology = TopologySpec("test", (), (), rpc_timeout=2)
        for fail in (False, True):
            with (
                self.subTest(fail=fail),
                patch.object(self.backend, "_driver_claim", threading.Lock()),
            ):
                events.clear()
                if fail:
                    with self.assertRaisesRegex(ValueError, "partial manager"):
                        self.backend.RLinfRayRuntime().start(topology)
                else:
                    session = self.backend.RLinfRayRuntime().start(topology)
                    session.close()
                    session.close()
                self.assertEqual(events, ["flushed", "disconnected"])

    def test_hung_collector_cannot_exhaust_driver_disconnect_budget(self):
        started, release, disconnected = (threading.Event() for _ in range(3))

        def flush(*, join_timeout_s):
            started.set()
            release.wait(3)

        def attach(instance, **kwargs):
            instance._distributed_log_collector = SimpleNamespace(stop=flush)

        self.start_managers.side_effect = attach
        self.ray.shutdown.side_effect = disconnected.set
        session = self.backend.RLinfRayRuntime().start(
            TopologySpec("test", (), (), rpc_timeout=0.1)
        )
        with ThreadPoolExecutor(1) as pool:
            closing = pool.submit(session.close)
            try:
                self.assertTrue(started.wait(1))
                closing.result(timeout=1)
                self.assertTrue(disconnected.is_set())
            finally:
                release.set()
        self.assertTrue(disconnected.wait(1))

    def test_collector_exception_still_disconnects_ray(self):
        stop = Mock(side_effect=OSError("flush failed"))

        def attach(instance, **kwargs):
            instance._distributed_log_collector = SimpleNamespace(stop=stop)

        self.start_managers.side_effect = attach
        session = self.backend.RLinfRayRuntime().start(TopologySpec("test", (), (), rpc_timeout=1))
        session.close()
        stop.assert_called_once()
        self.assertLess(stop.call_args.kwargs["join_timeout_s"], 0.5)
        self.ray.shutdown.assert_called_once()
        self.assertFalse(self.ray.is_initialized())

    def test_unregistered_local_head_shutdown_falls_back_to_owned_processes(self):
        process = Mock()
        process.poll.return_value = None
        node = SimpleNamespace(
            ray_init_cluster=True,
            all_processes={"gcs": [SimpleNamespace(process=process)]},
            kill_all_processes=Mock(),
        )

        def init(*, address, **kwargs):
            if address == "auto":
                raise ConnectionError("no existing head")
            self.node._global_node = node
            raise ValueError("driver registration failed")

        def kill(**kwargs):
            self.ray.shutdown.assert_called_once()
            process.poll.return_value = 0

        self.ray.init.side_effect = init
        node.kill_all_processes.side_effect = kill
        with self.assertRaisesRegex(ValueError, "registration failed"):
            self.backend.RLinfRayRuntime().start(TopologySpec("test", (), ()))
        self.assertFalse(self.ray.is_initialized())
        node.kill_all_processes.assert_called_once_with(check_alive=False, wait=True)
        self.assertEqual(process.poll(), 0)
