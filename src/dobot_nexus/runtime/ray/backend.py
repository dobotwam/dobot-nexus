"""One dedicated RLinf driver, one single-node session."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

import ray
from ray._common.constants import HEAD_NODE_RESOURCE_NAME
from ray._common.usage.usage_constants import CLUSTER_METADATA_KEY
from ray._private.ray_constants import KV_NAMESPACE_CLUSTER
from ray.experimental.internal_kv import _internal_kv_get
from rlinf.scheduler import Cluster, NodePlacementStrategy, PackedPlacementStrategy
from rlinf.scheduler.cluster import cluster as cluster_module

from ..config import validate_ray_tmpdir
from ..contracts import rpc_deadline, wait_any
from ._deadline import call_until, submit_daemon
from .channel import RayChannelBus
from .worker import ManagedRoleWorker

SUPPORTED_REVISION = "7e90641c4135229db6d3426950cdb45cee6259a6"
# RLinf Cluster is a process singleton, including its signal handler and managers.
# A fresh production session needs a fresh driver even after orderly shutdown.
_driver_claim = threading.Lock()
_EPHEMERAL_HEAD_SENTINEL = "dobot_nexus_ephemeral_head"


class _RayFuture:
    def __init__(self, result, deadline):
        self.result = result
        self.deadline = deadline

    def done(self):
        return self.result.done()

    def wait(self, timeout=None):
        deadline = self.deadline
        if timeout is not None:
            deadline = min(deadline, time.monotonic() + timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RLinf worker RPC deadline exceeded.")
            if self.done():
                values = self.result.wait()
                if not isinstance(values, list) or len(values) != 1:
                    raise RuntimeError("Expected exactly one worker result per role.")
                return values[0]
            time.sleep(min(0.01, remaining))


class _RayRole:
    def __init__(self, group, session):
        self.group, self.session = group, session
        self.lock = threading.Lock()

    def call(self, method, *args, timeout=None, **kwargs):
        deadline = rpc_deadline(self.session.timeout if timeout is None else timeout)
        with self.lock:
            if self.session.closed:
                raise RuntimeError("Session closed.")
            result = self.group.invoke(method, *args, **kwargs)
        return _RayFuture(result, deadline)

    def close(self):
        with self.lock:
            return self.group.close()


def _bounded(action, deadline, logger, operation="Ray teardown operation"):
    """RLinf's private teardown and even a Ray kill can block inside Ray."""
    try:
        finished = submit_daemon(action)
        finished.result(timeout=max(0, deadline - time.monotonic()))
        return True
    except Exception as exc:
        logger.warning("%s failed or timed out: %r", operation, exc)
        return False


def _release_group(group, deadline, logger, *, close_rpc):
    # Reserve time for forced teardown even if graceful close consumes its allowance.
    def halfway():
        return (time.monotonic() + deadline) / 2

    workers = tuple(info.worker for info in group.worker_info_list)

    def kill_workers():
        for worker in workers:
            _bounded(lambda w=worker: ray.kill(w, no_restart=True), halfway(), logger)

    if close_rpc is not None:
        until = halfway()
        if not _bounded(lambda: _RayFuture(close_rpc(), until).wait(), until, logger):
            kill_workers()
    if not _bounded(group._close, halfway(), logger):
        kill_workers()


def _driver_node():
    # Pinned Ray exposes local-head ownership on the driver Node. An attached
    # driver can be initialized and have a session directory on this same host.
    return getattr(ray._private.worker, "_global_node", None)


class _ExplicitAddressFailure(Exception):
    """Carry the original exception past RLinf's automatic discovery fallback."""


class _RayHeadGuard:
    """Guard the pinned Cluster module's auto-attach and local-start init calls."""

    def __init__(self, timeout):
        self.timeout = timeout

    def __getattr__(self, name):
        return getattr(ray, name)

    def init(self, *, address=None, **kwargs):
        configured = os.environ.get("RAY_ADDRESS") or "auto"
        if configured != "auto":
            address = configured
        if address in (None, "local"):
            # The sentinel is advertised in the head's initial node record. An
            # attaching job can see it even before this driver has registered.
            address = "local"
            kwargs["resources"] = {**kwargs.get("resources", {}), _EPHEMERAL_HEAD_SENTINEL: 1}
        try:
            context = ray.init(address=address, **kwargs)
        except ConnectionError as exc:
            if configured != "auto":
                raise _ExplicitAddressFailure from exc
            raise
        if address == "local":
            return context

        def head_info():
            record = _internal_kv_get(CLUSTER_METADATA_KEY, namespace=KV_NAMESPACE_CLUSTER)
            return ray.nodes(), json.loads(record) if record is not None else None

        try:
            nodes, metadata = call_until(
                head_info, rpc_deadline(self.timeout), "Ray head lookup timed out."
            )
        except Exception as exc:
            # A ConnectionError here must not trigger RLinf's local-start fallback.
            raise RuntimeError("Could not verify ownership of the attached Ray head.") from exc
        heads = [
            node["Resources"]
            for node in nodes
            if node["Alive"] and HEAD_NODE_RESOURCE_NAME in node["Resources"]
        ]
        if len(heads) != 1:
            raise RuntimeError("Could not identify a single live Ray head.")
        if _EPHEMERAL_HEAD_SENTINEL in heads[0]:
            raise RuntimeError("Cannot attach to another session's ephemeral Ray head.")
        if not isinstance(metadata, dict) or metadata.get("ray_init_cluster") is not False:
            raise RuntimeError("Attachment requires a confirmed externally managed Ray head.")
        return context


class _CpuRoleCluster:
    """Override only the actor env passed to the pinned Cluster.allocate."""

    def __init__(self, cluster):
        self.cluster = cluster

    def __getattr__(self, name):
        return getattr(self.cluster, name)

    def allocate(self, *, env_vars, **kwargs):
        # WorkerGroup.launch has no env override. Its per-worker allocate
        # kwargs take precedence over node config in Ray's actor runtime_env.
        env_vars = {
            **env_vars,
            "CUDA_VISIBLE_DEVICES": "",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "VISIBLE_DEVICES": "",
            "LOCAL_ACCELERATOR_RANK": "-1",
        }
        return self.cluster.allocate(env_vars=env_vars, **kwargs)


class RLinfRaySession:
    wait_any = staticmethod(wait_any)

    def __init__(self, topology, logger):
        self.timeout = topology.rpc_timeout
        self.logger = logger
        self.init_results = {}
        self._roles = {}
        self._channel_groups = []
        self.cluster_attempted = False
        self._rlinf_cluster = None
        self._opened_ray_connection = False
        self._connection_node = None
        self._owned_node = None
        self._tmpdir = None
        self._close_lock = threading.Lock()
        self.closed = False

    def role(self, name):
        return self._roles[name]

    def _cluster(self, topology):
        initialized = ray.is_initialized()
        if initialized or Cluster.has_initialized():
            raise RuntimeError("An exclusive RLinf namespace requires a fresh driver connection.")
        previous_node = _driver_node()
        previous = os.environ.get("RAY_TMPDIR")
        if topology.ray_tmpdir is not None:
            root = Path(topology.ray_tmpdir).expanduser().absolute()
            # The session suffix has eight characters. Check the budget before mkdir.
            validate_ray_tmpdir(root / "r-00000000")
            root.mkdir(parents=True, exist_ok=True)
            self._tmpdir = Path(tempfile.mkdtemp(prefix="r-", dir=root))
            validate_ray_tmpdir(self._tmpdir)
            os.environ["RAY_TMPDIR"] = str(self._tmpdir)
        try:
            self.cluster_attempted = True
            # The SIGUSR1 handler reads this class attribute at signal time.
            # Keep the private namespace for the lifetime of the dedicated driver.
            Cluster.NAMESPACE = os.environ["CLUSTER_NAMESPACE"] = f"dobot-nexus-{uuid.uuid4().hex}"
            cluster = self._rlinf_cluster = Cluster.__new__(Cluster)
            launch_managers = cluster._init_and_launch_managers

            def launch_exclusively(*args, **kwargs):
                try:
                    return launch_managers(*args, **kwargs)
                except Cluster.NamespaceConflictError as exc:
                    # The upstream retry would switch to a shared RLinf_N namespace.
                    raise RuntimeError("RLinf could not claim its exclusive namespace.") from exc

            cluster._init_and_launch_managers = launch_exclusively
            # Only this module's view of Ray changes, while the dedicated driver
            # constructs its singleton. The check runs before any managers launch.
            original_ray = cluster_module.ray
            cluster_module.ray = _RayHeadGuard(self.timeout)
            try:
                Cluster.__init__(cluster, num_nodes=1, distributed_log_dir=topology.log_dir)
                return cluster
            except _ExplicitAddressFailure as exc:
                connection_error = exc.__cause__
            finally:
                cluster_module.ray = original_ray
                del cluster._init_and_launch_managers
            raise connection_error
        finally:
            if self._tmpdir is not None:
                if previous is None:
                    os.environ.pop("RAY_TMPDIR", None)
                else:
                    os.environ["RAY_TMPDIR"] = previous
            self._opened_ray_connection = not initialized and ray.is_initialized()
            node = self._connection_node = _driver_node()
            if node is not previous_node and getattr(node, "ray_init_cluster", False) is True:
                self._owned_node = node

    def launch(self, topology):
        cluster = self._cluster(topology)
        prefix = f"{topology.run_prefix}-{uuid.uuid4().hex}"
        bus = RayChannelBus()
        bus.create(
            topology.channels,
            prefix,
            cluster,
            self._channel_groups,
            deadline=rpc_deadline(self.timeout),
        )
        for spec in topology.roles:
            gpu = spec.placement.gpu_id
            placement = (
                NodePlacementStrategy([0]) if gpu is None else PackedPlacementStrategy(gpu, gpu)
            )
            group = ManagedRoleWorker.create_group(spec, bus)
            # launch mutates this group incrementally; keep it even on partial failure.
            self._roles[spec.name] = _RayRole(group, self)
            group.launch(
                _CpuRoleCluster(cluster) if gpu is None else cluster,
                placement_strategy=placement,
                name=f"{prefix}-{spec.name}",
                max_concurrency=spec.max_concurrency,
            )
        pending = {}
        for name, role in self._roles.items():
            deadline = rpc_deadline(self.timeout)
            pending[name] = _RayFuture(role.group.initialize(), deadline)
        while pending:
            for name, result in self.wait_any(pending):
                self.init_results[name] = result
                del pending[name]

    def _shutdown_ray(self, deadline):
        collector = getattr(self._rlinf_cluster, "_distributed_log_collector", None)
        if collector is not None:
            flush_until = time.monotonic() + max(0, deadline - time.monotonic()) / 3
            # A failed or hung final drain must leave time to disconnect Ray.
            _bounded(
                lambda: collector.stop(join_timeout_s=max(0, flush_until - time.monotonic())),
                flush_until,
                self.logger,
                "RLinf log collector flush",
            )

        node = self._owned_node
        if not self._opened_ray_connection and node is None:
            return
        # Ray mutates all_processes during shutdown. Keep the original handles
        # to verify exit even if driver registration failed after starting a head.
        processes = []
        if node is not None:
            try:
                processes = [p.process for group in node.all_processes.values() for p in group]
            except Exception:
                self.logger.exception("Reading owned Ray head processes")

        shutdown_ok = False
        current = _driver_node()
        if current is None or current is self._connection_node:
            until = deadline if node is None else (deadline + time.monotonic()) / 2
            shutdown_ok = _bounded(ray.shutdown, until, self.logger)
        if node is None:
            return
        if not shutdown_ok or not processes or any(p.poll() is None for p in processes):
            _bounded(
                lambda: node.kill_all_processes(check_alive=False, wait=True),
                deadline,
                self.logger,
            )

    def close(self):
        with self._close_lock:
            if self.closed:
                return
            self.closed = True
        deadline = rpc_deadline(self.timeout)
        groups = [(role.group, role.close) for role in reversed(self._roles.values())]
        groups.extend((group, None) for group in reversed(self._channel_groups))
        for index, (group, close_rpc) in enumerate(groups):
            now = time.monotonic()
            # All groups and the Ray shutdown share one absolute deadline.
            until = now + max(0, deadline - now) / (len(groups) - index + 1)
            try:
                _release_group(group, until, self.logger, close_rpc=close_rpc)
            except Exception:
                self.logger.exception("Releasing an RLinf worker group")
        try:
            self._shutdown_ray(deadline)
        except Exception:
            self.logger.exception("Shutting down the Ray runtime")
        finally:
            if self._tmpdir is not None:
                self.logger.info("Retained Ray session directory for diagnostics: %s", self._tmpdir)


class RLinfRayRuntime:
    def __init__(self, config=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def start(self, topology):
        session = RLinfRaySession(topology, self.logger)
        if not _driver_claim.acquire(blocking=False):
            raise RuntimeError("A dedicated RLinf driver supports only one session.")
        try:
            session.launch(topology)
        except BaseException:
            try:
                session.close()
            except Exception:
                self.logger.exception("Rolling back partial RLinf startup")
            if not session.cluster_attempted:
                _driver_claim.release()
            raise
        return session
