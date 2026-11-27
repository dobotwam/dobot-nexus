"""In-process test runtime. Each role owns one thread for its entire lifetime."""

from __future__ import annotations

import copy
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from dobot_nexus.core.import_utils import import_from_string

from ..contracts import RoleContext, invoke_role, rpc_deadline, wait_any
from .channel import LocalChannelBus

_LOCAL_INIT_LOCK = threading.Lock()


class _LocalFuture:
    def __init__(self, future, deadline):
        self._future = future
        self.deadline = deadline

    def done(self):
        return self._future.done()

    def wait(self, timeout=None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Local RPC deadline exceeded.")
        try:
            return self._future.result(
                timeout=remaining if timeout is None else min(timeout, remaining)
            )
        except FutureTimeoutError as exc:
            if self._future.done():
                raise
            raise TimeoutError("Local RPC deadline exceeded.") from exc


class _LocalRole:
    def __init__(self, spec, session):
        self.spec, self.session = spec, session
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=spec.name)
        self.instance = None
        self.thread = None

    def initialize(self):
        self.thread = threading.current_thread()
        # Construction and checkpoint restore can both touch process-wide Torch RNG.
        with _LOCAL_INIT_LOCK:
            if self.session.closed:
                raise RuntimeError("Session closed.")
            target = import_from_string(self.spec.target)
            self.instance = target(**copy.deepcopy(self.spec.init_kwargs))
            ctx = RoleContext(self.spec.name, 0, 1, "cpu", self.session.channels)
            return self.instance.initialize(ctx)

    def call(self, method, *args, timeout=None, **kwargs):
        deadline = rpc_deadline(self.session.timeout if timeout is None else timeout)
        with self.session.lock:
            if self.session.closed:
                raise RuntimeError("Session closed.")
            future = self.executor.submit(invoke_role, self.instance, method, args, kwargs)
        return _LocalFuture(future, deadline)

    def close(self):
        if self.instance is not None:
            self.instance.close()


class LocalSession:
    wait_any = staticmethod(wait_any)

    def __init__(self, topology, logger):
        self.channels = LocalChannelBus(topology.channels)
        self.timeout = topology.rpc_timeout
        self.logger = logger
        self.lock = threading.Lock()
        self.closed = False
        self._roles = {}
        self.init_results = {}

    def role(self, name):
        return self._roles[name]

    def start_role(self, spec):
        role = self._roles[spec.name] = _LocalRole(spec, self)
        deadline = rpc_deadline(self.timeout)
        self.init_results[spec.name] = _LocalFuture(
            role.executor.submit(role.initialize), deadline
        ).wait()

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.channels.close()
        deadline = rpc_deadline(self.timeout)
        closing = []
        for role in reversed(self._roles.values()):
            closing.append((role, role.executor.submit(role.close)))
            role.executor.shutdown(wait=False)

        for role, closed in closing:
            try:
                closed.result(timeout=max(0, deadline - time.monotonic()))
            except Exception:
                self.logger.exception("Closing local role %s", role.spec.name)
            finally:
                # A stuck external operation can outlive the bounded join.
                if role.thread is not None:
                    role.thread.join(timeout=max(0, deadline - time.monotonic()))


class LocalRuntime:
    def __init__(self, config=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def start(self, topology):
        session = LocalSession(topology, self.logger)
        try:
            # Model construction and checkpoint restore can touch process-wide Torch RNG.
            for spec in topology.roles:
                session.start_role(spec)
            return session
        except BaseException:
            session.close()
            raise
