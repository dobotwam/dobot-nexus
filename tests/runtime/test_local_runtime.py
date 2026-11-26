import os
import sqlite3
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from unittest.mock import patch

from dobot_nexus.runtime.contracts import (
    ChannelEmpty,
    ChannelFull,
    ChannelSpec,
    PlacementSpec,
    RoleSpec,
    TopologySpec,
)
from dobot_nexus.runtime.local.backend import LocalRuntime
from dobot_nexus.runtime.local.channel import BoundedLocalChannel


class DatabaseRole:
    rpc_methods = ("read", "receive", "send")
    closed = []

    def __init__(self, fail=False):
        self.owner = threading.get_ident()
        self.fail = fail
        self.connection = sqlite3.connect(":memory:")

    def initialize(self, ctx):
        self.ctx = ctx
        self.connection.execute("create table affinity (value integer)")
        if self.fail:
            raise ValueError("injected init failure")
        return self.owner, ctx.device

    def read(self):
        self.connection.execute("select * from affinity").fetchall()
        return threading.get_ident()

    def receive(self):
        return self.ctx.channels.recv("lane", timeout=10)

    def send(self, payload):
        self.ctx.channels.put_nowait("lane", payload)

    def close(self):
        self.connection.close()
        self.closed.append((self.ctx.role_name, threading.get_ident()))


class LocalRuntimeTest(unittest.TestCase):
    def start(self, fail=False):
        roles = tuple(
            RoleSpec(
                name,
                f"{__name__}:DatabaseRole",
                PlacementSpec("gpu", 0),
                init_kwargs={"fail": fail and name == "second"},
            )
            for name in ("first", "second")
        )
        DatabaseRole.closed.clear()
        session = LocalRuntime().start(TopologySpec("test", roles, (ChannelSpec("lane"),), 3))
        self.addCleanup(session.close)
        return session

    def test_thread_affinity_independent_progress_and_reverse_cleanup(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            session = self.start()
            first, second = session.role("first"), session.role("second")
            owner, device = session.init_results["first"]
            other, _ = session.init_results["second"]
            self.assertEqual(device, "cpu")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")
            self.assertNotEqual(owner, other)
            for _ in range(3):
                self.assertEqual(first.call("read").wait(), owner)
            future = first.call("receive")
            second.call("send", "hello").wait()
            self.assertEqual(future.wait(), "hello")
            with self.assertRaisesRegex(ValueError, "not allowed"):
                first.call("close").wait()
            blocked = first.call("receive")
            session.close()
            with self.assertRaisesRegex(RuntimeError, "Channel closed"):
                blocked.wait()
            with self.assertRaisesRegex(RuntimeError, "Session closed"):
                second.call("read")
            self.assertCountEqual(DatabaseRole.closed, [("second", other), ("first", owner)])

    def test_wait_any_checks_expired_before_ready_and_wait_cannot_extend_deadline(self):
        session = self.start()
        slow = session.role("first").call("receive", timeout=0.03)
        fast = session.role("second").call("read")
        fast.wait()
        time.sleep(0.04)
        with self.assertRaisesRegex(TimeoutError, "slow"):
            session.wait_any({"fast": fast, "slow": slow})
        with self.assertRaises(TimeoutError):
            slow.wait(timeout=10)

    def test_channel_copy_full_empty_receive_timeout_and_close_wakeup(self):
        channel = BoundedLocalChannel(ChannelSpec("lane"))
        with self.assertRaises(ChannelEmpty):
            channel.try_get()
        payload = [{"value": 1}]
        channel.put_nowait(payload)
        payload[0]["value"] = 2
        with self.assertRaises(ChannelFull):
            channel.put_nowait(None)
        self.assertEqual(channel.recv(0.5), [{"value": 1}])
        with self.assertRaises(TimeoutError):
            channel.recv(0.01)
        with ThreadPoolExecutor(max_workers=2) as pool:
            waiting = [pool.submit(channel.recv, 5) for _ in range(2)]
            channel.close()
            for future in waiting:
                with self.assertRaisesRegex(RuntimeError, "Channel closed"):
                    future.result(timeout=0.5)

    def test_receiver_reacquiring_lock_at_deadline_leaves_ack_queued(self):
        channel = BoundedLocalChannel(ChannelSpec("acks"))
        waiting = threading.Event()
        wait = channel._changed.wait

        def signal_then_wait(timeout):
            waiting.set()
            return wait(timeout)

        with (
            patch("dobot_nexus.runtime.local.channel.time") as clock,
            patch.object(channel._changed, "wait", signal_then_wait),
            ThreadPoolExecutor(1) as pool,
            closing(channel),
        ):
            clock.monotonic.return_value = 100.0
            receive = pool.submit(channel.recv, 5.0)
            self.assertTrue(waiting.wait(1))
            with channel._changed:
                channel.put_nowait("next ACK")
                clock.monotonic.return_value = 105.0
            with self.assertRaisesRegex(TimeoutError, "receive deadline"):
                receive.result(timeout=1)
            self.assertEqual(channel.try_get(), "next ACK")

    def test_enqueue_wakes_live_receiver_when_first_waiter_expires(self):
        channel = BoundedLocalChannel(ChannelSpec("acks"))
        waiting = threading.Event()
        wait = channel._changed.wait

        def signal_then_wait(timeout):
            waiting.set()
            return wait(timeout)

        with (
            patch("dobot_nexus.runtime.local.channel.time") as clock,
            patch.object(channel._changed, "wait", signal_then_wait),
            ThreadPoolExecutor(2) as pool,
            closing(channel),
        ):
            clock.monotonic.return_value = 100.0
            first = pool.submit(channel.recv, 5.0)
            self.assertTrue(waiting.wait(1))
            waiting.clear()
            second = pool.submit(channel.recv, 30.0)
            self.assertTrue(waiting.wait(1))
            with channel._changed:
                channel.put_nowait("next ACK")
                clock.monotonic.return_value = 105.0
            with self.assertRaisesRegex(TimeoutError, "receive deadline"):
                first.result(timeout=1)
            self.assertEqual(second.result(timeout=1), "next ACK")
            with self.assertRaises(ChannelEmpty):
                channel.try_get()

    def test_close_bounded_when_external_io_blocks_owner_thread(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import threading
import time
from dobot_nexus.runtime.contracts import PlacementSpec, RoleSpec, TopologySpec
from dobot_nexus.runtime.local import LocalRuntime

started, unblock, closed = (threading.Event() for _ in range(3))

class BlockingRole:
    rpc_methods = ("read",)
    def initialize(self, ctx):
        pass
    def read(self):
        started.set()
        unblock.wait()
    def close(self):
        closed.set()

role = RoleSpec("io", "__main__:BlockingRole", PlacementSpec("cpu"))
session = LocalRuntime().start(TopologySpec("test", (role,), (), rpc_timeout=0.2))
try:
    session.role("io").call("read")
    assert started.wait(2)
    now = time.monotonic()
    session.close()
    assert time.monotonic() - now < 1.0 and not closed.is_set()
finally:
    unblock.set()
assert closed.wait(2)
""",
            ],
            check=True,
            timeout=15,
            capture_output=True,
        )

    def test_healthy_peer_closes_before_blocked_close_exhausts_shared_budget(self):
        session = self.start()
        session.timeout = 0.3
        healthy, started, release = (threading.Event() for _ in range(3))
        close_role = DatabaseRole.close

        def close(role):
            if role.ctx.role_name == "second":
                started.set()
                release.wait(3)
            else:
                healthy.set()
            close_role(role)

        with patch.object(DatabaseRole, "close", close), ThreadPoolExecutor(1) as pool:
            closing = pool.submit(session.close)
            try:
                self.assertTrue(started.wait(1))
                self.assertTrue(healthy.wait(0.15))
                self.assertFalse(closing.done())
            finally:
                release.set()
            closing.result(timeout=1)

    def test_concurrent_sessions_serialize_construction_and_rng_initialization(self):
        import torch

        from dobot_nexus.runtime.local.backend import _LocalRole

        first_in_init, second_attempt, second_constructed, release = (
            threading.Event() for _ in range(4)
        )

        class SeedRole:
            def __init__(self, seed):
                torch.random.default_generator.manual_seed(seed)
                if seed == 2:
                    second_constructed.set()

            def initialize(self, ctx):
                if ctx.role_name == "first":
                    first_in_init.set()
                    release.wait(3)
                return torch.rand(1).item()

            def close(self):
                pass

        initialize = _LocalRole.initialize

        def attempt(owner):
            if owner.spec.name == "second":
                second_attempt.set()
            return initialize(owner)

        def start(name, seed):
            role = RoleSpec(name, "rng:SeedRole", PlacementSpec("cpu"), init_kwargs={"seed": seed})
            return LocalRuntime().start(TopologySpec(name, (role,), (), rpc_timeout=5))

        with (
            torch.random.fork_rng(devices=[]),
            patch("dobot_nexus.runtime.local.backend.import_from_string", return_value=SeedRole),
            patch.object(_LocalRole, "initialize", attempt),
            ThreadPoolExecutor(2) as pool,
        ):
            first = pool.submit(start, "first", 1)
            self.assertTrue(first_in_init.wait(1))
            second = pool.submit(start, "second", 2)
            try:
                self.assertTrue(second_attempt.wait(1))
                self.assertFalse(second_constructed.wait(0.05))
            finally:
                release.set()
            for seed, name, future in ((1, "first", first), (2, "second", second)):
                session = future.result(timeout=2)
                self.addCleanup(session.close)
                expected = torch.rand(1, generator=torch.Generator().manual_seed(seed)).item()
                self.assertEqual(session.init_results[name], expected)

    def test_partial_start_closes_failing_and_initialized_roles(self):
        with self.assertRaisesRegex(ValueError, "injected init failure"):
            self.start(fail=True)
        self.assertCountEqual([name for name, _ in DatabaseRole.closed], ["second", "first"])
