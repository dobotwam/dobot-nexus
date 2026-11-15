from __future__ import annotations

import asyncio
import importlib.util
import subprocess
import sys
import threading
import unittest
import urllib.request
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dobot_nexus.inference.handlers import PredictionHandler
from dobot_nexus.inference.transport import msgpack_numpy as codec
from dobot_nexus.inference.transport.client import WebsocketPolicyClient
from dobot_nexus.inference.transport.websocket import WebsocketPolicyServer
from dobot_nexus.inference.worker import SerialWorker

HAS_WIRE = all(importlib.util.find_spec(name) is not None for name in ("websockets", "msgpack"))


class TransportBoundaryTest(unittest.TestCase):
    def test_imports_do_not_load_model_or_optional_backends(self):
        subprocess.run([sys.executable, "-c", """
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, *args):
        blocked = {'torch', 'jax', 'openpi', 'openpi_client', 'websockets', 'msgpack'}
        if fullname.split('.')[0] in blocked:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from dobot_nexus.inference.transport.client import WebsocketPolicyClient
from dobot_nexus.inference.transport.websocket import WebsocketPolicyServer
assert not any(n.startswith('dobot_nexus.models.') for n in sys.modules)
"""], check=True, capture_output=True, text=True)

    @unittest.skipUnless(HAS_WIRE, "requires websocket optional dependencies")
    def test_codec_matches_legacy_wire_bytes_and_rejects_unsupported_dtypes(self):
        import msgpack

        array = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
        legacy = msgpack.packb({b"__ndarray__": True, b"data": array.tobytes(),
                               b"dtype": array.dtype.str, b"shape": array.shape})
        self.assertEqual(codec.packb(array), legacy)
        np.testing.assert_array_equal(codec.unpackb(legacy), array)
        for scalar in (np.float32(1.25), np.int64(7), np.bool_(True)):
            decoded = codec.unpackb(codec.packb(scalar))
            self.assertEqual(decoded, scalar)
            self.assertEqual(decoded.dtype, scalar.dtype)
        for dtype in (object, complex, "V4"):
            with self.assertRaises(ValueError):
                codec.packb(np.zeros(1, dtype=dtype))

    @unittest.skipUnless(HAS_WIRE, "requires websocket optional dependencies")
    def test_client_timeout_closes_without_retry_or_late_reply_reuse(self):
        wire = mock.Mock()
        wire.recv.side_effect = [codec.packb({"name": "test"}), TimeoutError("late")]
        with mock.patch("websockets.sync.client.connect", return_value=wire):
            client = WebsocketPolicyClient(timeout=0.01)
            with self.assertRaises(TimeoutError):
                client.infer({"add_buffer": True})
            with self.assertRaisesRegex(RuntimeError, "closed"):
                client.infer({"reset": True})
        self.assertEqual(wire.send.call_count, 1)
        wire.close.assert_called_once()

    @unittest.skipUnless(HAS_WIRE, "requires websocket optional dependencies")
    def test_bad_metadata_and_server_errors_close_client(self):
        for replies in (["metadata error"], [codec.packb({}), "prediction error"]):
            wire = mock.Mock()
            wire.recv.side_effect = replies
            with mock.patch("websockets.sync.client.connect", return_value=wire):
                with self.assertRaisesRegex(RuntimeError, "error"):
                    client = WebsocketPolicyClient()
                    client.infer({})
            wire.close.assert_called_once()


class WorkerTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_drains_running_work_and_skips_queued_work(self):
        worker = SerialWorker()
        loop = asyncio.get_running_loop()
        started, release = asyncio.Event(), threading.Event()
        events = []

        def blocking():
            events.append(("begin", threading.get_ident()))
            loop.call_soon_threadsafe(started.set)
            if not release.wait(5):
                raise TimeoutError("test worker was not released")
            events.append(("end", threading.get_ident()))

        running = asyncio.create_task(worker.call(blocking))
        try:
            await asyncio.wait_for(started.wait(), 2)
            skipped = mock.Mock()
            pending = asyncio.create_task(worker.call(skipped))
            await asyncio.sleep(0)
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
            running.cancel()
            await asyncio.sleep(0)
            self.assertFalse(running.done())
            cleanup = asyncio.create_task(worker.call(
                lambda: events.append(("cleanup", threading.get_ident())), cancel_pending=False,
            ))
            await asyncio.sleep(0)
            cleanup.cancel()
            await asyncio.sleep(0)
            self.assertFalse(cleanup.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await running
            with self.assertRaises(asyncio.CancelledError):
                await cleanup
            skipped.assert_not_called()
            self.assertEqual([e[0] for e in events], ["begin", "end", "cleanup"])
            self.assertEqual(len({e[1] for e in events}), 1)
            self.assertNotEqual(events[0][1], threading.get_ident())
        finally:
            release.set()
            await worker.close()


@unittest.skipUnless(HAS_WIRE, "requires websocket optional dependencies")
class WebsocketContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_generic_non_robot_policy_client_and_legacy_binary_requests(self):
        from websockets.asyncio.client import connect

        threads = []

        def predict(observation, options):
            threads.append(threading.get_ident())
            self.assertEqual(observation["state"].shape, (3,))
            return SimpleNamespace(actions=np.full((1, 4, 2), options.seed, np.float32), timing={})

        prediction = PredictionHandler(SimpleNamespace(predict_chunk=predict), seed=5)
        server = WebsocketPolicyServer(
            metadata={"state_dim": 3, "action_dim": 2},
            prepare=lambda: (threads.append(threading.get_ident()) or {"ready": True}),
            open_connection=lambda context: nullcontext(prediction), port=0,
        )
        async with server.start() as listener:
            port = listener.sockets[0].getsockname()[1]

            def client_requests():
                with WebsocketPolicyClient(port=port) as client:
                    self.assertTrue(client.get_server_metadata()["ready"])
                    result = client.infer({"state": np.zeros(3, np.float32)})
                    self.assertEqual(result["actions"].shape, (1, 4, 2))
                    self.assertEqual(result["seed"], 5)
                    self.assertEqual(client.request({"state": np.ones(3)})["seed"], 6)

            await asyncio.to_thread(client_requests)
            # The existing binary metadata-first protocol works without the new client.
            async with connect(f"ws://127.0.0.1:{port}", proxy=None) as wire:
                self.assertEqual(codec.unpackb(await wire.recv())["action_dim"], 2)
                await wire.send(codec.packb({"state": np.zeros(3)}))
                result = codec.unpackb(await wire.recv())
                self.assertEqual(result["seed"], 7)
                self.assertIn("server_timing", result)
        self.assertEqual(len(set(threads)), 1)  # warmup and predictions share execution thread

    async def test_disconnect_drains_before_cleanup_and_health_remains_responsive(self):
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        loop = asyncio.get_running_loop()
        started, cleaned = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        events = []

        @contextmanager
        def connection(context):
            events.append(("open", threading.get_ident()))

            def infer(request):
                events.append(("infer", threading.get_ident()))
                loop.call_soon_threadsafe(started.set)
                if not release.wait(5):
                    raise TimeoutError("test worker was not released")
                events.append(("done", threading.get_ident()))
                return {"actions": np.zeros((4, 2))}

            try:
                yield infer
            finally:
                events.append(("close", threading.get_ident()))
                loop.call_soon_threadsafe(cleaned.set)

        server = WebsocketPolicyServer(
            metadata={}, open_connection=connection, port=0, max_connections=1,
        )
        async with server.start() as listener:
            port = listener.sockets[0].getsockname()[1]
            wire = await connect(f"ws://127.0.0.1:{port}", proxy=None)
            try:
                await wire.recv()
                await wire.send(codec.packb({}))
                await asyncio.wait_for(started.wait(), 2)

                def health():
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                        return response.read()

                self.assertEqual(await asyncio.to_thread(health), b"OK\n")
                async with connect(f"ws://127.0.0.1:{port}", proxy=None) as extra:
                    with self.assertRaises(ConnectionClosed) as raised:
                        await extra.recv()
                    self.assertEqual(raised.exception.rcvd.code, 1013)
                await wire.close()
                self.assertFalse(cleaned.is_set())
                release.set()
                await asyncio.wait_for(cleaned.wait(), 2)
            finally:
                release.set()
                await wire.close()
        self.assertEqual([e[0] for e in events], ["open", "infer", "done", "close"])
        self.assertEqual(len({e[1] for e in events}), 1)

    async def test_invalid_requests_report_legacy_text_error_and_cleanup(self):
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed

        closed = threading.Event()

        @contextmanager
        def connection(context):
            try:
                yield lambda request: {}
            finally:
                closed.set()

        server = WebsocketPolicyServer(metadata={}, open_connection=connection, port=0)
        async with server.start() as listener:
            port = listener.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}", proxy=None) as wire:
                await wire.recv()
                with self.assertLogs(level="ERROR"):
                    await wire.send("bad request")
                    self.assertEqual("Policy request failed; check server logs.", await wire.recv())
                    with self.assertRaises(ConnectionClosed):
                        await wire.recv()
        self.assertTrue(closed.is_set())

@unittest.skipUnless(HAS_WIRE, "requires websocket optional dependencies")
class TransportSecurityTest(unittest.IsolatedAsyncioTestCase):
    async def test_authentication_and_message_size_limits(self):
        from websockets.asyncio.client import connect
        from websockets.exceptions import InvalidStatus, ConnectionClosed
        server = WebsocketPolicyServer(metadata={}, open_connection=lambda _: nullcontext(lambda x: x),
                                       port=0, token='x'*32, max_message_bytes=128)
        async with server.start() as listener:
            uri = f'ws://127.0.0.1:{listener.sockets[0].getsockname()[1]}'
            with self.assertRaises(InvalidStatus):
                async with connect(uri, proxy=None):
                    pass
            async with connect(uri, proxy=None, additional_headers={'Authorization': 'Bearer '+'x'*32}) as ws:
                await ws.recv()
                await ws.send(b'x'*129)
                with self.assertRaises(ConnectionClosed):
                    await ws.recv()

    def test_malformed_numpy_frames_are_rejected(self):
        import msgpack
        for dtype, shape, data in [('O', [1], b'\0'*8), ('f4', [1000000], b''),
                                   ('f4', [-1], b''), ('V4', [1], b'\0'*4)]:
            value = {b'__ndarray__': True, b'dtype': dtype, b'shape': shape, b'data': data}
            with self.assertRaises(ValueError):
                codec.unpackb(msgpack.packb(value))

    def test_outbound_messages_are_bounded_on_client_and_server(self):
        wire = mock.Mock()
        wire.recv.return_value = codec.packb({})
        with mock.patch('websockets.sync.client.connect', return_value=wire):
            client = WebsocketPolicyClient(max_message_bytes=64)
            with self.assertRaises(ValueError):
                client.request({'data': b'x'*65})
            wire.send.assert_not_called()
            client.close()
        server = WebsocketPolicyServer(metadata={}, open_connection=lambda _: None,
                                       max_message_bytes=64)
        with self.assertRaises(ValueError):
            server._pack_message(codec.Packer(), {'data': b'x'*65})
