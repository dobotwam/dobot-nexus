"""Binary request/reply serving with injected, connection-scoped handlers."""

from __future__ import annotations

import asyncio
import http
import logging
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from dobot_nexus.inference.worker import SerialWorker

from . import msgpack_numpy

RequestHandler = Callable[[dict], dict]


@dataclass(frozen=True)
class ConnectionContext:
    remote_address: Any


class WebsocketPolicyServer:
    """No model or observation schema knowledge belongs in this transport.

    ``open_connection`` returns a context manager yielding a request callable.
    Its enter/exit, requests, and optional startup ``prepare`` all run on one
    worker. Each connection submits one job at a time; admission bounds queued
    jobs to ``max_connections``. Connection factories must defer resource work
    to their context manager's enter/exit.
    """

    def __init__(
        self, *, metadata: Mapping[str, Any],
        open_connection: Callable[[ConnectionContext], AbstractContextManager[RequestHandler]],
        host: str = "127.0.0.1", port: int = 8000,
        prepare: Callable[[], Mapping[str, Any] | None] | None = None,
        max_connections: int = 16, max_message_bytes: int = 32*1024*1024,
        token: str | None = None,
    ):
        if max_connections < 1:
            raise ValueError("max_connections must be positive.")
        from dobot_nexus.inference.security import serving_token
        self._token = serving_token(host, token)
        if type(max_message_bytes) is not int or max_message_bytes < 1:
            raise ValueError("max_message_bytes must be a positive integer.")
        self.metadata = dict(metadata)
        self.open_connection = open_connection
        self.host, self.port = host, int(port)
        self.prepare = prepare
        self.max_connections, self.max_message_bytes = max_connections, max_message_bytes
        self._connections = 0
        self._worker = None

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        async with self.start() as server:
            await server.serve_forever()

    @asynccontextmanager
    async def start(self):
        try:
            from websockets.asyncio.server import serve
        except ImportError as exc:
            raise ImportError(
                "WebSocket serving requires: pip install 'dobot-nexus[websocket]'"
            ) from exc
        if self._worker is not None:
            raise RuntimeError("This server is already running.")
        self._worker = SerialWorker()
        try:
            if self.prepare is not None:
                self.metadata.update(await self._worker.call(self.prepare) or {})
            async with serve(
                self._handler, self.host, self.port, compression=None,
                max_size=self.max_message_bytes, max_queue=1, process_request=self._health_check,
            ) as server:
                logging.info("Policy listening on ws://%s:%d", self.host, self.port)
                yield server
        finally:
            await self._worker.close()
            self._worker = None

    async def _dispatch(self, handler, request, closed):
        job = asyncio.create_task(self._worker.call(handler, request))
        try:
            await asyncio.wait((job, closed), return_when=asyncio.FIRST_COMPLETED)
            if closed.done():
                job.cancel()
                return None
            return await job
        finally:
            if not job.done():
                job.cancel()
            # Wait for any running computation before closing its session.
            try:
                await job
            except asyncio.CancelledError:
                pass

    def _pack_message(self, packer, value):
        payload = packer.pack(value)
        if len(payload) > self.max_message_bytes:
            raise ValueError("Policy response exceeds message size limit.")
        return payload

    async def _handler(self, websocket):
        from websockets.exceptions import ConnectionClosed

        if self._connections >= self.max_connections:
            await websocket.close(code=1013, reason="Policy connection limit reached")
            return
        self._connections += 1
        context = None
        entered = False
        closed = asyncio.create_task(websocket.wait_closed())
        previous_total_ms = None

        def enter():
            nonlocal entered
            handler = context.__enter__()
            entered = True
            return handler

        try:
            context = self.open_connection(ConnectionContext(websocket.remote_address))
            handler = await self._worker.call(enter)
            packer = msgpack_numpy.Packer()
            await websocket.send(self._pack_message(packer, self.metadata))
            while not closed.done():
                started = time.monotonic()
                payload = await websocket.recv()
                if isinstance(payload, str):
                    raise TypeError("Inference requests must be binary msgpack frames.")
                request = msgpack_numpy.unpackb(payload)
                if not isinstance(request, dict):
                    raise TypeError("Inference requests must be objects.")
                infer_started = time.monotonic()
                result = await self._dispatch(handler, request, closed)
                if result is None:
                    return
                result["server_timing"] = {"infer_ms": (time.monotonic() - infer_started) * 1000}
                if previous_total_ms is not None:
                    result["server_timing"]["prev_total_ms"] = previous_total_ms
                await websocket.send(self._pack_message(packer, result))
                previous_total_ms = (time.monotonic() - started) * 1000
        except ConnectionClosed:
            logging.info("Policy client %s disconnected", websocket.remote_address)
        except Exception:
            logging.exception("Policy request failed")
            try:
                await websocket.send("Policy request failed; check server logs.")
                await websocket.close(code=1011, reason="Internal server error")
            except ConnectionClosed:
                pass
        finally:
            try:
                if entered:
                    await self._worker.call(
                        context.__exit__, None, None, None, cancel_pending=False,
                    )
            finally:
                closed.cancel()
                await asyncio.gather(closed, return_exceptions=True)
                self._connections -= 1

    async def _health_check(self, connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        from dobot_nexus.inference.security import authorized
        if not authorized(request.headers.get("Authorization"), self._token):
            return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Authentication required\n")
        return None
