"""Synchronous NumPy WebSocket client with no model/backend dependency."""

from __future__ import annotations

import threading
import os

from . import msgpack_numpy


class WebsocketPolicyClient:
    """One request in flight per connection; never replay a failed request.

    A receive timeout closes the connection, so its late reply cannot be mistaken
    for the next request. ``timeout=None`` preserves the legacy unlimited wait.
    """

    def __init__(self, host="127.0.0.1", port=None, *, connect_timeout=20, timeout=30, token=None,
                 max_message_bytes=32*1024*1024):
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise ImportError(
                "WebSocket clients require: pip install 'dobot-nexus[websocket]'"
            ) from exc
        uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}"
        if port is not None:
            uri += f":{int(port)}"
        if type(max_message_bytes) is not int or max_message_bytes < 1:
            raise ValueError("max_message_bytes must be a positive integer.")
        self._max_message_bytes = max_message_bytes
        secret = os.environ.get("NEXUS_POLICY_TOKEN", "") if token is None else token
        self._lock = threading.Lock()
        self._timeout = timeout
        self._closed = False
        self._ws = connect(
            uri, compression=None, max_size=max_message_bytes, open_timeout=connect_timeout, proxy=None,
            additional_headers={"Authorization": "Bearer " + secret} if secret else None,
        )
        try:
            self._metadata = self._decode(self._ws.recv(timeout=connect_timeout))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _decode(payload):
        if isinstance(payload, str):
            raise RuntimeError(f"Error in inference server:\n{payload}")
        value = msgpack_numpy.unpackb(payload)
        if not isinstance(value, dict):
            raise TypeError("Policy responses must be objects.")
        return value

    def get_server_metadata(self):
        return dict(self._metadata)

    def request(self, payload, *, timeout=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("Policy client is closed; create a new connection explicitly.")
            encoded = msgpack_numpy.packb(payload)
            if len(encoded) > self._max_message_bytes:
                raise ValueError("Policy request exceeds message size limit.")
            try:
                self._ws.send(encoded)
                wait = self._timeout if timeout is None else timeout
                return self._decode(self._ws.recv(timeout=wait))
            except BaseException:
                self.close()
                raise

    def infer(self, observation):
        return self.request(observation)

    def close(self):
        if not self._closed:
            self._closed = True
            self._ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
