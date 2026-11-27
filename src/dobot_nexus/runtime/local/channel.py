"""Single-slot mailboxes with process-equivalent message ownership."""

import pickle
import threading
import time

from ..contracts import ChannelEmpty, ChannelFull, ChannelSpec, check_message_bytes

_EMPTY = object()


class BoundedLocalChannel:
    def __init__(self, spec: ChannelSpec):
        self.spec = spec
        self._changed = threading.Condition()
        self._closed = False
        self._payload = _EMPTY

    def put_nowait(self, payload):
        message = pickle.loads(check_message_bytes(payload, self.spec.max_message_bytes))
        with self._changed:
            if self._closed:
                raise RuntimeError("Channel closed.")
            if self._payload is not _EMPTY:
                raise ChannelFull(self.spec.name)
            self._payload = message
            self._changed.notify_all()

    def _take(self):
        if self._closed:
            raise RuntimeError("Channel closed.")
        if self._payload is _EMPTY:
            raise ChannelEmpty(self.spec.name)
        message, self._payload = self._payload, _EMPTY
        return message

    def try_get(self):
        with self._changed:
            return self._take()

    def recv(self, timeout):
        deadline = time.monotonic() + timeout
        with self._changed:
            while self._payload is _EMPTY and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._changed.wait(remaining)
            if not self._closed and time.monotonic() >= deadline:
                raise TimeoutError(f"Channel {self.spec.name!r} receive deadline exceeded.")
            return self._take()

    def close(self):
        with self._changed:
            self._closed = True
            self._payload = _EMPTY
            self._changed.notify_all()


class LocalChannelBus:
    def __init__(self, specs):
        self._channels = {spec.name: BoundedLocalChannel(spec) for spec in specs}

    def put_nowait(self, name, payload):
        self._channels[name].put_nowait(payload)

    def try_get(self, name):
        return self._channels[name].try_get()

    def recv(self, name, *, timeout):
        return self._channels[name].recv(timeout)

    def close(self):
        for channel in self._channels.values():
            channel.close()
