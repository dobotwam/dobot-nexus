"""Authenticated robot TCP v2. Standalone: NumPy, msgpack and the standard library."""
from __future__ import annotations

import hmac
import math
import os
import struct
import time

import numpy as np

MAX_FRAME_BYTES = 32 * 1024 * 1024
TIMEOUT_SECONDS = 30.0
MAGIC = b'DNX2'


def configured_key():
    key = os.environ.get('NEXUS_ROBOT_TOKEN', '').encode('utf-8')
    if len(key) < 32:
        raise ValueError('Set NEXUS_ROBOT_TOKEN to the same random secret (at least 32 bytes) on both hosts.')
    return key


def _read(sock, count, deadline):
    parts = []
    while count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Robot TCP frame deadline exceeded.')
        sock.settimeout(remaining)
        part = sock.recv(min(count, 65536))
        if not part:
            raise ConnectionError('Robot TCP connection closed.')
        parts.append(part)
        count -= len(part)
    return b''.join(parts)


def _pack(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf':
            raise ValueError('Only numeric or boolean arrays are supported.')
        return {'__numpy__': True, 'data': value.tobytes(),
                'dtype': value.dtype.str, 'shape': value.shape}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported robot wire type: {type(value).__name__}')


def _unpack(value):
    if value.get('__numpy__'):
        dtype, shape, data = np.dtype(value['dtype']), value['shape'], value['data']
        if dtype.kind not in 'biuf' or len(shape) > 8 or any(
            type(n) is not int or n < 0 for n in shape
        ):
            raise ValueError('Invalid robot array dtype or shape.')
        if not isinstance(data, bytes) or math.prod(shape) * dtype.itemsize != len(data):
            raise ValueError('Robot array shape does not match its bytes.')
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    return value


class Channel:
    def __init__(self, sock, key, *, server, timeout=TIMEOUT_SECONDS):
        self.sock, self.key, self.timeout = sock, key, timeout
        self.send_role, self.recv_role = (b'S', b'C') if server else (b'C', b'S')
        self.sent, self.received = 0, 0

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.sock.close()

    def send(self, message):
        import msgpack
        payload = msgpack.packb(message, default=_pack, use_bin_type=True)
        if not 0 < len(payload) <= MAX_FRAME_BYTES:
            raise ValueError('Robot message exceeds frame size limit.')
        header = struct.pack('!QQ', len(payload), self.sent)
        signature = hmac.digest(self.key, self.send_role + header + payload, 'sha256')
        self.sock.settimeout(self.timeout)
        self.sock.sendall(header + signature + payload)
        self.sent += 1
        return len(payload) + 48

    def receive(self, *, wait_for_message=False):
        import msgpack
        first = b''
        if wait_for_message:
            # Only authenticated clients awaiting operator-driven observations
            # opt into idle waiting. Once a frame starts, its deadline is fixed.
            self.sock.settimeout(None)
            first = self.sock.recv(1)
            if not first:
                raise ConnectionError('Robot TCP connection closed.')
        deadline = time.monotonic() + self.timeout
        header = first + _read(self.sock, 16 - len(first), deadline)
        size, sequence = struct.unpack('!QQ', header)
        if not 0 < size <= MAX_FRAME_BYTES or sequence != self.received:
            raise ValueError('Invalid robot frame size or sequence.')
        signature = _read(self.sock, 32, deadline)
        payload = _read(self.sock, size, deadline)
        expected = hmac.digest(self.key, self.recv_role + header + payload, 'sha256')
        if not hmac.compare_digest(signature, expected):
            raise PermissionError('Robot frame authentication failed.')
        value = msgpack.unpackb(payload, object_hook=_unpack, raw=False,
                                max_array_len=MAX_FRAME_BYTES, max_map_len=4096)
        if not isinstance(value, dict):
            raise TypeError('Robot messages must be objects.')
        self.received += 1
        return value


def authenticate(sock, *, server, key=None, timeout=TIMEOUT_SECONDS):
    """Mutual nonce proof derives a fresh, direction-bound key for each connection."""
    key = configured_key() if key is None else key
    if len(key) < 32:
        raise ValueError('Robot authentication key must contain at least 32 bytes.')
    deadline = time.monotonic() + timeout
    sock.settimeout(timeout)
    if server:
        server_nonce = os.urandom(32)
        sock.sendall(MAGIC + server_nonce)
        reply = _read(sock, 64, deadline)
        client_nonce, proof = reply[:32], reply[32:]
        transcript = MAGIC + server_nonce + client_nonce
        if not hmac.compare_digest(proof, hmac.digest(key, b'client' + transcript, 'sha256')):
            raise PermissionError('Robot client authentication failed.')
        sock.sendall(hmac.digest(key, b'server' + transcript, 'sha256'))
    else:
        hello = _read(sock, 36, deadline)
        if hello[:4] != MAGIC:
            raise ValueError('Robot TCP v2 is required on both hosts.')
        client_nonce = os.urandom(32)
        transcript = hello + client_nonce
        sock.sendall(client_nonce + hmac.digest(key, b'client' + transcript, 'sha256'))
        proof = _read(sock, 32, deadline)
        if not hmac.compare_digest(proof, hmac.digest(key, b'server' + transcript, 'sha256')):
            raise PermissionError('Robot server authentication failed.')
    return Channel(sock, hmac.digest(key, b'session' + transcript, 'sha256'),
                   server=server, timeout=timeout)


def send_message(connection, message):
    if not isinstance(connection, Channel):
        raise TypeError('An authenticated robot Channel is required.')
    return connection.send(message)


def recv_message(connection, *, wait_for_message=False):
    if not isinstance(connection, Channel):
        raise TypeError('An authenticated robot Channel is required.')
    return connection.receive(wait_for_message=wait_for_message)
