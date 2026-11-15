import os
import socket
import struct
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

from examples.pi05_real_robot._runtime import wire


class RobotWireTest(unittest.TestCase):
    def test_authenticated_handshake_array_roundtrip_and_wrong_key(self):
        for matching in (True, False):
            with self.subTest(matching=matching):
                left, right = socket.socketpair()
                with left, right, ThreadPoolExecutor(1) as pool:
                    future = pool.submit(wire.authenticate, left, server=True, key=b'x'*32,
                                         timeout=.2)
                    if not matching:
                        with self.assertRaises((PermissionError, TimeoutError, ConnectionError)):
                            wire.authenticate(right, server=False, key=b'y'*32, timeout=.2)
                        with self.assertRaises(PermissionError):
                            future.result()
                        continue
                    client = wire.authenticate(right, server=False, key=b'x'*32)
                    server = future.result()
                    wire.send_message(client, {'array': np.arange(18, dtype=np.float32)})
                    np.testing.assert_array_equal(wire.recv_message(server)['array'], np.arange(18))
                    wire.send_message(server, {'ok': True})
                    self.assertTrue(wire.recv_message(client)['ok'])

    def test_rejects_oversized_forged_and_replayed_frames(self):
        for kind in ('oversize', 'forged', 'replay'):
            with self.subTest(kind=kind):
                left, right = socket.socketpair()
                with left, right:
                    server = wire.Channel(left, b'x'*32, server=True, timeout=.1)
                    client = wire.Channel(right, b'x'*32, server=False, timeout=.1)
                    if kind == 'oversize':
                        right.sendall(struct.pack('!QQ', wire.MAX_FRAME_BYTES+1, 0))
                    elif kind == 'forged':
                        right.sendall(struct.pack('!QQ', 1, 0) + b'0'*32 + b'\x80')
                    else:
                        client.send({'ok': True})
                        self.assertTrue(server.receive()['ok'])
                        client.sent = 0
                        client.send({'ok': True})
                    with self.assertRaises((ValueError, PermissionError)):
                        server.receive()

    def test_missing_secret_raw_socket_and_stalled_frame_are_rejected(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            wire.configured_key()
        left, right = socket.socketpair()
        with left, right:
            with self.assertRaises(TypeError):
                wire.send_message(left, {})
            with self.assertRaises(TimeoutError):
                wire.Channel(left, b'x'*32, server=True, timeout=.02).receive()

    def test_array_decoder_rejects_object_dtype_and_invalid_size(self):
        for dtype, shape, data in [('O', [1], b'\0'*8), ('f4', [1000000], b''),
                                   ('f4', [-1], b'')]:
            with self.assertRaises(ValueError):
                wire._unpack({'__numpy__': True, 'dtype': dtype, 'shape': shape, 'data': data})

    def test_operator_pause_does_not_expire_before_observation_arrives(self):
        left, right = socket.socketpair()
        with left, right, ThreadPoolExecutor(1) as pool:
            client = wire.Channel(left, b'x'*32, server=False, timeout=.05)
            server = wire.Channel(right, b'x'*32, server=True, timeout=.05)
            future = pool.submit(wire.recv_message, client, wait_for_message=True)
            try:
                time.sleep(.15)
                self.assertFalse(future.done())
                server.send({'type': 'observation', 'state': np.zeros(14, np.float32)})
                self.assertEqual(future.result(timeout=1)['type'], 'observation')
            finally:
                right.shutdown(socket.SHUT_RDWR)

    def test_idle_wait_still_bounds_partial_frames_and_detects_disconnect(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                left, right = socket.socketpair()
                with left, right:
                    client = wire.Channel(left, b'x'*32, server=False, timeout=.02)
                    if partial:
                        right.sendall(b'\x00')
                    else:
                        right.shutdown(socket.SHUT_WR)
                    with self.assertRaises(TimeoutError if partial else ConnectionError):
                        wire.recv_message(client, wait_for_message=True)
