import importlib.util
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from dobot_nexus.core.contracts import PolicyOutput
from dobot_nexus.inference.server import create_app
from dobot_nexus.inference.session import PolicySession, SessionOptions
from dobot_nexus.inference.security import serving_token


class _Policy:
    loaded = True
    metadata = {}
    spec = SimpleNamespace(action_horizon=50)

    def create_session(self):
        return PolicySession(self, SessionOptions(50))

    def predict_chunk(self, observation, options):
        base = float(observation.state[0, 0])
        return PolicyOutput((base + np.arange(50, dtype=np.float32))[None, :, None])


def observation(value, *, env='robot', prompt='pick'):
    return {'images': {'base': np.zeros((1, 1, 3), dtype=np.uint8).tolist()},
            'state': [value], 'prompt': [prompt], 'env_ids': [env]}


@unittest.skipUnless(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'),
                     'requires serve and httpx')
class HttpSessionsTest(unittest.TestCase):
    def test_isolation_auth_identity_limits_and_concurrent_buffer_access(self):
        from fastapi.testclient import TestClient
        with TestClient(create_app(_Policy(), token='x'*32, max_sessions=2)) as client:
            self.assertEqual(client.post('/sessions').status_code, 401)
            client.headers['Authorization'] = 'Bearer ' + 'x'*32
            a, b = [client.post('/sessions').json()['session_id'] for _ in range(2)]
            self.assertNotEqual(a, b)
            self.assertEqual(client.post('/sessions').status_code, 429)
            def act(key, value, **kw):
                return client.post('/act', json={'session_id': key,
                                                 'observation': observation(value, **kw)})
            self.assertEqual(act(a, 1).json()['action'], [[1]])
            self.assertEqual(act(b, 100).json()['action'], [[100]])
            self.assertEqual(client.post('/reset', json={'session_id': a}).status_code, 200)
            self.assertEqual(act(b, 100).json()['action'], [[101]])
            self.assertEqual(act(b, 100, env='other').status_code, 400)
            self.assertEqual(act('unknown', 1).status_code, 404)
            with ThreadPoolExecutor(4) as pool:
                values = list(pool.map(lambda _: act(a, 10).json()['action'][0][0], range(8)))
            self.assertEqual(sorted(values), list(range(10, 18)))
            self.assertEqual(client.delete('/sessions/'+a).status_code, 200)
            self.assertEqual(act(a, 10).status_code, 404)
            self.assertEqual(client.post('/sessions').status_code, 200)
            self.assertEqual(client.post('/predict', json={'observation': observation(1),
                             'options': {'num_steps': 1000000}}).status_code, 400)

    def test_body_limit_and_expiry(self):
        from fastapi.testclient import TestClient
        from unittest.mock import patch
        with TestClient(create_app(_Policy(), max_body_bytes=1024, session_ttl=10)) as client:
            self.assertEqual(client.post('/predict', content=b'x'*1025).status_code, 413)
            with patch('time.monotonic', return_value=0):
                key = client.post('/sessions').json()['session_id']
            with patch('time.monotonic', return_value=11):
                self.assertEqual(client.post('/reset', json={'session_id': key}).status_code, 404)

    def test_network_bind_requires_a_secret(self):
        with self.assertRaises(ValueError):
            serving_token('0.0.0.0', '')
        self.assertEqual(serving_token('127.0.0.1', ''), '')
