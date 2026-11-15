from __future__ import annotations

from typing import Any

import numpy as np

from dobot_nexus.core.contracts import SamplingOptions
from dobot_nexus.data.observation import observation_from_json as _observation_from_json


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)



def create_app(policy, *, token=None, max_sessions=16, session_ttl=900, max_body_bytes=8*1024*1024):
    import asyncio
    import secrets
    import time
    from threading import RLock

    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from starlette.responses import JSONResponse
    except ImportError as exc:
        raise ImportError("Serving requires `pip install 'dobot-nexus[serve]'`.") from exc
    from dobot_nexus.inference.security import authorized, serving_token

    if max_sessions < 1 or session_ttl <= 0 or max_body_bytes < 1:
        raise ValueError("HTTP session, expiry and body limits must be positive.")
    token = serving_token(token=token)
    app = FastAPI(title="Dobot Nexus Policy Server", version="1")
    lock, sessions = RLock(), {}

    class BodyLimit:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope['type'] != 'http':
                return await self.app(scope, receive, send)
            body, size = [], 0
            deadline = time.monotonic() + 30
            while True:
                try:
                    message = await asyncio.wait_for(receive(), timeout=max(0, deadline-time.monotonic()))
                except TimeoutError:
                    return await JSONResponse({'detail': 'Request body timeout'}, 408)(scope, receive, send)
                if message['type'] == 'http.disconnect':
                    return
                size += len(message.get('body', b''))
                if size > max_body_bytes:
                    return await JSONResponse({'detail': 'Request body too large'}, 413)(scope, receive, send)
                body.append(message.get('body', b''))
                if not message.get('more_body', False):
                    break
            delivered = False
            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {'type': 'http.request', 'body': b''.join(body), 'more_body': False}
                return await receive()
            await self.app(scope, replay, send)

    app.add_middleware(BodyLimit)

    def authenticate(authorization: str | None = Header(default=None)):
        if not authorized(authorization, token):
            raise HTTPException(401, 'Authentication required')

    protected = [Depends(authenticate)]

    def prune():
        now = time.monotonic()
        for key in [key for key, (_, used) in sessions.items() if now-used >= session_ttl]:
            sessions.pop(key)

    def lookup(request):
        prune()
        key = request.get('session_id')
        if not isinstance(key, str) or key not in sessions:
            raise HTTPException(404, 'Unknown or expired session_id; create a session first')
        session = sessions[key][0]
        sessions[key] = (session, time.monotonic())
        return session

    @app.get('/healthz')
    def healthz():
        return {'status': 'ok', 'loaded': policy.loaded}

    @app.get('/metadata', dependencies=protected)
    def metadata():
        return _jsonable(policy.metadata)

    @app.post('/sessions', dependencies=protected)
    def create_session():
        with lock:
            prune()
            if len(sessions) >= max_sessions:
                raise HTTPException(429, 'Session limit reached; delete or expire an existing session')
            key = secrets.token_urlsafe(32)
            sessions[key] = (policy.create_session(), time.monotonic())
            return {'session_id': key}

    @app.delete('/sessions/{session_id}', dependencies=protected)
    def delete_session(session_id: str):
        with lock:
            lookup({'session_id': session_id})
            sessions.pop(session_id)
        return {'status': 'ok'}

    @app.post('/reset', dependencies=protected)
    def reset(request: dict[str, Any]):
        with lock:
            lookup(request).reset(seed=request.get('seed'))
        return {'status': 'ok'}

    @app.post('/predict', dependencies=protected)
    def predict(request: dict[str, Any]):
        try:
            observation = _observation_from_json(request['observation'])
            raw_options = request.get('options', {})
            num_steps = raw_options.get('num_steps')
            if num_steps is not None and (type(num_steps) is not int or not 1 <= num_steps <= 100):
                raise ValueError('num_steps must be an integer between 1 and 100.')
            with lock:
                output = policy.predict_chunk(observation, SamplingOptions(
                    seed=raw_options.get('seed'), num_steps=num_steps))
            return _jsonable({'actions': output.actions, 'valid_mask': output.valid_mask,
                              'timing': output.timing, 'aux': output.aux})
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/act', dependencies=protected)
    def act(request: dict[str, Any]):
        try:
            observation = _observation_from_json(request['observation'])
            with lock:
                session = lookup(request)
                action = session.act(observation)
                return {'action': _jsonable(action), 'remaining_steps': session.remaining_steps}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


def run_server(policy, *, host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Serving requires `pip install 'dobot-nexus[serve]'`.") from exc
    from dobot_nexus.inference.security import serving_token
    token = serving_token(host)
    uvicorn.run(create_app(policy, token=token), host=host, port=port, limit_concurrency=64)
