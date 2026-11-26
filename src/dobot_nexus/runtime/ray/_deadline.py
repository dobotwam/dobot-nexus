"""Wait for synchronous RLinf operations against a caller-owned deadline."""

import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError


def submit_daemon(action):
    finished = Future()

    def run():
        try:
            finished.set_result(action())
        except BaseException as exc:
            finished.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    return finished


def call_until(action, deadline, message):
    if time.monotonic() >= deadline:
        raise TimeoutError(message)
    finished = submit_daemon(action)
    try:
        value = finished.result(timeout=max(0, deadline - time.monotonic()))
    except FutureTimeoutError as exc:
        if time.monotonic() >= deadline or not finished.done():
            raise TimeoutError(message) from exc
        raise
    if time.monotonic() >= deadline:
        raise TimeoutError(message)
    return value
