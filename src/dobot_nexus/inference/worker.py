"""One execution thread for a shared inference instance and its session state."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor


class SerialWorker:
    """Callers must bound admission and keep at most one outstanding call per session.

    Cancellation skips queued work, but drains running work before returning. It
    cannot interrupt CUDA or make resetting a running session safe.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy")

    async def call(self, function, *args, cancel_pending=True):
        concurrent = self._executor.submit(function, *args)
        future = asyncio.wrap_future(concurrent)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Cleanup must still run if its caller is cancelled while it is queued.
            if cancel_pending:
                concurrent.cancel()
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            # Retrieve failures even when the disconnected client cannot receive them.
            if not future.cancelled():
                future.exception()
            raise

    async def close(self):
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)
