"""Coordinate a single receive/send RPC pair on the driver thread."""

import time


class SingleInFlightSyncCoordinator:
    def __init__(self):
        self._pending = None
        self._results = {}
        self._validator = None
        self.last_result = None
        self.failed = False

    def _check_health(self):
        if self.failed:
            raise RuntimeError("A previous synchronization failed.")

    def _submit(self, recv_submit, send_submit, validator):
        try:
            self._pending = (recv_submit(), send_submit())
            self._validator = validator
            self._results.clear()
        except BaseException:
            self.failed = True
            raise

    def poll(self):
        """Observe each completed endpoint, including failures while its peer is pending."""
        self._check_health()
        if self._pending is None:
            return None
        try:
            for index, call in enumerate(self._pending):
                if index in self._results:
                    continue
                if time.monotonic() >= call.deadline:
                    raise TimeoutError("Synchronization RPC exceeded its deadline.")
                if call.done():
                    self._results[index] = call.wait()
            if len(self._results) != 2:
                return None
            result = (self._results[0], self._results[1])
            if self._validator is not None:
                self._validator(*result)
            self.last_result = result
            self._pending = self._validator = None
            self._results.clear()
            return result
        except BaseException:
            self.failed = True
            raise

    def drain(self):
        while True:
            result = self.poll()
            if self._pending is None:
                return result
            deadline = min(
                call.deadline for i, call in enumerate(self._pending) if i not in self._results
            )
            time.sleep(min(0.005, max(0, deadline - time.monotonic())))

    def sync_blocking(self, recv_submit, send_submit, validator=None):
        """Validate both endpoint results before committing the pair to last_result."""
        self.drain()
        self._submit(recv_submit, send_submit, validator)
        return self.drain()

    def sync_async(self, recv_submit, send_submit, validator=None):
        self.poll()
        if self._pending is None:
            self._submit(recv_submit, send_submit, validator)
