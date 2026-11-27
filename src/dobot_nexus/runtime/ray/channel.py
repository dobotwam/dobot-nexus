"""Bounded, named RLinf channels declared by the job topology."""

import asyncio
import threading
import time

import ray
from rlinf.scheduler import Channel, NodePlacementStrategy
from rlinf.scheduler.channel.channel_worker import ChannelWorker

from ..contracts import ChannelEmpty, ChannelFull, check_message_bytes
from ._deadline import call_until


class DeadlineChannelWorker(ChannelWorker):
    async def take_before(self, deadline):
        # Phase 1 places both processes on one node with one monotonic clock.
        # The pinned nowait path dequeues on this actor loop without yielding.
        if time.monotonic() < deadline:
            try:
                return True, await self.get_via_ray(nowait=True)
            except asyncio.QueueEmpty:
                pass
        return False, None


class RayChannelBus:
    def __init__(self):
        self._channels = {}
        self._readers = {}
        self._limits = {}
        self._poisoned = set()

    def create(self, specs, prefix, cluster, owned_groups, *, deadline):
        launched = []
        for spec in specs:
            name = f"{prefix}-{spec.name}"
            # Channel.create can implicitly adopt an existing group and hides
            # partially launched workers. Own the group before calling launch.
            group = DeadlineChannelWorker.create_group(maxsize=spec.maxsize)
            owned_groups.append(group)
            group.launch(
                cluster,
                name=name,
                placement_strategy=NodePlacementStrategy([0]),
                max_concurrency=2**31 - 1,
                disable_distributed_log=True,
            )
            launched.append((spec, name, group))
        for spec, name, group in launched:
            # launch submits readiness asynchronously. A completed worker method
            # proves rank zero finished construction and scheduler registration.
            capacity = call_until(
                lambda g=group: g.maxsize().wait(), deadline, "Channel startup timed out."
            )
            if capacity != [spec.maxsize]:
                raise RuntimeError(f"Unexpected capacity for channel {name!r}: {capacity!r}")
        for spec, name, group in launched:
            self._channels[spec.name] = call_until(
                lambda n=name: Channel.connect(n, current_worker=None),
                deadline,
                "Channel connection timed out.",
            )
            self._readers[spec.name] = group.worker_info_list[0].worker
            self._limits[spec.name] = spec.max_message_bytes

    def put_nowait(self, name, payload):
        check_message_bytes(payload, self._limits[name])
        try:
            self._channels[name].put_nowait(payload)
        except asyncio.QueueFull as exc:
            raise ChannelFull(name) from exc

    def _check_readable(self, name):
        if name in self._poisoned:
            raise RuntimeError(
                f"Channel {name!r} is unusable after an indeterminate receive timeout."
            )

    def try_get(self, name):
        self._check_readable(name)
        try:
            return self._channels[name].get_nowait()
        except asyncio.QueueEmpty as exc:
            raise ChannelEmpty(name) from exc

    def recv(self, name, *, timeout):
        self._check_readable(name)
        deadline = time.monotonic() + timeout
        message = f"Channel {name!r} receive deadline exceeded."
        # Synchronize entry to the transport with expiration in the caller.
        gate = threading.Lock()
        may_have_dequeued = False

        def receive():
            nonlocal may_have_dequeued
            while True:
                with gate:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(message)
                    may_have_dequeued = True
                ref = self._readers[name].take_before.remote(deadline)
                found, item = ray.get(ref, timeout=max(0, deadline - time.monotonic()))
                if found:
                    return item
                with gate:
                    may_have_dequeued = False
                time.sleep(max(0, min(0.005, deadline - time.monotonic())))

        # Bound submission and deserialization as well as the transport wait.
        try:
            return call_until(receive, deadline, message)
        except TimeoutError as exc:
            with gate:
                if may_have_dequeued:
                    self._poisoned.add(name)
            raise TimeoutError(message) from exc
