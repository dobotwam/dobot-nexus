"""Bounded graph replay for the immutable frozen history vision model."""

import threading
from collections import OrderedDict

import torch

from dobot_nexus.models.pi05.torch_core.numerics import pool_siglip_tokens

from .engine import _signature


class FrozenVisionEngine:
    def __init__(self, model, *, cuda_graph=False, max_graphs=2):
        if max_graphs < 1:
            raise ValueError("Memory vision graph capacity must be positive.")
        self.model = model
        self.cuda_graph = cuda_graph
        self.max_graphs = max_graphs
        self._graphs = OrderedDict()
        self._lock = threading.RLock()
        self.capture_count = 0

    @property
    def metadata(self):
        return {"cuda_graph": self.cuda_graph, "cuda_graph_entries": len(self._graphs),
                "cuda_graph_captures": self.capture_count}

    def _encode(self, images):
        tokens, _ = self.model(images)
        return pool_siglip_tokens(tokens).float()

    @torch.inference_mode()
    def __call__(self, images):
        if not self.cuda_graph:
            return self._encode(images)
        if images.device.type != "cuda":
            raise ValueError("Memory vision CUDA Graph requires a CUDA device.")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Memory vision CUDA Graph requires a frozen eval model.")
        with self._lock, torch.cuda.device(images.device):
            key = _signature(images)
            entry = self._graphs.get(key)
            if entry is None:
                if len(self._graphs) >= self.max_graphs:
                    _, evicted = self._graphs.popitem(last=False)
                    evicted[1].synchronize()
                    del evicted
                # A callable (rather than a Module) avoids replacing model.forward
                # when another batch shape gets its own independent graph.
                replay = torch.cuda.make_graphed_callables(self._encode, (images.clone(),))
                ready = torch.cuda.Event()
                ready.record()
                entry = (replay, ready)
                self._graphs[key] = entry
                self.capture_count += 1
            self._graphs.move_to_end(key)
            replay, ready = entry
            stream = torch.cuda.current_stream(images.device)
            stream.wait_event(ready)
            # Do not expose graph storage: callers may retain a result after replay.
            result = replay(images).clone()
            ready.record(stream)
            return result
