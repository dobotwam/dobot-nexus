from __future__ import annotations

from dobot_nexus.modules.memory import runtime as _shared

from .sampling import select_frames as select_frames

MemoryObservationBatch = _shared.MemoryObservationBatch
MemorySnapshot = _shared.MemorySnapshot
ObservationBatch = _shared.ObservationBatch
OrderedDict = _shared.OrderedDict
PolicySession = _shared.PolicySession
RLock = _shared.RLock
SessionOptions = _shared.SessionOptions
Sized = _shared.Sized
_step_index = _shared._step_index
assemble_memory = _shared.assemble_memory
contextmanager = _shared.contextmanager
copy = _shared.copy
islice = _shared.islice
np = _shared.np
replace = _shared.replace


class HistoryBuffer(_shared.HistoryBuffer):
    single_frame = "current"


class MemorySession(_shared.MemorySession):
    _buffer_type = HistoryBuffer
