from __future__ import annotations

from dobot_nexus.modules.memory import sampling as _shared

MemorySnapshot = _shared.MemorySnapshot
assemble_memory = _shared.assemble_memory
np = _shared.np
position_embeddings = _shared.position_embeddings


def select_frames(current, count, *, segment_start=0, window_frames=None, single_frame="current"):
    return _shared.select_frames(
        current, count, segment_start=segment_start, window_frames=window_frames,
        single_frame=single_frame,
    )
