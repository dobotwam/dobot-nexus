# RLinf scheduler attribution

This backend uses the Cluster, Worker, WorkerGroup, Channel and placement
strategies provided by the RLinf scheduler at revision
`7e90641c4135229db6d3426950cdb45cee6259a6`. The single-in-flight sync
coordinator follows RLinf's AsyncWeightSyncMixin coalescing behavior.
[RLinf source at the pinned revision](https://github.com/RLinf/RLinf/tree/7e90641c4135229db6d3426950cdb45cee6259a6)

RLinf is copyright 2025–2026 The RLinf Authors and licensed under the
Apache License, Version 2.0. See the accompanying LICENSE.
