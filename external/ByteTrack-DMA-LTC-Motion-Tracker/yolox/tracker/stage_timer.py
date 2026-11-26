"""
Optional per-stage wall-clock profiler for the tracking pipeline.

Only active when a StageTimer is attached AND its `enabled` flag is True
(tools/track_dma.py --profile-timing). With no timer attached, the call sites
in byte_tracker.py / fuse.py / mot_evaluator.py do nothing, so default runs
are unchanged (no extra CUDA synchronisation, no extra work).

When enabled, start()/stop() call torch.cuda.synchronize() so that GPU work
queued by a stage is attributed to that stage. This inflates absolute latency
slightly; use the profiled run only for the per-stage breakdown, and the
unprofiled run for headline FPS.
"""

import time
from collections import defaultdict

try:
    import torch
    _CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch always present in this repo
    torch = None
    _CUDA = False


class StageTimer:
    def __init__(self, sync_cuda=None):
        self.enabled = False
        self.sync_cuda = _CUDA if sync_cuda is None else bool(sync_cuda)
        self.totals = defaultdict(float)   # seconds, summed over recorded frames
        self.counts = defaultdict(int)     # number of start/stop pairs
        self._t0 = {}
        self.frames = 0                    # recorded (post-warmup) frames

    def _sync(self):
        if self.sync_cuda:
            torch.cuda.synchronize()

    def start(self, name):
        if not self.enabled:
            return
        self._sync()
        self._t0[name] = time.perf_counter()

    def stop(self, name):
        if not self.enabled or name not in self._t0:
            return
        self._sync()
        self.totals[name] += time.perf_counter() - self._t0.pop(name)
        self.counts[name] += 1

    def add_frame(self):
        self.frames += 1

    def ms_per_frame(self, name):
        if self.frames == 0:
            return None
        return 1000.0 * self.totals.get(name, 0.0) / self.frames
