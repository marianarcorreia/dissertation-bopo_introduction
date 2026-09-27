"""Efficiency metrics: memory used and inference time.

GPU memory is exact per measured block (torch's peak allocator counter is reset on entry).
CPU memory is the process's resident set size (RSS): the value at the end of the block and
its growth during it. The operating system only keeps a lifetime peak for the process, so
`process_peak_mb` is the highest the whole process has reached so far, not just this block.
"""
import sys
import time

import psutil
import torch


def _mb(n_bytes):
    return float(n_bytes) / 2 ** 20


def process_peak_mb():
    info = psutil.Process().memory_info()
    if sys.platform == "win32":
        return _mb(info.peak_wset)
    import resource  # not available on Windows
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return _mb(peak) if sys.platform == "darwin" else peak / 1024.0  # linux reports KiB


def model_size(module):
    """Parameter count and in-memory size (parameters + buffers) of a model."""
    params = sum(p.numel() for p in module.parameters())
    size = sum(p.numel() * p.element_size() for p in module.parameters())
    size += sum(b.numel() * b.element_size() for b in module.buffers())
    return {"param_count": int(params), "model_size_mb": _mb(size)}


class MemoryTracker:
    """with MemoryTracker() as mem: ...  ->  mem.result"""

    def __enter__(self):
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._cuda_start = torch.cuda.memory_allocated()
        self._rss_start = psutil.Process().memory_info().rss
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._t0
        rss = psutil.Process().memory_info().rss
        self.result = {
            "time_sec": elapsed,
            "rss_mb": _mb(rss),
            "rss_delta_mb": _mb(rss - self._rss_start),
            "process_peak_mb": process_peak_mb(),
            "gpu_peak_mb": _mb(torch.cuda.max_memory_allocated()) if self.cuda else None,
            "gpu_peak_delta_mb": _mb(torch.cuda.max_memory_allocated() - self._cuda_start) if self.cuda else None,
        }
        return False
