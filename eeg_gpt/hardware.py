"""Machine specs (CPU / RAM / GPU / limits) and the settings derived from them.

Everything here answers "what does this process actually get", which on a
cluster is less than what the node has: CPU affinity and SLURM allocations
bound the cores, the cgroup bounds the memory. Used by training/device.py
(logging, worker counts) and data/datasets.py (cache budget, reader threads).

Overrides:  EEG_GPT_CACHE_GB   total RAM the recording caches may use (0 = count-based LRU only)
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Optional

log = logging.getLogger("eeg_gpt.hardware")

GB = 1024 ** 3
CACHE_RAM_FRACTION = 0.5     # share of the RAM available at startup that recording caches may fill
_FDS_PER_WORKER = 8          # pipes + shared tensors in flight, per DataLoader worker


def usable_cpus() -> int:
    """Cores this process may run on (affinity / SLURM allocation, not the node total)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    for var in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        try:
            v = int(os.environ.get(var, ""))
        except ValueError:
            continue
        if v > 0:
            n = min(n, v)
    return max(1, n)


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as f:
            s = f.read().strip()
        return None if s == "max" else int(s)
    except (OSError, ValueError):
        return None


def _cgroup_memory() -> tuple[Optional[int], Optional[int]]:
    """(limit, usage) in bytes of the tightest enclosing cgroup, None where unlimited/unknown."""
    try:
        with open("/proc/self/cgroup") as f:
            lines = f.read().splitlines()
    except OSError:
        return None, None
    best: tuple[Optional[int], Optional[int]] = (None, None)
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, ctrl, path = parts
        if ctrl == "":                                   # cgroup v2
            base, lim_f, use_f = "/sys/fs/cgroup", "memory.max", "memory.current"
        elif "memory" in ctrl.split(","):                # cgroup v1
            base, lim_f, use_f = "/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes"
        else:
            continue
        path = path.strip("/")
        while True:
            d = os.path.join(base, path) if path else base
            lim = _read_int(os.path.join(d, lim_f))
            if lim is not None and lim < (1 << 60) and (best[0] is None or lim < best[0]):
                best = (lim, _read_int(os.path.join(d, use_f)))
            if not path:
                break
            path = os.path.dirname(path)
    return best


def memory_bytes() -> tuple[Optional[int], Optional[int]]:
    """(total, available) RAM for this process, cgroup limit included. None if it cannot be determined."""
    total = avail = None
    try:
        import psutil
        vm = psutil.virtual_memory()
        total, avail = int(vm.total), int(vm.available)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                info = {ln.split(":")[0]: int(ln.split()[1]) * 1024 for ln in f if ":" in ln}
            total, avail = info.get("MemTotal"), info.get("MemAvailable", info.get("MemFree"))
        except (OSError, ValueError, IndexError):
            pass
    if total is None and os.name == "nt":
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                                      "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
            ms = _MS(); ms.dwLength = ctypes.sizeof(_MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                total, avail = int(ms.ullTotalPhys), int(ms.ullAvailPhys)
        except Exception:
            pass
    if total is None:
        try:                                             # macOS / other POSIX: no cheap "available" figure
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            avail = total // 2
        except (AttributeError, ValueError, OSError):
            return None, None
    lim, used = _cgroup_memory()
    if lim is not None and lim < total:
        total = lim
        avail = min(avail if avail is not None else lim, max(0, lim - (used or 0)))
    return total, avail


def open_file_limit() -> Optional[int]:
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        return None if soft == resource.RLIM_INFINITY else int(soft)
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def detect(device_str: Optional[str] = None) -> dict:
    """Specs of this machine as seen by this process (cached; RAM figures are from the first call)."""
    total, avail = memory_bytes()
    spec = {"cpus": usable_cpus(), "cpus_node": os.cpu_count() or 1, "ram_total": total, "ram_available": avail,
            "open_files": open_file_limit(), "gpu": None, "gpu_mem": None, "gpu_count": 0}
    try:
        import torch
        if torch.cuda.is_available() and (device_str is None or device_str.startswith("cuda")):
            idx = torch.device(device_str).index if device_str else None
            idx = torch.cuda.current_device() if idx is None else idx
            spec.update(gpu=torch.cuda.get_device_name(idx), gpu_mem=int(torch.cuda.get_device_properties(idx).total_memory),
                        gpu_count=torch.cuda.device_count())
    except Exception:
        pass
    return spec


def describe(spec: dict) -> str:
    g = lambda b: "?" if b is None else f"{b / GB:.1f}G"  # noqa: E731
    s = f"cpus {spec['cpus']}/{spec['cpus_node']} | ram {g(spec['ram_available'])} free of {g(spec['ram_total'])} | open files {spec['open_files'] or 'unlimited'}"
    if spec["gpu"]:
        s += f" | gpu {spec['gpu']} {g(spec['gpu_mem'])} x{spec['gpu_count']}"
    return s


def recommend_num_workers(requested: Optional[int] = None, n_loaders: int = 2) -> int:
    """DataLoader workers per loader that this machine can carry.

    `requested` (the user's --num-workers) is only ever lowered: to the cores
    left beside the training process, and to what the open-file limit allows
    for `n_loaders` loaders alive at once. None asks for the best value.
    """
    spec = detect()
    cap = max(0, spec["cpus"] - 1)
    if spec["open_files"]:
        cap = min(cap, max(0, (spec["open_files"] - 256) // (_FDS_PER_WORKER * max(1, n_loaders))))
    return min(cap, 16) if requested is None else max(0, min(int(requested), cap))


def cache_budget_bytes(n_processes: int = 1) -> int:
    """RAM one process may fill with decoded recordings, on top of the cache's minimum capacity.

    0 means "unknown / disabled": the caches then fall back to their plain count-based LRU.
    """
    env = os.environ.get("EEG_GPT_CACHE_GB")
    if env is not None:
        try:
            total = float(env) * GB
        except ValueError:
            total = 0.0
    else:
        _, avail = memory_bytes()
        total = CACHE_RAM_FRACTION * avail if avail else 0.0
    # the main process and the train and eval loaders' workers all hold caches at the same time
    return int(total / max(1, n_processes) / 3)


def reader_threads(n_processes: int = 1) -> int:
    """Threads a parquet read may use in one of `n_processes` concurrently reading processes."""
    return max(1, usable_cpus() // max(1, n_processes))
