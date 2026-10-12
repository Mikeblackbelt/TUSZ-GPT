"""Hardware detection, multi-GPU/device support, and GPU memory usage control."""
from __future__ import annotations

import argparse
import logging
from typing import Optional

import torch

from eeg_gpt import hardware

log = logging.getLogger("eeg_gpt.device")


def parse_gpu_usage(val: str) -> float:
    """Parse and validate --max-gpu-usage as a float in [0.0, 1.0]."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"Invalid float value for --max-gpu-usage: {val}")
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError(f"--max-gpu-usage must be between 0.0 and 1.0 inclusive, got {v}")
    return v


def get_device(requested_device: Optional[str] = None) -> torch.device:
    """Return the requested device or auto-detect the best available GPU/device.

    Supports NVIDIA (CUDA), Apple Silicon (MPS), AMD (ROCm via CUDA), Intel (XPU),
    with graceful fallback to CPU.
    """
    if requested_device is not None:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def raise_open_file_limit() -> None:
    """Lift the soft RLIMIT_NOFILE to the hard limit (POSIX only).

    DataLoader workers cost several descriptors each (pipes + one per shared
    tensor in flight); the usual soft limit of 1024 is easy to exhaust.
    """
    try:
        import resource
    except ImportError:  # Windows
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == hard:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        log.info(f"Raised open-file limit {soft} -> {hard}")
    except (ValueError, OSError) as e:
        log.warning(f"Could not raise open-file limit from {soft}: {e}")


def shutdown_loader(loader) -> None:
    """Stop a DataLoader's persistent workers now instead of waiting for garbage collection."""
    it = getattr(loader, "_iterator", None)
    if it is not None and hasattr(it, "_shutdown_workers"):
        it._shutdown_workers()
    if loader is not None:
        loader._iterator = None


_SPEC_LOGGED = False


def adapt_num_workers(requested: Optional[int], n_loaders: int = 2) -> int:
    """--num-workers bounded by the cores and open files this process really has (None = pick for me)."""
    n = hardware.recommend_num_workers(requested, n_loaders)
    if requested is not None and n != requested:
        log.info(f"num_workers {requested} -> {n} (usable cpus {hardware.usable_cpus()}, open files {hardware.open_file_limit()})")
    return n


def setup_device_and_hardware(
    device: torch.device,
    max_gpu_usage: Optional[float] = None,
) -> None:
    """Configure hardware acceleration and maximum GPU usage limit."""
    raise_open_file_limit()
    global _SPEC_LOGGED
    if not _SPEC_LOGGED:        # the sweep calls this once per patient: detect and report once
        _SPEC_LOGGED = True
        hardware.detect.cache_clear()       # the open-file limit may just have changed
        spec = hardware.detect(str(device))
        log.info(f"Hardware: {hardware.describe(spec)} | cache budget/process {hardware.cache_budget_bytes() / hardware.GB:.1f}G")
        # torch sizes its intra-op pool from the node's cores, not from the ones this job was given
        if torch.get_num_threads() > spec["cpus"]:
            torch.set_num_threads(spec["cpus"])
    # 1. Hardware acceleration flags on CUDA (Tensor Cores / cuDNN auto-tuning)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 2. Maximum GPU usage constraint
    if max_gpu_usage is not None:
        if not (0.0 <= max_gpu_usage <= 1.0):
            raise ValueError(f"--max-gpu-usage must be between 0.0 and 1.0 inclusive, got {max_gpu_usage}")
        if device.type == "cuda" and torch.cuda.is_available():
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.set_per_process_memory_fraction(float(max_gpu_usage), device=dev_idx)
            log.info(f"Set CUDA device {dev_idx} max memory fraction to {max_gpu_usage:.1%}")
        elif device.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "set_per_process_memory_fraction"):
            dev_idx = device.index if device.index is not None else torch.xpu.current_device()
            torch.xpu.set_per_process_memory_fraction(float(max_gpu_usage), device=dev_idx)
            log.info(f"Set XPU device {dev_idx} max memory fraction to {max_gpu_usage:.1%}")
        elif device.type == "mps":
            log.info(f"--max-gpu-usage ({max_gpu_usage:.1%}) recorded for MPS device")
        else:
            log.info(f"--max-gpu-usage ({max_gpu_usage:.1%}) recorded for {device}")


def resolve_amp(flag: Optional[bool], device: torch.device) -> bool:
    """--amp / --no-amp; unset means mixed precision on CUDA and full precision elsewhere."""
    return device.type == "cuda" if flag is None else bool(flag)


def _use_bf16(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def get_amp_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    """Create a GradScaler compatible with the device type (bfloat16 needs no loss scaling)."""
    scaler_device = device.type if device.type in ("cuda", "xpu") else "cuda"
    scaler_enabled = enabled and (device.type in ("cuda", "xpu")) and not _use_bf16(device)
    return torch.amp.GradScaler(scaler_device, enabled=scaler_enabled)


def get_autocast_kwargs(device: torch.device, enabled: bool) -> dict:
    """Get kwargs for torch.autocast compatible with the device type."""
    dev_type = device.type if device.type in ("cuda", "xpu", "mps", "cpu") else "cuda"
    dev_enabled = enabled and (device.type in ("cuda", "xpu", "mps", "cpu"))
    kw = {"device_type": dev_type, "enabled": dev_enabled}
    if dev_enabled and _use_bf16(device):
        kw["dtype"] = torch.bfloat16
    return kw


def make_adamw(param_groups, device: torch.device, **kw) -> torch.optim.AdamW:
    """AdamW with the fused CUDA kernel where available (one kernel per step instead of per tensor)."""
    return torch.optim.AdamW(param_groups, fused=device.type == "cuda", **kw)
