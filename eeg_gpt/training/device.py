"""Hardware detection, multi-GPU/device support, and GPU memory usage control."""
from __future__ import annotations

import argparse
import logging
from typing import Optional

import torch

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


def setup_device_and_hardware(
    device: torch.device,
    max_gpu_usage: Optional[float] = None,
) -> None:
    """Configure hardware acceleration and maximum GPU usage limit."""
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


def get_amp_scaler(device: torch.device, enabled: bool) -> torch.amp.GradScaler:
    """Create a GradScaler compatible with the device type."""
    scaler_device = device.type if device.type in ("cuda", "xpu") else "cuda"
    scaler_enabled = enabled and (device.type in ("cuda", "xpu"))
    return torch.amp.GradScaler(scaler_device, enabled=scaler_enabled)


def get_autocast_kwargs(device: torch.device, enabled: bool) -> dict:
    """Get kwargs for torch.autocast compatible with the device type."""
    dev_type = device.type if device.type in ("cuda", "xpu", "mps", "cpu") else "cuda"
    dev_enabled = enabled and (device.type in ("cuda", "xpu", "mps", "cpu"))
    return {"device_type": dev_type, "enabled": dev_enabled}
