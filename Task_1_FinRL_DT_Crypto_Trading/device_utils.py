#!/usr/bin/env python3
"""
Device utilities for optimal hardware selection on macOS and other platforms.
Supports CUDA, MPS (Apple Silicon), and CPU fallback.
"""

import torch
import platform
import warnings
from typing import Optional


def get_optimal_device(gpu_id: int = -1, verbose: bool = True) -> torch.device:
    """
    Get the optimal device for PyTorch operations.
    
    Priority order:
    1. CUDA (if available and gpu_id >= 0)
    2. MPS (Apple Silicon Metal Performance Shaders)
    3. CPU (fallback)
    
    Args:
        gpu_id: GPU ID for CUDA (-1 for CPU/MPS)
        verbose: Whether to print device selection info
        
    Returns:
        torch.device: The selected device
    """
    device_name = "cpu"
    reason = ""
    
    # Check CUDA first (if gpu_id is specified and CUDA is available)
    if gpu_id >= 0 and torch.cuda.is_available():
        device_name = f"cuda:{gpu_id}"
        reason = f"CUDA GPU {gpu_id} available"
    
    # Check MPS (Apple Silicon) if CUDA not available or gpu_id < 0
    elif torch.backends.mps.is_available() and platform.system() == "Darwin":
        device_name = "mps"
        reason = "Apple Silicon MPS available"
        
        # Warn about MPS limitations
        if verbose:
            warnings.warn(
                "Using MPS (Metal Performance Shaders) on Apple Silicon. "
                "Note: Some operations may be slower than CUDA, and memory usage patterns differ. "
                "Consider using smaller batch sizes if you encounter memory issues.",
                UserWarning
            )
    
    # Fallback to CPU
    else:
        device_name = "cpu"
        if torch.cuda.is_available():
            reason = f"CUDA available but gpu_id={gpu_id} < 0, using CPU"
        elif platform.system() == "Darwin" and not torch.backends.mps.is_available():
            reason = "macOS detected but MPS not available, using CPU"
        else:
            reason = "No GPU available, using CPU"
    
    device = torch.device(device_name)
    
    if verbose:
        print(f"🔧 Device selected: {device} ({reason})")
        if device.type == "mps":
            print("   💡 MPS Tips:")
            print("   - Use smaller batch sizes if you encounter memory issues")
            print("   - Some operations may be slower than CUDA")
            print("   - Monitor memory usage with Activity Monitor")
    
    return device


def get_device_info() -> dict:
    """
    Get comprehensive device information.
    
    Returns:
        dict: Device information including capabilities and memory
    """
    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            info[f"cuda_device_{i}_name"] = torch.cuda.get_device_name(i)
            info[f"cuda_device_{i}_memory"] = torch.cuda.get_device_properties(i).total_memory
    
    if torch.backends.mps.is_available():
        info["mps_built"] = torch.backends.mps.is_built()
    
    return info


def print_device_info():
    """Print detailed device information."""
    info = get_device_info()
    
    print("🖥️  Device Information:")
    print(f"   Platform: {info['platform']} ({info['architecture']})")
    print(f"   Python: {info['python_version']}")
    print(f"   PyTorch: {info['pytorch_version']}")
    print(f"   CPU Threads: {info['cpu_count']}")
    
    print(f"\n🚀 GPU Support:")
    print(f"   CUDA Available: {info['cuda_available']}")
    if info['cuda_available']:
        print(f"   CUDA Version: {info['cuda_version']}")
        print(f"   CUDA Devices: {info['cuda_device_count']}")
        for i in range(info['cuda_device_count']):
            memory_gb = info[f'cuda_device_{i}_memory'] / (1024**3)
            print(f"     Device {i}: {info[f'cuda_device_{i}_name']} ({memory_gb:.1f} GB)")
    
    print(f"   MPS Available: {info['mps_available']}")
    if info['mps_available']:
        print(f"   MPS Built: {info['mps_built']}")
        print("     💡 MPS provides GPU acceleration on Apple Silicon")


def optimize_for_device(device: torch.device) -> None:
    """
    Apply device-specific optimizations.
    
    Args:
        device: The selected device
    """
    if device.type == "mps":
        # MPS-specific optimizations
        torch.backends.mps.empty_cache()
        # Set memory fraction if needed (MPS doesn't support this directly)
        print("   🔧 Applied MPS optimizations")
    
    elif device.type == "cuda":
        # CUDA-specific optimizations
        torch.cuda.empty_cache()
        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        print("   🔧 Applied CUDA optimizations")
    
    else:
        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())
        print("   🔧 Applied CPU optimizations")


# Convenience function for easy import
def get_device(gpu_id: int = -1, verbose: bool = True) -> torch.device:
    """Convenience function that returns optimal device and applies optimizations."""
    device = get_optimal_device(gpu_id, verbose)
    optimize_for_device(device)
    return device


if __name__ == "__main__":
    print_device_info()
    print("\n" + "="*50)
    device = get_device(verbose=True)
    print(f"\n✅ Selected device: {device}")