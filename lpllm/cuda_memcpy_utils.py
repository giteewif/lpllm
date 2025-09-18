"""
CUDA Memory Copy Utilities

This module provides high-level Python interfaces for CUDA memory copy operations
using the underlying C++ cudaMemcpy implementation.
"""

import torch
from typing import Optional

try:
    from sllm_store._C import (
        cuda_memcpy_h2d,
        cuda_memcpy_d2h, 
        cuda_memcpy_d2d,
        cuda_memcpy_smart
    )
    CUDA_MEMCPY_AVAILABLE = True
except ImportError:
    CUDA_MEMCPY_AVAILABLE = False


def is_cuda_memcpy_available() -> bool:
    """Check if CUDA memcpy functions are available"""
    return CUDA_MEMCPY_AVAILABLE


def cuda_copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    """
    Copy tensor using cudaMemcpy with automatic direction detection
    
    Args:
        dst: Destination tensor
        src: Source tensor
        non_blocking: Whether to use asynchronous copy
        
    Returns:
        Destination tensor for method chaining
        
    Raises:
        RuntimeError: If copy fails or tensors are incompatible
        ImportError: If CUDA memcpy functions are not available
    """
    if not CUDA_MEMCPY_AVAILABLE:
        # Fallback to PyTorch's copy_
        dst.copy_(src, non_blocking=non_blocking)
        return dst
    
    # Use smart CUDA memcpy
    result = cuda_memcpy_smart(dst, src, non_blocking)
    
    if result != 0:
        error_messages = {
            -1: "CUDA memory copy failed",
            -2: "Tensor size mismatch"
        }
        error_msg = error_messages.get(result, f"Unknown error code: {result}")
        raise RuntimeError(f"CUDA memcpy failed: {error_msg}")
    
    return dst


def cuda_copy_h2d(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    """
    Copy tensor from host to device using cudaMemcpy
    
    Args:
        dst: Destination device tensor
        src: Source host tensor
        non_blocking: Whether to use asynchronous copy
        
    Returns:
        Destination tensor for method chaining
        
    Raises:
        RuntimeError: If copy fails or tensors are incompatible
        ImportError: If CUDA memcpy functions are not available
    """
    if not CUDA_MEMCPY_AVAILABLE:
        dst.copy_(src, non_blocking=non_blocking)
        return dst
    
    result = cuda_memcpy_h2d(dst, src, non_blocking)
    
    if result != 0:
        error_messages = {
            -1: "CUDA H2D memory copy failed",
            -2: "Tensor size mismatch"
        }
        error_msg = error_messages.get(result, f"Unknown error code: {result}")
        raise RuntimeError(f"CUDA H2D memcpy failed: {error_msg}")
    
    return dst


def cuda_copy_d2h(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    """
    Copy tensor from device to host using cudaMemcpy
    
    Args:
        dst: Destination host tensor
        src: Source device tensor
        non_blocking: Whether to use asynchronous copy
        
    Returns:
        Destination tensor for method chaining
        
    Raises:
        RuntimeError: If copy fails or tensors are incompatible
        ImportError: If CUDA memcpy functions are not available
    """
    if not CUDA_MEMCPY_AVAILABLE:
        dst.copy_(src, non_blocking=non_blocking)
        return dst
    
    result = cuda_memcpy_d2h(dst, src, non_blocking)
    
    if result != 0:
        error_messages = {
            -1: "CUDA D2H memory copy failed",
            -2: "Tensor size mismatch"
        }
        error_msg = error_messages.get(result, f"Unknown error code: {result}")
        raise RuntimeError(f"CUDA D2H memcpy failed: {error_msg}")
    
    return dst


def cuda_copy_d2d(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    """
    Copy tensor from device to device using cudaMemcpy
    
    Args:
        dst: Destination device tensor
        src: Source device tensor
        non_blocking: Whether to use asynchronous copy
        
    Returns:
        Destination tensor for method chaining
        
    Raises:
        RuntimeError: If copy fails or tensors are incompatible
        ImportError: If CUDA memcpy functions are not available
    """
    if not CUDA_MEMCPY_AVAILABLE:
        dst.copy_(src, non_blocking=non_blocking)
        return dst
    
    result = cuda_memcpy_d2d(dst, src, non_blocking)
    
    if result != 0:
        error_messages = {
            -1: "CUDA D2D memory copy failed",
            -2: "Tensor size mismatch"
        }
        error_msg = error_messages.get(result, f"Unknown error code: {result}")
        raise RuntimeError(f"CUDA D2D memcpy failed: {error_msg}")
    
    return dst


class CudaMemcpyTensor(torch.Tensor):
    """
    A tensor wrapper that uses cudaMemcpy for copy operations
    """
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        # Intercept copy_ operations
        if func is torch.Tensor.copy_:
            if len(args) >= 2:
                dst, src = args[0], args[1]
                non_blocking = kwargs.get('non_blocking', False)
                return cuda_copy_(dst, src, non_blocking)
        
        # For all other operations, use default behavior
        return super().__torch_function__(func, types, args, kwargs)


def enable_cuda_memcpy_globally():
    """
    Monkey-patch torch.Tensor.copy_ to use CUDA memcpy by default
    
    Warning: This affects all tensor copy operations globally
    """
    if not CUDA_MEMCPY_AVAILABLE:
        print("Warning: CUDA memcpy not available, using PyTorch default")
        return
    
    original_copy = torch.Tensor.copy_
    
    def cuda_copy_wrapper(self, src, non_blocking=False):
        return cuda_copy_(self, src, non_blocking)
    
    torch.Tensor.copy_ = cuda_copy_wrapper
    print("CUDA memcpy enabled globally for all tensor copy operations")


def disable_cuda_memcpy_globally():
    """
    Restore original torch.Tensor.copy_ behavior
    """
    # This would require storing the original function reference
    # For now, just print a warning
    print("Warning: Global CUDA memcpy cannot be easily disabled. Restart Python to restore default behavior.")


# Convenience function for backward compatibility
def safe_copy_(dst: torch.Tensor, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    """
    Safe copy function that falls back to PyTorch's copy_ if CUDA memcpy fails
    """
    try:
        return cuda_copy_(dst, src, non_blocking)
    except Exception as e:
        print(f"CUDA memcpy failed ({e}), falling back to PyTorch copy")
        dst.copy_(src, non_blocking=non_blocking)
        return dst
