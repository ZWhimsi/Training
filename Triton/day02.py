"""
Day 2: Memory Operations in Triton
==================================
Estimated time: 1-2 hours
Prerequisites: Day 1 (basic kernel structure)

Learning objectives:
- Master tl.load() and tl.store() operations
- Understand pointer arithmetic in Triton
- Work with different data types
- Handle memory alignment and coalescing basics
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Memory Operations
# ============================================================================
# GPU memory operations are the foundation of kernel programming.
# 
# tl.load(pointer, mask, other):
#   - pointer: where to read from
#   - mask: which elements to actually load (for bounds checking)
#   - other: default value for masked-out elements
#
# tl.store(pointer, value, mask):
#   - pointer: where to write to
#   - value: what to write
#   - mask: which elements to actually store
# ============================================================================


# ============================================================================
# Exercise 1: Copy Kernel
# ============================================================================
# Write a kernel that copies data from one array to another.
# This is the simplest memory operation pattern.

@triton.jit
def copy_kernel(
    src_ptr,    # Source pointer
    dst_ptr,    # Destination pointer
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Copy elements from src to dst.
    
    This teaches the basic load/store pattern.
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range [start, end)
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement copy kernel
    pass


def copy(src: torch.Tensor) -> torch.Tensor:
    """Copy tensor using Triton kernel."""
    assert src.is_cuda
    dst = torch.empty_like(src)
    n_elements = src.numel()
    BLOCK_SIZE = 1024
    
    # API hints:
    # - triton.cdiv(n, d) -> ceiling division
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Calculate grid and launch kernel
    pass
    
    return dst


# ============================================================================
# Exercise 2: Scaled Copy
# ============================================================================
# Copy with scaling: dst[i] = src[i] * scale

@triton.jit
def scaled_copy_kernel(
    src_ptr,
    dst_ptr,
    scale,      # Scalar multiplier
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Copy with scaling: dst = src * scale
    
    Same pattern as copy, but multiply by scale before storing.
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement scaled copy kernel
    pass


def scaled_copy(src: torch.Tensor, scale: float) -> torch.Tensor:
    """Copy with scaling using Triton kernel."""
    assert src.is_cuda
    dst = torch.empty_like(src)
    n_elements = src.numel()
    BLOCK_SIZE = 1024
    
    # API hints:
    # - triton.cdiv(n, d) -> ceiling division
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel with scale parameter
    pass
    
    return dst


# ============================================================================
# Exercise 3: Strided Load
# ============================================================================
# Load every Nth element (strided access pattern)

@triton.jit
def strided_load_kernel(
    src_ptr,
    dst_ptr,
    stride,         # Load every 'stride' elements
    n_output,       # Number of output elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Load every stride-th element from source.
    
    If stride=2: dst[0]=src[0], dst[1]=src[2], dst[2]=src[4], ...
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement strided load kernel
    pass


def strided_load(src: torch.Tensor, stride: int) -> torch.Tensor:
    """Load every stride-th element."""
    assert src.is_cuda
    n_output = src.numel() // stride
    dst = torch.empty(n_output, dtype=src.dtype, device=src.device)
    BLOCK_SIZE = 1024
    
    # API hints:
    # - triton.cdiv(n, d) -> ceiling division
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return dst


# ============================================================================
# Exercise 4: Conditional Store
# ============================================================================
# Only store elements that satisfy a condition

@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    ReLU activation: out[i] = max(0, x[i])
    
    This uses tl.where() or tl.maximum() for conditional operations.
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.maximum(a, b) -> element-wise maximum
    # - tl.where(cond, a, b) -> select a where cond else b
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement ReLU kernel
    pass


def relu(x: torch.Tensor) -> torch.Tensor:
    """Apply ReLU using Triton kernel."""
    assert x.is_cuda
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # API hints:
    # - triton.cdiv(n, d) -> ceiling division
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Exercise 5: Fused Add-ReLU
# ============================================================================
# Combine two operations in one kernel (kernel fusion)

@triton.jit
def add_relu_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused add + ReLU: out = max(0, a + b)
    
    Kernel fusion reduces memory traffic by avoiding intermediate storage.
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.maximum(a, b) -> element-wise maximum
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement fused add-ReLU kernel
    pass


def add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused add + ReLU."""
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    out = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # API hints:
    # - triton.cdiv(n, d) -> ceiling division
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 2: Memory Operations in Triton")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print("\nRun test_day02.py to check your implementations!")
