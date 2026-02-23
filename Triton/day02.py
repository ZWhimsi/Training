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

Hints:
- Pointers in Triton are like array indices
- tl.load(ptr + offset) loads from memory at ptr + offset
- Always use masks for bounds checking
- Data types matter: float32, float16, int32, etc.

Resources:
- https://triton-lang.org/main/python-api/triton.language.html
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
    # TODO: Get program ID
    pid = None  # Replace
    
    # TODO: Calculate offsets for this block
    # HINT: offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets = None  # Replace
    
    # TODO: Create mask for bounds checking
    mask = None  # Replace
    
    # TODO: Load from source
    # HINT: tl.load(src_ptr + offsets, mask=mask)
    data = None  # Replace
    
    # TODO: Store to destination
    # HINT: tl.store(dst_ptr + offsets, data, mask=mask)
    pass  # Replace


def copy(src: torch.Tensor) -> torch.Tensor:
    """Copy tensor using Triton kernel."""
    assert src.is_cuda
    dst = torch.empty_like(src)
    n_elements = src.numel()
    BLOCK_SIZE = 1024
    
    # TODO: Calculate grid and launch kernel
    grid = None  # Replace
    # Launch kernel
    
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
    """
    # TODO: Implement scaled copy
    # Same pattern as copy, but multiply by scale before storing
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load, scale, store
    pass


def scaled_copy(src: torch.Tensor, scale: float) -> torch.Tensor:
    """Copy with scaling using Triton kernel."""
    assert src.is_cuda
    dst = torch.empty_like(src)
    n_elements = src.numel()
    BLOCK_SIZE = 1024
    
    # TODO: Launch kernel with scale parameter
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    # Launch kernel
    
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
    pid = tl.program_id(0)
    
    # Output indices (where we write)
    out_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = out_offsets < n_output
    
    # TODO: Calculate source indices (multiply by stride)
    # HINT: src_offsets = out_offsets * stride
    src_offsets = None  # Replace
    
    # TODO: Load from strided positions
    data = None  # Replace
    
    # TODO: Store to contiguous output
    pass


def strided_load(src: torch.Tensor, stride: int) -> torch.Tensor:
    """Load every stride-th element."""
    assert src.is_cuda
    n_output = src.numel() // stride
    dst = torch.empty(n_output, dtype=src.dtype, device=src.device)
    BLOCK_SIZE = 1024
    
    # TODO: Launch kernel
    grid = (triton.cdiv(n_output, BLOCK_SIZE),)
    # Launch kernel
    
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
    
    This uses tl.where() for conditional operations.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load input
    x = None  # Replace
    
    # TODO: Apply ReLU using tl.where or tl.maximum
    # HINT: tl.where(condition, value_if_true, value_if_false)
    # HINT: Or use tl.maximum(x, 0.0)
    output = None  # Replace
    
    # TODO: Store result
    pass


def relu(x: torch.Tensor) -> torch.Tensor:
    """Apply ReLU using Triton kernel."""
    assert x.is_cuda
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # TODO: Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    # Launch kernel
    
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
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load both inputs
    a = None  # Replace
    b = None  # Replace
    
    # TODO: Add them
    sum_ab = None  # Replace
    
    # TODO: Apply ReLU
    output = None  # Replace
    
    # TODO: Store
    pass


def add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused add + ReLU."""
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    out = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # TODO: Launch kernel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    # Launch kernel
    
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
