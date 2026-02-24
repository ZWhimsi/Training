"""
Day 3: Masking and Bounds Checking
==================================
Estimated time: 1-2 hours
Prerequisites: Day 2 (memory operations)

Learning objectives:
- Understand why masking is essential in Triton
- Handle edge cases where data doesn't fit block size
- Use masks effectively for conditional operations
- Avoid out-of-bounds memory access
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Why Masking?
# ============================================================================
# When processing arrays, the last block often has fewer elements than BLOCK_SIZE.
# Without masking, we would access invalid memory!
#
# Example: 1000 elements with BLOCK_SIZE=256
# - Block 0: elements 0-255 (full)
# - Block 1: elements 256-511 (full)
# - Block 2: elements 512-767 (full)
# - Block 3: elements 768-999 (ONLY 232 elements, not 256!)
#
# The mask prevents accessing elements 1000-1023 which don't exist.
# ============================================================================


# ============================================================================
# Exercise 1: Safe Load with Default Value
# ============================================================================
# Load with a default value for out-of-bounds elements

@triton.jit
def safe_load_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    default_value,  # Value to use for out-of-bounds
    BLOCK_SIZE: tl.constexpr,
):
    """
    Load elements, using default_value for out-of-bounds positions.
    
    The 'other' parameter in tl.load() specifies what value to use
    when the mask is False.
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=default) -> load with default for masked
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement safe load kernel
    pass


def safe_load(x: torch.Tensor, default_value: float = 0.0) -> torch.Tensor:
    """Safe load with default value."""
    assert x.is_cuda
    n = x.numel()
    BLOCK_SIZE = 256
    
    # Output is padded to multiple of BLOCK_SIZE
    n_padded = triton.cdiv(n, BLOCK_SIZE) * BLOCK_SIZE
    out = torch.empty(n_padded, dtype=x.dtype, device=x.device)
    
    # API hints:
    # - triton.cdiv(n, d) -> ceiling division
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Exercise 2: Masked Conditional Operations
# ============================================================================
# Apply operation only to elements satisfying a condition

@triton.jit
def threshold_kernel(
    x_ptr,
    out_ptr,
    threshold,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Set elements below threshold to 0.
    out[i] = x[i] if x[i] >= threshold else 0
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.where(condition, val_true, val_false) -> conditional select
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement threshold kernel
    pass


def threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply threshold."""
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
# Exercise 3: Clamp Operation
# ============================================================================
# Clamp values to a range [min_val, max_val]

@triton.jit
def clamp_kernel(
    x_ptr,
    out_ptr,
    min_val,
    max_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Clamp values: out[i] = clamp(x[i], min_val, max_val)
    
    Use tl.minimum and tl.maximum for efficient clamping.
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.minimum(a, b) -> element-wise minimum
    # - tl.maximum(a, b) -> element-wise maximum
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement clamp kernel
    pass


def clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """Clamp values to range."""
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
# Exercise 4: Masked Accumulation
# ============================================================================
# Only add values that satisfy a condition

@triton.jit
def positive_sum_kernel(
    x_ptr,
    out_ptr,  # Single element output
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sum only positive elements.
    
    This kernel processes one block and outputs a partial sum.
    The full reduction would require multiple kernel launches or atomics.
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.where(condition, val_true, val_false) -> conditional select
    # - tl.sum(x, axis=0) -> sum reduction
    # - tl.atomic_add(ptr, value) -> atomic addition
    
    # TODO: Implement positive sum kernel
    pass


def positive_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum positive elements only."""
    assert x.is_cuda
    out = torch.zeros(1, dtype=x.dtype, device=x.device)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # API hints:
    # - triton.cdiv(n, d) -> ceiling division
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Exercise 5: Where Operation (Ternary Select)
# ============================================================================
# Select from two arrays based on condition

@triton.jit
def where_kernel(
    cond_ptr,   # Boolean condition (stored as int)
    a_ptr,      # Values if true
    b_ptr,      # Values if false
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    out[i] = a[i] if cond[i] else b[i]
    
    This is like torch.where(condition, a, b)
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.where(condition, val_true, val_false) -> conditional select
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement where kernel
    pass


def where(cond: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Conditional select."""
    assert cond.is_cuda and a.is_cuda and b.is_cuda
    out = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Convert bool to int for Triton
    cond_int = cond.int()
    
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
    print("Day 3: Masking and Bounds Checking")
    print("=" * 50)
    print("Run test_day03.py to check your implementations!")
