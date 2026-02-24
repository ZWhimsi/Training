"""
Day 5: Block-Level Programming
==============================
Estimated time: 1-2 hours
Prerequisites: Day 4 (program IDs)

Learning objectives:
- Understand block-level parallelism in depth
- Work with multiple block size configurations
- Use tl.constexpr for compile-time constants
- Optimize block sizes for different operations
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Exercise 1: Parameterized Block Sizes
# ============================================================================
# Experiment with different block sizes

@triton.jit
def add_vectors_configurable(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vector addition with configurable block size.
    Try launching with different BLOCK_SIZE values!
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement vector add kernel
    pass


def add_vectors(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """Add vectors with specified block size."""
    assert a.is_cuda and b.is_cuda
    out = torch.empty_like(a)
    n_elements = a.numel()
    
    grid = (triton.cdiv(n_elements, block_size),)
    
    # API hints:
    # - kernel_name[grid](args..., BLOCK_SIZE=value) -> launch with constexpr
    
    # TODO: Launch with specified block_size
    pass
    
    return out


# ============================================================================
# Exercise 2: Multi-Block Reduction
# ============================================================================
# Two-phase reduction: local then global

@triton.jit
def partial_sum_kernel(
    x_ptr,
    partial_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 1: Each block computes a partial sum.
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.sum(x, axis=0) -> sum reduction
    # - tl.store(ptr, value) -> store single value
    
    # TODO: Implement partial sum kernel
    pass


@triton.jit
def final_sum_kernel(
    partial_ptr,
    out_ptr,
    n_partials,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Phase 2: Sum all partial sums.
    """
    # API hints:
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.sum(x, axis=0) -> sum reduction
    # - tl.store(ptr, value) -> store single value
    
    # TODO: Implement final sum kernel
    pass


def two_phase_sum(x: torch.Tensor) -> torch.Tensor:
    """Two-phase reduction for sum."""
    assert x.is_cuda
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_sums = torch.empty(n_blocks, dtype=x.dtype, device=x.device)
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    # - triton.next_power_of_2(n) -> next power of 2
    
    # Phase 1: Partial sums
    # TODO: Launch partial_sum_kernel
    pass
    
    # Phase 2: Final sum
    out = torch.empty(1, dtype=x.dtype, device=x.device)
    # TODO: Launch final_sum_kernel
    pass
    
    return out


# ============================================================================
# Exercise 3: Block-Tiled Matrix Transpose
# ============================================================================
# Use blocks to efficiently transpose a matrix

@triton.jit
def transpose_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    stride_in_m,
    stride_out_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Transpose a matrix using block tiling.
    
    Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    """
    # API hints:
    # - tl.program_id(0), tl.program_id(1) -> 2D block indices
    # - tl.arange(start, end) -> create range
    # - offs[:, None] * stride + offs[None, :] -> 2D indices
    # - mask[:, None] & mask[None, :] -> 2D mask
    # - tl.load(ptr + indices, mask=mask) -> load 2D block
    # - tl.trans(block) -> transpose (or compute transposed indices)
    # - tl.store(ptr + indices, value, mask=mask) -> store 2D block
    
    # TODO: Implement transpose kernel
    pass


def transpose(x: torch.Tensor) -> torch.Tensor:
    """Transpose a 2D matrix."""
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty(N, M, dtype=x.dtype, device=x.device)
    
    BLOCK_M = 32
    BLOCK_N = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Exercise 4: Block-wise Maximum
# ============================================================================
# Find max element using blocks

@triton.jit  
def block_max_kernel(
    x_ptr,
    partial_max_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Find max within each block.
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=-float('inf')) -> load with -inf default
    # - tl.max(x, axis=0) -> max reduction
    # - tl.store(ptr, value) -> store single value
    
    # TODO: Implement block max kernel
    pass


def find_max(x: torch.Tensor) -> torch.Tensor:
    """Find maximum element."""
    assert x.is_cuda
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_maxes = torch.empty(n_blocks, dtype=x.dtype, device=x.device)
    
    grid = (n_blocks,)
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch block_max_kernel
    pass
    
    # Use PyTorch for final max (small tensor)
    return partial_maxes.max().unsqueeze(0)


# ============================================================================
# Exercise 5: Softmax Numerator (Block-Level)
# ============================================================================
# First step of softmax: exp(x - max(x))

@triton.jit
def softmax_numerator_kernel(
    x_ptr,
    out_ptr,
    row_max_ptr,  # Pre-computed row maxes
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute exp(x - max) for one row.
    Grid: (n_rows,)
    """
    # API hints:
    # - tl.program_id(axis) -> get row index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr) -> load single scalar
    # - tl.load(ptr + offsets, mask=mask) -> load row
    # - tl.exp(x) -> element-wise exponential
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement softmax numerator kernel
    pass


def softmax_numerator(x: torch.Tensor, row_maxes: torch.Tensor) -> torch.Tensor:
    """Compute softmax numerator: exp(x - max)."""
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 5: Block-Level Programming")
    print("=" * 50)
    print("Run test_day05.py to check your implementations!")
