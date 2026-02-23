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

Hints:
- BLOCK_SIZE must be a power of 2 for many operations
- Use triton.next_power_of_2() for automatic sizing
- tl.constexpr means the value is known at compile time
- Different block sizes suit different problems
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
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load, add, store
    a = None  # Replace
    b = None  # Replace
    out = None  # Replace
    pass


def add_vectors(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """Add vectors with specified block size."""
    assert a.is_cuda and b.is_cuda
    out = torch.empty_like(a)
    n_elements = a.numel()
    
    grid = (triton.cdiv(n_elements, block_size),)
    # TODO: Launch with specified block_size
    
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
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load with mask
    x = None  # Replace
    
    # TODO: Compute block sum
    block_sum = None  # Replace with tl.sum(x, axis=0)
    
    # TODO: Store partial sum (one per block)
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
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_partials
    
    # TODO: Load partial sums
    partials = None  # Replace
    
    # TODO: Final sum
    total = None  # Replace
    
    # TODO: Store final result
    pass


def two_phase_sum(x: torch.Tensor) -> torch.Tensor:
    """Two-phase reduction for sum."""
    assert x.is_cuda
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_sums = torch.empty(n_blocks, dtype=x.dtype, device=x.device)
    
    # Phase 1: Partial sums
    grid1 = (n_blocks,)
    # TODO: Launch partial_sum_kernel
    
    # Phase 2: Final sum
    out = torch.empty(1, dtype=x.dtype, device=x.device)
    BLOCK_SIZE_2 = triton.next_power_of_2(n_blocks)
    grid2 = (1,)
    # TODO: Launch final_sum_kernel
    
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Block starting positions
    block_m = pid_m * BLOCK_M
    block_n = pid_n * BLOCK_N
    
    # Offsets within block
    offs_m = block_m + tl.arange(0, BLOCK_M)
    offs_n = block_n + tl.arange(0, BLOCK_N)
    
    # Masks
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # TODO: Calculate input indices (row-major: m * stride + n)
    in_indices = None  # Replace with offs_m[:, None] * stride_in_m + offs_n[None, :]
    
    # TODO: Load block
    block = None  # Replace
    
    # TODO: Calculate output indices (transposed: n * stride + m)
    # For transpose: out[n, m] = in[m, n]
    out_indices = None  # Replace with offs_n[:, None] * stride_out_m + offs_m[None, :]
    
    # TODO: Store transposed block
    # Note: need to transpose the mask too!
    pass


def transpose(x: torch.Tensor) -> torch.Tensor:
    """Transpose a 2D matrix."""
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty(N, M, dtype=x.dtype, device=x.device)
    
    BLOCK_M = 32
    BLOCK_N = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # TODO: Launch kernel
    
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
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load with very negative default for masked elements
    # HINT: other=-float('inf')
    x = None  # Replace
    
    # TODO: Block max
    block_max = None  # Replace with tl.max(x, axis=0)
    
    # TODO: Store
    pass


def find_max(x: torch.Tensor) -> torch.Tensor:
    """Find maximum element."""
    assert x.is_cuda
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_maxes = torch.empty(n_blocks, dtype=x.dtype, device=x.device)
    
    grid = (n_blocks,)
    # TODO: Launch block_max_kernel
    
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
    row_idx = tl.program_id(0)
    
    # Load max for this row
    row_max = tl.load(row_max_ptr + row_idx)
    
    # Column offsets
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols
    
    # TODO: Calculate pointer to this row
    row_ptr = None  # Replace with x_ptr + row_idx * n_cols
    
    # TODO: Load row
    x = None  # Replace
    
    # TODO: Compute exp(x - max)
    # HINT: out = tl.exp(x - row_max)
    out = None  # Replace
    
    # TODO: Store
    pass


def softmax_numerator(x: torch.Tensor, row_maxes: torch.Tensor) -> torch.Tensor:
    """Compute softmax numerator: exp(x - max)."""
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    
    # TODO: Launch kernel
    
    return out


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 5: Block-Level Programming")
    print("=" * 50)
    print("Run test_day05.py to check your implementations!")
