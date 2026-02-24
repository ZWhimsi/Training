"""
Day 4: Program IDs and Grid Configuration
=========================================
Estimated time: 1-2 hours
Prerequisites: Day 3 (masking)

Learning objectives:
- Deep understanding of tl.program_id()
- Configure 1D, 2D, and 3D grids
- Map program IDs to data indices
- Understand parallelization strategies
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Program IDs and Grids
# ============================================================================
# 
# In Triton, work is divided into a grid of programs.
# Each program has a unique ID on each axis (0, 1, 2).
#
# 1D Grid: grid = (num_programs,)
#   - program_id(0) ranges from 0 to num_programs-1
#
# 2D Grid: grid = (num_x, num_y)
#   - program_id(0) ranges from 0 to num_x-1
#   - program_id(1) ranges from 0 to num_y-1
#
# Total programs = product of grid dimensions
# ============================================================================


# ============================================================================
# Exercise 1: Row-wise Operations (2D Grid, 1D data access)
# ============================================================================
# Process each row independently using 2D indexing

@triton.jit
def row_sum_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sum each row of a 2D matrix.
    
    Grid: (n_rows,) - one program per row
    """
    # API hints:
    # - tl.program_id(axis) -> get block index on given axis
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.sum(x, axis=0) -> sum reduction
    # - tl.store(ptr, value) -> store single value
    
    # TODO: Implement row sum kernel
    pass


def row_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum each row."""
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    out = torch.empty(n_rows, dtype=x.dtype, device=x.device)
    
    # One program per row
    grid = (n_rows,)
    
    # BLOCK_SIZE must be >= n_cols for simple implementation
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Exercise 2: Column-wise Operations
# ============================================================================
# Process each column independently

@triton.jit
def col_sum_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride_row,  # Stride between rows
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sum each column of a 2D matrix.
    
    Grid: (n_cols,) - one program per column
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.sum(x, axis=0) -> sum reduction
    # - tl.store(ptr, value) -> store single value
    
    # TODO: Implement column sum kernel
    pass


def col_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum each column."""
    assert x.is_cuda and x.dim() == 2
    n_rows, n_cols = x.shape
    out = torch.empty(n_cols, dtype=x.dtype, device=x.device)
    
    # Ensure contiguous for correct stride
    x = x.contiguous()
    stride_row = x.stride(0)
    
    grid = (n_cols,)
    BLOCK_SIZE = triton.next_power_of_2(n_rows)
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Exercise 3: 2D Grid for Matrix Operations
# ============================================================================
# Use 2D program IDs for processing 2D data

@triton.jit
def add_matrices_2d_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_rows,
    n_cols,
    stride,
    BLOCK_M: tl.constexpr,  # Block size for rows
    BLOCK_N: tl.constexpr,  # Block size for cols
):
    """
    Add two matrices using 2D grid.
    
    Grid: (ceil(n_rows/BLOCK_M), ceil(n_cols/BLOCK_N))
    Each program handles a BLOCK_M x BLOCK_N tile.
    """
    # API hints:
    # - tl.program_id(0) -> row block index
    # - tl.program_id(1) -> column block index
    # - tl.arange(start, end) -> create range
    # - mask[:, None] & mask[None, :] -> 2D mask from 1D masks
    # - offs[:, None] * stride + offs[None, :] -> 2D indices
    # - tl.load(ptr + indices, mask=mask) -> load 2D block
    # - tl.store(ptr + indices, value, mask=mask) -> store 2D block
    
    # TODO: Implement 2D matrix add kernel
    pass


def add_matrices_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add matrices using 2D grid."""
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape and a.dim() == 2
    
    n_rows, n_cols = a.shape
    out = torch.empty_like(a)
    
    BLOCK_M = 32
    BLOCK_N = 32
    
    # 2D grid
    grid = (triton.cdiv(n_rows, BLOCK_M), triton.cdiv(n_cols, BLOCK_N))
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Exercise 4: Block-Cyclic Processing
# ============================================================================
# When there are more blocks than SMs, use cyclic distribution

@triton.jit
def vector_add_cyclic_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vector add with potentially multiple elements per program.
    
    Uses a grid-stride loop pattern common in CUDA.
    """
    # API hints:
    # - tl.program_id(axis) -> get block index
    # - tl.num_programs(axis) -> get total number of programs
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement vector add kernel
    pass


def vector_add_cyclic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector add."""
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    
    out = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Exercise 5: Batch Processing with 2D Grid
# ============================================================================
# Process batches of vectors

@triton.jit
def batch_scale_kernel(
    x_ptr,
    scales_ptr,  # One scale per batch
    out_ptr,
    batch_size,
    vec_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scale each vector in a batch by its corresponding scale.
    
    x: [batch_size, vec_size]
    scales: [batch_size]
    out[b, i] = x[b, i] * scales[b]
    """
    # API hints:
    # - tl.program_id(axis) -> get block index (batch index)
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr) -> load single scalar
    # - tl.load(ptr + offsets, mask=mask) -> load vector
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement batch scale kernel
    pass


def batch_scale(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Scale each batch element."""
    assert x.is_cuda and scales.is_cuda
    assert x.dim() == 2 and scales.dim() == 1
    assert x.shape[0] == scales.shape[0]
    
    batch_size, vec_size = x.shape
    out = torch.empty_like(x)
    
    # One program per batch
    grid = (batch_size,)
    BLOCK_SIZE = triton.next_power_of_2(vec_size)
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 4: Program IDs and Grid Configuration")
    print("=" * 50)
    print("Run test_day04.py to check your implementations!")
