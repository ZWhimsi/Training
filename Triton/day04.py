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

Hints:
- program_id(0) = x-axis, program_id(1) = y-axis, program_id(2) = z-axis
- Grid size determines total number of parallel programs
- Each program should process a unique portion of data
- Think about how to tile your problem for parallelism
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
    # TODO: Get row index from program ID
    row_idx = None  # Replace with tl.program_id(0)
    
    # TODO: Calculate pointer to start of this row
    # HINT: row_start = row_idx * n_cols
    row_start = None  # Replace
    
    # TODO: Create offsets within the row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # TODO: Load the row
    # HINT: tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    row_data = None  # Replace
    
    # TODO: Sum the row
    row_sum = None  # Replace with tl.sum(...)
    
    # TODO: Store result (one value per row)
    # HINT: tl.store(out_ptr + row_idx, row_sum)
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
    
    # TODO: Launch kernel
    
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
    # TODO: Get column index
    col_idx = None  # Replace
    
    # Row indices
    row_offsets = tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < n_rows
    
    # TODO: Calculate indices into the matrix
    # For column access, we need: row * stride_row + col
    # HINT: indices = row_offsets * stride_row + col_idx
    indices = None  # Replace
    
    # TODO: Load the column
    col_data = None  # Replace
    
    # TODO: Sum and store
    col_sum = None  # Replace
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
    
    # TODO: Launch kernel
    
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
    # TODO: Get 2D program IDs
    pid_m = None  # Row block ID - Replace with tl.program_id(0)
    pid_n = None  # Col block ID - Replace with tl.program_id(1)
    
    # TODO: Calculate block start positions
    block_start_m = None  # Replace with pid_m * BLOCK_M
    block_start_n = None  # Replace with pid_n * BLOCK_N
    
    # Generate offsets within the block
    offs_m = block_start_m + tl.arange(0, BLOCK_M)
    offs_n = block_start_n + tl.arange(0, BLOCK_N)
    
    # Create masks for bounds
    mask_m = offs_m < n_rows
    mask_n = offs_n < n_cols
    
    # TODO: Create 2D mask
    # HINT: mask = mask_m[:, None] & mask_n[None, :]
    mask = None  # Replace
    
    # TODO: Calculate 2D indices
    # HINT: indices = offs_m[:, None] * stride + offs_n[None, :]
    indices = None  # Replace
    
    # TODO: Load, add, store
    a = None  # Replace
    b = None  # Replace
    out = None  # Replace
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
    
    # TODO: Launch kernel
    
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
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    # TODO: Process multiple blocks per program (grid-stride loop)
    # Each program starts at pid * BLOCK_SIZE and jumps by num_pids * BLOCK_SIZE
    # HINT: Use a conceptual loop (in Triton, we typically launch enough programs
    #       but this pattern is useful for understanding)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load, add, store
    a = None  # Replace
    b = None  # Replace
    out = None  # Replace
    pass


def vector_add_cyclic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector add."""
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    
    out = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    # TODO: Launch kernel
    
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
    # TODO: Get batch index from program ID
    batch_idx = None  # Replace
    
    # TODO: Load the scale for this batch
    scale = None  # Replace with tl.load(scales_ptr + batch_idx)
    
    # Vector offsets
    vec_offsets = tl.arange(0, BLOCK_SIZE)
    mask = vec_offsets < vec_size
    
    # TODO: Calculate pointer to this batch's vector
    x_batch_ptr = None  # Replace with x_ptr + batch_idx * vec_size
    out_batch_ptr = None  # Replace
    
    # TODO: Load, scale, store
    x = None  # Replace
    out = None  # Replace
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
    
    # TODO: Launch kernel
    
    return out


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 4: Program IDs and Grid Configuration")
    print("=" * 50)
    print("Run test_day04.py to check your implementations!")
