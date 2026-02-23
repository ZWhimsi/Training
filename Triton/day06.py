"""
Day 6: Multi-dimensional Grids
==============================
Estimated time: 1-2 hours
Prerequisites: Day 5 (block programming)

Learning objectives:
- Work with 2D and 3D grids
- Map multi-dimensional program IDs to data
- Handle batched operations efficiently
- Understand memory layout for multi-dim data

Hints:
- tl.program_id(0), tl.program_id(1), tl.program_id(2) for 3 axes
- Grid dimensions: grid = (dim0, dim1, dim2)
- Think about how to map program IDs to tensor indices
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Exercise 1: 2D Grid Element-wise Operation
# ============================================================================

@triton.jit
def elementwise_2d_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M, N,
    stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Element-wise multiply using 2D grid.
    out[i,j] = x[i,j] * y[i,j]
    """
    # TODO: Get 2D program IDs
    pid_m = None  # Replace
    pid_n = None  # Replace
    
    # Block start positions
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # Offsets
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    
    # Masks
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # TODO: Compute indices
    indices = None  # Replace: offs_m[:, None] * stride_m + offs_n[None, :]
    
    # TODO: Load, multiply, store
    x = None  # Replace
    y = None  # Replace
    out = None  # Replace
    pass


def elementwise_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """2D element-wise multiply."""
    assert x.is_cuda and x.shape == y.shape
    M, N = x.shape
    out = torch.empty_like(x)
    
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # TODO: Launch kernel
    
    return out


# ============================================================================
# Exercise 2: Batch Matrix-Vector Multiply
# ============================================================================

@triton.jit
def batch_matvec_kernel(
    A_ptr,      # [B, M, N]
    x_ptr,      # [B, N]
    out_ptr,    # [B, M]
    B, M, N,
    stride_ab, stride_am,  # A strides
    stride_xb,              # x stride
    stride_ob,              # out stride
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Batched matrix-vector multiplication.
    out[b, m] = sum_n(A[b, m, n] * x[b, n])
    
    Grid: (B, ceil(M/BLOCK_M))
    """
    # TODO: Get batch and row block indices
    batch_idx = None  # Replace: tl.program_id(0)
    pid_m = None      # Replace: tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Pointers for this batch
    A_batch = A_ptr + batch_idx * stride_ab
    x_batch = x_ptr + batch_idx * stride_xb
    out_batch = out_ptr + batch_idx * stride_ob
    
    # TODO: Load x vector for this batch
    x = None  # Replace
    
    # Accumulator
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # TODO: Loop over N dimension if BLOCK_N < N
    # For simplicity, assume N <= BLOCK_N
    
    # TODO: Load A block [BLOCK_M, BLOCK_N]
    A_indices = offs_m[:, None] * stride_am + offs_n[None, :]
    A_block = tl.load(A_batch + A_indices, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    
    # TODO: Compute dot product for each row
    # HINT: acc = tl.sum(A_block * x[None, :], axis=1)
    acc = None  # Replace
    
    # TODO: Store result
    pass


def batch_matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Batched matrix-vector multiply."""
    assert A.is_cuda and x.is_cuda
    B, M, N = A.shape
    assert x.shape == (B, N)
    
    out = torch.empty(B, M, dtype=A.dtype, device=A.device)
    
    BLOCK_M = 32
    BLOCK_N = triton.next_power_of_2(N)
    
    grid = (B, triton.cdiv(M, BLOCK_M))
    
    # TODO: Launch kernel
    
    return out


# ============================================================================
# Exercise 3: 3D Grid for Batch Operations
# ============================================================================

@triton.jit
def batch_add_3d_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    B, M, N,
    stride_b, stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    3D batched addition using 3D grid.
    
    Grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    """
    # TODO: Get 3D program IDs
    pid_b = None  # Replace: tl.program_id(0) - batch
    pid_m = None  # Replace: tl.program_id(1) - rows
    pid_n = None  # Replace: tl.program_id(2) - cols
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # TODO: Compute indices within this batch
    batch_offset = pid_b * stride_b
    indices = offs_m[:, None] * stride_m + offs_n[None, :]
    
    # TODO: Load, add, store
    a = None  # Replace
    b = None  # Replace
    out = None  # Replace
    pass


def batch_add_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """3D batched addition."""
    assert a.is_cuda and a.shape == b.shape and a.dim() == 3
    B, M, N = a.shape
    out = torch.empty_like(a)
    
    BLOCK_M, BLOCK_N = 32, 32
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # TODO: Launch kernel
    
    return out


# ============================================================================
# Exercise 4: Reduction Along Axis
# ============================================================================

@triton.jit
def sum_axis0_kernel(
    x_ptr,
    out_ptr,
    M, N,
    stride_m,
    BLOCK_M: tl.constexpr,
):
    """
    Sum along axis 0 (sum over rows).
    out[j] = sum_i(x[i, j])
    
    Grid: (N,) - one program per column
    """
    col_idx = tl.program_id(0)
    
    if col_idx >= N:
        return
    
    # Row offsets
    offs_m = tl.arange(0, BLOCK_M)
    mask = offs_m < M
    
    # TODO: Calculate indices for this column
    indices = None  # Replace: offs_m * stride_m + col_idx
    
    # TODO: Load column
    col_data = None  # Replace
    
    # TODO: Sum and store
    col_sum = None  # Replace
    pass


def sum_axis0(x: torch.Tensor) -> torch.Tensor:
    """Sum along axis 0."""
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    
    BLOCK_M = triton.next_power_of_2(M)
    grid = (N,)
    
    # TODO: Launch kernel
    
    return out


# ============================================================================
# Exercise 5: Broadcast Add
# ============================================================================

@triton.jit
def broadcast_add_kernel(
    x_ptr,      # [M, N]
    bias_ptr,   # [N]
    out_ptr,    # [M, N]
    M, N,
    stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Add bias to each row: out[i, j] = x[i, j] + bias[j]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # TODO: Load bias (1D)
    bias = None  # Replace: tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    
    # TODO: Load x block (2D)
    x_indices = offs_m[:, None] * stride_m + offs_n[None, :]
    x = None  # Replace
    
    # TODO: Add bias (broadcasts automatically)
    out = None  # Replace: x + bias[None, :]
    
    # TODO: Store
    pass


def broadcast_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Add bias to each row."""
    assert x.is_cuda and bias.is_cuda
    M, N = x.shape
    assert bias.shape == (N,)
    
    out = torch.empty_like(x)
    
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # TODO: Launch kernel
    
    return out


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 6: Multi-dimensional Grids")
    print("=" * 50)
    print("Run test_day06.py to check your implementations!")
