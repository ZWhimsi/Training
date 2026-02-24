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
    # API hints:
    # - tl.program_id(0), tl.program_id(1) -> 2D block indices
    # - tl.arange(start, end) -> create range
    # - mask[:, None] & mask[None, :] -> 2D mask from 1D masks
    # - offs[:, None] * stride + offs[None, :] -> 2D indices
    # - tl.load(ptr + indices, mask=mask) -> load 2D block
    # - tl.store(ptr + indices, value, mask=mask) -> store 2D block
    
    # TODO: Implement 2D element-wise kernel
    pass


def elementwise_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """2D element-wise multiply."""
    assert x.is_cuda and x.shape == y.shape
    M, N = x.shape
    out = torch.empty_like(x)
    
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
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
    # API hints:
    # - tl.program_id(0) -> batch index
    # - tl.program_id(1) -> row block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.sum(x, axis=1) -> sum along axis 1
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement batch matvec kernel
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
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
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
    # API hints:
    # - tl.program_id(0) -> batch index
    # - tl.program_id(1) -> row block index
    # - tl.program_id(2) -> column block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + indices, mask=mask) -> load 2D block
    # - tl.store(ptr + indices, value, mask=mask) -> store 2D block
    
    # TODO: Implement 3D batch add kernel
    pass


def batch_add_3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """3D batched addition."""
    assert a.is_cuda and a.shape == b.shape and a.dim() == 3
    B, M, N = a.shape
    out = torch.empty_like(a)
    
    BLOCK_M, BLOCK_N = 32, 32
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
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
    # API hints:
    # - tl.program_id(0) -> column index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.sum(x, axis=0) -> sum reduction
    # - tl.store(ptr, value) -> store single value
    
    # TODO: Implement sum axis 0 kernel
    pass


def sum_axis0(x: torch.Tensor) -> torch.Tensor:
    """Sum along axis 0."""
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    out = torch.empty(N, dtype=x.dtype, device=x.device)
    
    BLOCK_M = triton.next_power_of_2(M)
    grid = (N,)
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
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
    # API hints:
    # - tl.program_id(0), tl.program_id(1) -> 2D block indices
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load 1D bias
    # - tl.load(ptr + indices, mask=mask) -> load 2D block
    # - bias[None, :] -> broadcast bias to match 2D shape
    # - tl.store(ptr + indices, value, mask=mask) -> store 2D block
    
    # TODO: Implement broadcast add kernel
    pass


def broadcast_add(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Add bias to each row."""
    assert x.is_cuda and bias.is_cuda
    M, N = x.shape
    assert bias.shape == (N,)
    
    out = torch.empty_like(x)
    
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # API hints:
    # - kernel_name[grid](args...) -> launch kernel
    
    # TODO: Launch kernel
    pass
    
    return out


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 6: Multi-dimensional Grids")
    print("=" * 50)
    print("Run test_day06.py to check your implementations!")
