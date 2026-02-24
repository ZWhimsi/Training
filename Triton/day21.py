"""
Day 21: Matrix-Vector Products
==============================
Estimated time: 1-2 hours
Prerequisites: Day 20 (transpose)

Learning objectives:
- Implement matrix-vector multiplication
- Understand reduction patterns in matvec
- Handle row-major vs column-major efficiently
- Build toward batch operations
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Matrix-Vector Multiplication
# ============================================================================
# y = A @ x where A is (M, N) and x is (N,)
# y[i] = sum_j(A[i,j] * x[j])
#
# Each output element is a dot product of a row with the vector.
# ============================================================================


# ============================================================================
# Exercise 1: Basic Matrix-Vector Product
# ============================================================================

@triton.jit
def matvec_kernel(
    A_ptr, x_ptr, y_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    """
    Compute y = A @ x where A is (M, N) and x is (N,).
    Each program computes one output element.
    """
    row_idx = tl.program_id(0)
    
    # Compute dot product of row with x
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    
    # TODO: Load row of A
    row_ptr = A_ptr + row_idx * N
    row = tl.load(row_ptr + offs_n, mask=mask, other=0.0)
    
    # TODO: Load x vector
    x = tl.load(x_ptr + offs_n, mask=mask, other=0.0)
    
    # TODO: Compute dot product and store result
    # API hints:
    # - tl.sum(tensor, axis=0) -> reduce along axis, returns scalar or reduced tensor
    # - tl.store(ptr, value) -> store value to memory location
    pass


def matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute y = A @ x."""
    M, N = A.shape
    y = torch.empty(M, device=A.device, dtype=A.dtype)
    
    BLOCK_N = triton.next_power_of_2(N)
    matvec_kernel[(M,)](A, x, y, M, N, BLOCK_N=BLOCK_N)
    
    return y


# ============================================================================
# Exercise 2: Blocked Matrix-Vector (for large N)
# ============================================================================

@triton.jit
def matvec_blocked_kernel(
    A_ptr, x_ptr, y_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    """
    Matrix-vector with blocking for large N.
    Accumulates partial sums across blocks.
    """
    row_idx = tl.program_id(0)
    
    # Accumulator for this row
    acc = tl.zeros((1,), dtype=tl.float32)
    
    # Process in blocks
    row_ptr = A_ptr + row_idx * N
    
    for block_start in range(0, N, BLOCK_N):
        offs = block_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        
        # Load block of row and x
        row_block = tl.load(row_ptr + offs, mask=mask, other=0.0)
        x_block = tl.load(x_ptr + offs, mask=mask, other=0.0)
        
        # Accumulate partial dot product
        acc += tl.sum(row_block * x_block, axis=0)
    
    # Store final result
    tl.store(y_ptr + row_idx, acc)


def matvec_blocked(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Blocked matrix-vector for large matrices."""
    M, N = A.shape
    y = torch.empty(M, device=A.device, dtype=A.dtype)
    
    BLOCK_N = 1024
    matvec_blocked_kernel[(M,)](A, x, y, M, N, BLOCK_N=BLOCK_N)
    
    return y


# ============================================================================
# Exercise 3: Vector-Matrix Product (v^T @ A)
# ============================================================================

@triton.jit
def vecmat_kernel(
    v_ptr, A_ptr, y_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
):
    """
    Compute y = v^T @ A where v is (M,) and A is (M, N).
    y[j] = sum_i(v[i] * A[i,j])
    
    Each program computes one output element.
    """
    col_idx = tl.program_id(0)
    
    offs_m = tl.arange(0, BLOCK_M)
    mask = offs_m < M
    
    # TODO: Load v
    v = tl.load(v_ptr + offs_m, mask=mask, other=0.0)
    
    # TODO: Load column of A
    # Column col_idx: elements at [0*N+col_idx, 1*N+col_idx, ...]
    col_offs = offs_m * N + col_idx
    col = tl.load(A_ptr + col_offs, mask=mask, other=0.0)
    
    # TODO: Compute dot product of v and column, then store result
    # API hints:
    # - tl.sum(tensor, axis=0) -> reduce along axis
    # - tl.store(ptr, value) -> store value to memory
    pass


def vecmat(v: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Compute y = v^T @ A."""
    M, N = A.shape
    y = torch.empty(N, device=A.device, dtype=A.dtype)
    
    BLOCK_M = triton.next_power_of_2(M)
    vecmat_kernel[(N,)](v, A, y, M, N, BLOCK_M=BLOCK_M)
    
    return y


# ============================================================================
# Exercise 4: Batched Matrix-Vector
# ============================================================================

@triton.jit
def batched_matvec_kernel(
    A_ptr, x_ptr, y_ptr,
    B, M, N,
    stride_Ab, stride_xb, stride_yb,
    BLOCK_N: tl.constexpr,
):
    """
    Batched matvec: y[b] = A[b] @ x[b]
    A: (B, M, N), x: (B, N), y: (B, M)
    """
    batch_idx = tl.program_id(0)
    row_idx = tl.program_id(1)
    
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    
    # Batch offsets
    A_batch = A_ptr + batch_idx * stride_Ab
    x_batch = x_ptr + batch_idx * stride_xb
    y_batch = y_ptr + batch_idx * stride_yb
    
    # Load row and x
    row = tl.load(A_batch + row_idx * N + offs_n, mask=mask, other=0.0)
    x = tl.load(x_batch + offs_n, mask=mask, other=0.0)
    
    # Compute dot product
    dot = tl.sum(row * x, axis=0)
    
    # Store
    tl.store(y_batch + row_idx, dot)


def batched_matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Batched matrix-vector: y[b] = A[b] @ x[b]."""
    B, M, N = A.shape
    y = torch.empty((B, M), device=A.device, dtype=A.dtype)
    
    BLOCK_N = triton.next_power_of_2(N)
    grid = (B, M)
    
    batched_matvec_kernel[grid](
        A, x, y, B, M, N,
        A.stride(0), x.stride(0), y.stride(0),
        BLOCK_N=BLOCK_N
    )
    
    return y


if __name__ == "__main__":
    print("Day 21: Matrix-Vector Products")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        M, N = 256, 128
        A = torch.randn(M, N, device=device)
        x = torch.randn(N, device=device)
        
        print("\nTesting matvec:")
        result = matvec(A, x)
        expected = A @ x
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        print("\nTesting batched matvec:")
        B = 8
        A_batch = torch.randn(B, M, N, device=device)
        x_batch = torch.randn(B, N, device=device)
        result = batched_matvec(A_batch, x_batch)
        expected = torch.bmm(A_batch, x_batch.unsqueeze(-1)).squeeze(-1)
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day21.py to verify!")
