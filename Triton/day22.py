"""
Day 22: Batch Matrix Multiply
=============================
Estimated time: 1-2 hours
Prerequisites: Day 21 (matvec), Day 16 (tiled matmul)

Learning objectives:
- Implement batched matrix multiplication
- Handle different batch dimensions
- Optimize for attention-like patterns
- Understand batch parallelism
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Batched Matrix Multiply
# ============================================================================
# For A: (B, M, K) and B: (B, K, N):
# C[b] = A[b] @ B[b] for each batch b
#
# In attention: Q @ K^T is a batched matmul where B = num_heads
# ============================================================================


# ============================================================================
# Exercise 1: Simple Batched Matmul
# ============================================================================

@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    B_size, M, N, K,
    stride_Ab, stride_Am, stride_Ak,
    stride_Bb, stride_Bk, stride_Bn,
    stride_Cb, stride_Cm, stride_Cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched matmul: C[b] = A[b] @ B[b]
    """
    batch_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # Block starting positions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers for this batch
    A_batch = A_ptr + batch_idx * stride_Ab
    B_batch = B_ptr + batch_idx * stride_Bb
    C_batch = C_ptr + batch_idx * stride_Cb
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Tiled loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        
        # Load A block [BLOCK_M, BLOCK_K]
        a_offs = offs_m[:, None] * stride_Am + k_offs[None, :] * stride_Ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(A_batch + a_offs, mask=a_mask, other=0.0)
        
        # Load B block [BLOCK_K, BLOCK_N]
        b_offs = k_offs[:, None] * stride_Bk + offs_n[None, :] * stride_Bn
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(B_batch + b_offs, mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a, b)
    
    # Store result
    c_offs = offs_m[:, None] * stride_Cm + offs_n[None, :] * stride_Cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_batch + c_offs, acc, mask=c_mask)


def batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute C[b] = A[b] @ B[b]."""
    assert A.dim() == 3 and B.dim() == 3
    B_size, M, K = A.shape
    _, K2, N = B.shape
    assert K == K2
    
    C = torch.empty((B_size, M, N), device=A.device, dtype=A.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    grid = (B_size, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    batched_matmul_kernel[grid](
        A, B, C,
        B_size, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return C


# ============================================================================
# Exercise 2: Batched Matmul with Transpose (A @ B^T)
# ============================================================================

@triton.jit
def batched_matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    B_size, M, N, K,
    stride_Ab, stride_Am, stride_Ak,
    stride_Bb, stride_Bn, stride_Bk,  # Note: B is stored as (B, N, K)
    stride_Cb, stride_Cm, stride_Cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched matmul with B transposed: C[b] = A[b] @ B[b]^T
    A: (B, M, K), B: (B, N, K), C: (B, M, N)
    """
    batch_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_batch = A_ptr + batch_idx * stride_Ab
    B_batch = B_ptr + batch_idx * stride_Bb
    C_batch = C_ptr + batch_idx * stride_Cb
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        
        # Load A [M, K]
        a_offs = offs_m[:, None] * stride_Am + k_offs[None, :] * stride_Ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(A_batch + a_offs, mask=a_mask, other=0.0)
        
        # Load B^T: B stored as (N, K), we want (K, N) view
        # B[n, k] -> B^T[k, n]
        b_offs = offs_n[None, :] * stride_Bn + k_offs[:, None] * stride_Bk
        b_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        b_t = tl.load(B_batch + b_offs, mask=b_mask, other=0.0)  # (BLOCK_K, BLOCK_N)
        
        acc += tl.dot(a, b_t)
    
    c_offs = offs_m[:, None] * stride_Cm + offs_n[None, :] * stride_Cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_batch + c_offs, acc, mask=c_mask)


def batched_matmul_bt(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Compute C[b] = A[b] @ B[b]^T. A: (B, M, K), B: (B, N, K)."""
    B_size, M, K = A.shape
    _, N, K2 = B.shape
    assert K == K2
    
    C = torch.empty((B_size, M, N), device=A.device, dtype=A.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    grid = (B_size, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    batched_matmul_bt_kernel[grid](
        A, B, C,
        B_size, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return C


# ============================================================================
# Exercise 3: Scaled Batched Matmul (for Attention)
# ============================================================================

@triton.jit
def scaled_batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    scale,
    B_size, M, N, K,
    stride_Ab, stride_Am, stride_Ak,
    stride_Bb, stride_Bk, stride_Bn,
    stride_Cb, stride_Cm, stride_Cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Scaled batched matmul: C = scale * (A @ B)
    Used in attention: scores = (Q @ K^T) / sqrt(d_k)
    """
    batch_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_batch = A_ptr + batch_idx * stride_Ab
    B_batch = B_ptr + batch_idx * stride_Bb
    C_batch = C_ptr + batch_idx * stride_Cb
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        
        a_offs = offs_m[:, None] * stride_Am + k_offs[None, :] * stride_Ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(A_batch + a_offs, mask=a_mask, other=0.0)
        
        b_offs = k_offs[:, None] * stride_Bk + offs_n[None, :] * stride_Bn
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(B_batch + b_offs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b)
    
    # Apply scale
    acc = acc * scale
    
    c_offs = offs_m[:, None] * stride_Cm + offs_n[None, :] * stride_Cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_batch + c_offs, acc, mask=c_mask)


def scaled_batched_matmul(A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    """Compute C = scale * (A @ B)."""
    B_size, M, K = A.shape
    _, K2, N = B.shape
    assert K == K2
    
    C = torch.empty((B_size, M, N), device=A.device, dtype=A.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 32
    
    grid = (B_size, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    scaled_batched_matmul_kernel[grid](
        A, B, C, scale,
        B_size, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return C


if __name__ == "__main__":
    print("Day 22: Batch Matrix Multiply")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        B_size, M, N, K = 8, 64, 64, 32
        A = torch.randn(B_size, M, K, device=device)
        B = torch.randn(B_size, K, N, device=device)
        
        print("\nTesting batched matmul:")
        result = batched_matmul(A, B)
        expected = torch.bmm(A, B)
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        print("\nTesting A @ B^T:")
        B_t = torch.randn(B_size, N, K, device=device)
        result = batched_matmul_bt(A, B_t)
        expected = torch.bmm(A, B_t.transpose(-2, -1))
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day22.py to verify!")
