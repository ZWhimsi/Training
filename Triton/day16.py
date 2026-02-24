"""
Day 16: Tiled Matrix Multiplication
===================================
Estimated time: 1-2 hours
Prerequisites: Day 15 (matrix basics)

Learning objectives:
- Understand tiled matrix multiplication
- Use tl.dot() for efficient block-level matmul
- Implement the K-dimension loop pattern
- Handle non-square matrices

Matrix multiplication: C[i,j] = sum_k(A[i,k] * B[k,j])
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Exercise 1: Simple Tiled Matmul
# ============================================================================

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Tiled matrix multiplication: C = A @ B
    
    A: [M, K]
    B: [K, N]
    C: [M, N]
    
    Each program computes a BLOCK_M x BLOCK_N tile of C.
    """
    # Program ID determines which C tile we compute
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets for this C tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # API hints:
    # - tl.zeros((rows, cols), dtype=tl.float32) -> create zero-initialized tensor
    # - tl.load(ptr, mask=mask, other=val) -> load with default for masked elements
    # - tl.dot(a, b) -> matrix multiplication of two 2D tensors
    # - tl.store(ptr, value, mask=mask) -> store elements to memory
    
    # TODO: Initialize accumulator to zeros with shape (BLOCK_M, BLOCK_N)
    acc = None  # Replace
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Create masks for A and B loads
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        # Calculate pointers for A block [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        
        # Calculate pointers for B block [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        # TODO: Load A and B blocks
        a = None  # Replace
        b = None  # Replace
        
        # TODO: Block matrix multiply and accumulate
        pass  # Replace
    
    # Store C tile
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    
    # TODO: Store result
    tl.store(c_ptrs, acc, mask=mask_c)


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using Triton."""
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return C


# ============================================================================
# Exercise 2: Matmul with Bias
# ============================================================================

@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Matmul with bias: C = A @ B + bias"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc += tl.dot(a, b)
    
    # API hints:
    # - tl.load(ptr, mask=mask, other=val) -> load with default for masked elements
    # - Broadcasting: bias[None, :] broadcasts over M dimension
    
    # TODO: Load bias vector
    bias_mask = offs_n < N
    bias = None  # Replace
    
    # TODO: Add bias to accumulator (broadcasts over M dimension)
    acc = None  # Replace
    
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_c)


def matmul_bias(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Matmul with bias."""
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, dtype=A.dtype, device=A.device)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    matmul_bias_kernel[grid](
        A, B, bias, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


# ============================================================================
# Exercise 3: Batched Matmul
# ============================================================================

@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    B_dim, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched matmul: C[b] = A[b] @ B[b]"""
    # 3D grid: batch, M tiles, N tiles
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    if pid_b >= B_dim:
        return
    
    # TODO: Offset pointers by batch
    A_batch = A_ptr + pid_b * stride_ab
    B_batch = B_ptr + pid_b * stride_bb
    C_batch = C_ptr + pid_b * stride_cb
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        a_ptrs = A_batch + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_batch + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc += tl.dot(a, b)
    
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_batch + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_c)


def batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Batched matrix multiplication."""
    batch, M, K = A.shape
    _, K2, N = B.shape
    assert K == K2
    
    C = torch.empty(batch, M, N, dtype=A.dtype, device=A.device)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    grid = (batch, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    batched_matmul_kernel[grid](
        A, B, C,
        batch, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C


if __name__ == "__main__":
    print("Day 16: Tiled Matrix Multiplication")
    print("Run test_day16.py to verify!")
