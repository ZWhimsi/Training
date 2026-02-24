"""
Day 17: Batched Outer Products
==============================
Estimated time: 1-2 hours
Prerequisites: Day 16 (tiled matmul)

Learning objectives:
- Understand outer product operation
- Implement batched outer products
- Handle 3D tensor operations in Triton
- Build toward attention score computation
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Outer Product
# ============================================================================
# For vectors a (M,) and b (N,):
# outer(a, b) = a.unsqueeze(-1) @ b.unsqueeze(0) = (M, N) matrix
#
# For batched: a (B, M), b (B, N) -> output (B, M, N)
# Each batch computes: output[b] = a[b] outer b[b]
# ============================================================================


# ============================================================================
# Exercise 1: Simple Outer Product
# ============================================================================

@triton.jit
def outer_product_kernel(
    a_ptr, b_ptr, output_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute outer product of two vectors: output[i,j] = a[i] * b[j]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # API hints:
    # - tl.load(ptr, mask=mask, other=val) -> load with default for masked elements
    # - tl.store(ptr, value, mask=mask) -> store elements to memory
    # - Broadcasting: a[:, None] expands to (M, 1), b[None, :] expands to (1, N)
    # - Element-wise multiply of (M, 1) * (1, N) gives (M, N) outer product
    
    a = tl.load(a_ptr + offs_m, mask=mask_m, other=0.0)
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    
    # TODO: Compute outer product using broadcasting
    # a is (BLOCK_M,), b is (BLOCK_N,) -> result should be (BLOCK_M, BLOCK_N)
    result = None  # Replace
    
    # Store with proper 2D indexing
    offs_output = offs_m[:, None] * N + offs_n[None, :]
    mask_output = mask_m[:, None] & mask_n[None, :]
    
    # TODO: Store result
    pass  # Replace


def outer_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute outer product of two 1D tensors."""
    M, N = a.shape[0], b.shape[0]
    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    outer_product_kernel[grid](a, b, output, M, N, BLOCK_M, BLOCK_N)
    return output


# ============================================================================
# Exercise 2: Batched Outer Product
# ============================================================================

@triton.jit
def batched_outer_kernel(
    a_ptr, b_ptr, output_ptr,
    B, M, N,
    stride_ab, stride_bb, stride_ob,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Batched outer product: output[b,i,j] = a[b,i] * b[b,j]
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # TODO: Load batch slice of a and b
    a_batch_ptr = a_ptr + pid_b * stride_ab
    b_batch_ptr = b_ptr + pid_b * stride_bb
    
    # API hints:
    # - Broadcasting: a[:, None] * b[None, :] computes outer product
    # - tl.store(ptr, value, mask=mask) -> store elements to memory
    
    a = tl.load(a_batch_ptr + offs_m, mask=mask_m, other=0.0)
    b = tl.load(b_batch_ptr + offs_n, mask=mask_n, other=0.0)
    
    # TODO: Compute outer product for this batch
    result = None  # Replace
    
    # Store to output
    output_batch_ptr = output_ptr + pid_b * stride_ob
    offs_output = offs_m[:, None] * N + offs_n[None, :]
    mask_output = mask_m[:, None] & mask_n[None, :]
    
    # TODO: Store result
    pass  # Replace


def batched_outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute batched outer product: (B, M) x (B, N) -> (B, M, N)"""
    B, M = a.shape
    _, N = b.shape
    output = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    batched_outer_kernel[grid](
        a, b, output, B, M, N,
        a.stride(0), b.stride(0), M * N,
        BLOCK_M, BLOCK_N
    )
    return output


# ============================================================================
# Exercise 3: Scaled Outer Product (for Attention)
# ============================================================================

@triton.jit
def scaled_outer_kernel(
    a_ptr, b_ptr, output_ptr,
    M, N, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Scaled outer product: output[i,j] = scale * a[i] * b[j]
    
    Used in attention: Q @ K^T / sqrt(d_k)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # API hints:
    # - Broadcasting: a[:, None] * b[None, :] computes outer product
    # - Multiply by scale factor for attention-style scaling
    # - tl.store(ptr, value, mask=mask) -> store elements to memory
    
    a = tl.load(a_ptr + offs_m, mask=mask_m, other=0.0)
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    
    # TODO: Compute scaled outer product: scale * outer(a, b)
    result = None  # Replace
    
    # Store
    offs_output = offs_m[:, None] * N + offs_n[None, :]
    mask_output = mask_m[:, None] & mask_n[None, :]
    
    # TODO: Store result
    pass  # Replace


def scaled_outer(a: torch.Tensor, b: torch.Tensor, scale: float) -> torch.Tensor:
    """Compute scaled outer product."""
    M, N = a.shape[0], b.shape[0]
    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    scaled_outer_kernel[grid](a, b, output, M, N, scale, BLOCK_M, BLOCK_N)
    return output


if __name__ == "__main__":
    print("Day 17: Batched Outer Products")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        # Test outer product
        a = torch.randn(64, device=device)
        b = torch.randn(128, device=device)
        
        print("\nTesting outer product:")
        result = outer_product(a, b)
        expected = torch.outer(a, b)
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        # Test batched
        a_batch = torch.randn(8, 64, device=device)
        b_batch = torch.randn(8, 128, device=device)
        
        print("\nTesting batched outer product:")
        result = batched_outer(a_batch, b_batch)
        expected = torch.bmm(a_batch.unsqueeze(-1), b_batch.unsqueeze(1))
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day17.py to verify!")
