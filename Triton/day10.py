"""
Day 10: Reduction Operations - Sum and Mean
==========================================
Estimated time: 1-2 hours
Prerequisites: Day 8-9 (vector operations)

Learning objectives:
- Implement reduction operations (sum, mean)
- Understand parallel reduction patterns
- Handle reductions along specific axes
- Use tl.sum() and tl.max() effectively
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Exercise 1: Simple Vector Sum
# ============================================================================

@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """
    Sum all elements of a vector.
    """
    # API hints:
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.sum(x, axis=0) -> sum reduction
    # - tl.store(ptr, value) -> store single value
    
    # TODO: Implement sum kernel
    pass


def vector_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum all elements."""
    out = torch.empty(1, dtype=x.dtype, device=x.device)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    sum_kernel[(1,)](x, out, n, BLOCK)
    return out


# ============================================================================
# Exercise 2: Row-wise Sum (2D)
# ============================================================================

@triton.jit
def row_sum_kernel(x_ptr, out_ptr, M, N, stride, BLOCK_N: tl.constexpr):
    """
    Sum each row of a matrix.
    """
    # API hints:
    # - tl.program_id(0) -> row index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.sum(x, axis=0) -> sum reduction
    # - tl.store(ptr, value) -> store single value
    
    # TODO: Implement row sum kernel
    pass


def row_wise_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum each row."""
    M, N = x.shape
    out = torch.empty(M, dtype=x.dtype, device=x.device)
    BLOCK_N = triton.next_power_of_2(N)
    row_sum_kernel[(M,)](x, out, M, N, x.stride(0), BLOCK_N)
    return out


# ============================================================================
# Exercise 3: Mean Operation
# ============================================================================

@triton.jit
def mean_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """
    Compute mean of all elements.
    """
    # API hints:
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.sum(x, axis=0) -> sum reduction
    # - tl.store(ptr, value) -> store single value
    
    # TODO: Implement mean kernel (sum / n)
    pass


def vector_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean of all elements."""
    out = torch.empty(1, dtype=x.dtype, device=x.device)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    mean_kernel[(1,)](x, out, n, BLOCK)
    return out


# ============================================================================
# Exercise 4: Variance (uses mean)
# ============================================================================

@triton.jit
def variance_kernel(x_ptr, out_ptr, mean_val, n, BLOCK: tl.constexpr):
    """
    Compute variance: E[(x - mean)²]
    """
    # API hints:
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.where(mask, val, 0.0) -> zero out masked elements before sum
    # - tl.sum(x, axis=0) -> sum reduction
    # - tl.store(ptr, value) -> store single value
    
    # TODO: Implement variance kernel
    pass


def vector_variance(x: torch.Tensor) -> torch.Tensor:
    """Variance of all elements."""
    mean_val = x.mean().item()
    out = torch.empty(1, dtype=x.dtype, device=x.device)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    variance_kernel[(1,)](x, out, mean_val, n, BLOCK)
    return out


# ============================================================================
# Exercise 5: Multi-block Sum (for large arrays)
# ============================================================================

@triton.jit
def partial_sum_kernel(x_ptr, partial_ptr, n, BLOCK: tl.constexpr):
    """
    Each block computes a partial sum.
    """
    # API hints:
    # - tl.program_id(0) -> block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask, other=0.0) -> load with default
    # - tl.sum(x, axis=0) -> sum reduction
    # - tl.store(ptr, value) -> store single value
    
    # TODO: Implement partial sum kernel
    pass


def large_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum for large arrays using multiple blocks."""
    n = x.numel()
    BLOCK = 1024
    n_blocks = triton.cdiv(n, BLOCK)
    
    partial = torch.empty(n_blocks, dtype=x.dtype, device=x.device)
    partial_sum_kernel[(n_blocks,)](x, partial, n, BLOCK)
    
    # Final sum of partials
    return partial.sum().unsqueeze(0)


if __name__ == "__main__":
    print("Day 10: Reduction Operations")
    print("Run test_day10.py to verify!")
