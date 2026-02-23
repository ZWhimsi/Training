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
    """Sum all elements of a vector."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    
    # TODO: Load with 0 for out-of-bounds
    x = None  # Replace: tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # TODO: Sum all elements
    total = None  # Replace: tl.sum(x, axis=0)
    
    # Store result (single value)
    tl.store(out_ptr, total)


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
    """Sum each row of a matrix."""
    row = tl.program_id(0)
    
    if row >= M:
        return
    
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    # TODO: Load row
    row_data = None  # Replace: tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)
    
    # TODO: Sum the row
    row_sum = None  # Replace: tl.sum(row_data, axis=0)
    
    # TODO: Store
    tl.store(out_ptr + row, row_sum)


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
    """Compute mean of all elements."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # TODO: Compute sum
    total = None  # Replace
    
    # TODO: Divide by n to get mean
    # Note: n needs to be cast to float
    mean = None  # Replace: total / n
    
    tl.store(out_ptr, mean)


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
    """Compute variance: E[(x - mean)²]"""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # TODO: Compute (x - mean)²
    diff_sq = None  # Replace: (x - mean_val) ** 2
    
    # TODO: Apply mask before sum (zero out masked elements)
    diff_sq = tl.where(mask, diff_sq, 0.0)
    
    # TODO: Sum and divide by n
    var = None  # Replace: tl.sum(diff_sq, axis=0) / n
    
    tl.store(out_ptr, var)


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
    """Each block computes a partial sum."""
    pid = tl.program_id(0)
    
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    
    # TODO: Block sum
    block_sum = None  # Replace: tl.sum(x, axis=0)
    
    # TODO: Store partial result
    tl.store(partial_ptr + pid, block_sum)


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
