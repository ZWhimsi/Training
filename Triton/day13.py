"""
Day 13: Fused Softmax
====================
Estimated time: 1-2 hours
Prerequisites: Day 10-12 (reductions, numerical stability)

Learning objectives:
- Implement softmax from scratch in Triton
- Apply numerical stability techniques
- Fuse multiple operations into one kernel
- Compare performance with PyTorch

Softmax formula: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Exercise 1: Naive Softmax (Numerically Unstable)
# ============================================================================

@triton.jit
def naive_softmax_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """
    WARNING: This is numerically unstable! For learning only.
    softmax(x)_i = exp(x_i) / sum(exp(x))
    """
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    
    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
    
    # TODO: Compute exp(x) - UNSTABLE for large x!
    exp_x = None  # Replace: tl.exp(x)
    
    # TODO: Sum of exponentials
    sum_exp = None  # Replace: tl.sum(exp_x, axis=0)
    
    # TODO: Normalize
    out = None  # Replace: exp_x / sum_exp
    
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================================
# Exercise 2: Safe Softmax (Numerically Stable)
# ============================================================================

@triton.jit
def safe_softmax_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """
    Stable softmax: subtract max before exp.
    softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
    """
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    
    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
    
    # TODO: Find maximum (for stability)
    x_max = None  # Replace: tl.max(x, axis=0)
    
    # TODO: Subtract max before exp
    x_shifted = None  # Replace: x - x_max
    
    # TODO: Compute exp of shifted values
    exp_x = None  # Replace: tl.exp(x_shifted)
    
    # TODO: Apply mask before sum
    exp_x_masked = tl.where(mask, exp_x, 0.0)
    
    # TODO: Sum of exponentials
    sum_exp = None  # Replace: tl.sum(exp_x_masked, axis=0)
    
    # TODO: Normalize
    out = None  # Replace: exp_x / sum_exp
    
    tl.store(out_ptr + offs, out, mask=mask)


def softmax_1d(x: torch.Tensor) -> torch.Tensor:
    """Softmax for 1D tensor."""
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    safe_softmax_kernel[(1,)](x, out, n, BLOCK)
    return out


# ============================================================================
# Exercise 3: Row-wise Softmax (2D)
# ============================================================================

@triton.jit
def row_softmax_kernel(x_ptr, out_ptr, M, N, stride, BLOCK_N: tl.constexpr):
    """
    Softmax along rows (last dimension).
    Each row is processed independently.
    """
    row = tl.program_id(0)
    
    if row >= M:
        return
    
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    # Pointer to this row
    row_ptr = x_ptr + row * stride
    
    # TODO: Load row
    x = None  # Replace: tl.load(row_ptr + cols, mask=mask, other=-float('inf'))
    
    # TODO: Max for stability
    x_max = None  # Replace: tl.max(x, axis=0)
    
    # TODO: Shift, exp, mask
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    exp_x = tl.where(mask, exp_x, 0.0)
    
    # TODO: Sum and normalize
    sum_exp = None  # Replace: tl.sum(exp_x, axis=0)
    out = None     # Replace: exp_x / sum_exp
    
    # Store
    out_ptr_row = out_ptr + row * stride
    tl.store(out_ptr_row + cols, out, mask=mask)


def softmax_2d(x: torch.Tensor) -> torch.Tensor:
    """Softmax along last dimension of 2D tensor."""
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    row_softmax_kernel[(M,)](x, out, M, N, x.stride(0), BLOCK_N)
    return out


# ============================================================================
# Exercise 4: Log-Softmax (More Stable)
# ============================================================================

@triton.jit
def log_softmax_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """
    Log-softmax: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
    
    More numerically stable than log(softmax(x)).
    """
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    
    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
    
    # Max for stability
    x_max = tl.max(x, axis=0)
    x_shifted = x - x_max
    
    # Exp and sum
    exp_x = tl.exp(x_shifted)
    exp_x_masked = tl.where(mask, exp_x, 0.0)
    sum_exp = tl.sum(exp_x_masked, axis=0)
    
    # TODO: Log-softmax = x - max - log(sum(exp))
    log_sum_exp = None  # Replace: tl.log(sum_exp)
    out = None          # Replace: x_shifted - log_sum_exp
    
    tl.store(out_ptr + offs, out, mask=mask)


def log_softmax_1d(x: torch.Tensor) -> torch.Tensor:
    """Log-softmax for 1D tensor."""
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    log_softmax_kernel[(1,)](x, out, n, BLOCK)
    return out


# ============================================================================
# Exercise 5: Softmax with Temperature
# ============================================================================

@triton.jit
def softmax_temp_kernel(x_ptr, out_ptr, temperature, n, BLOCK: tl.constexpr):
    """
    Softmax with temperature: softmax(x / T)
    
    T > 1: softer distribution (more uniform)
    T < 1: sharper distribution (more peaked)
    """
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    
    x = tl.load(x_ptr + offs, mask=mask, other=-float('inf'))
    
    # TODO: Scale by temperature
    x_scaled = None  # Replace: x / temperature
    
    # Standard softmax on scaled input
    x_max = tl.max(x_scaled, axis=0)
    x_shifted = x_scaled - x_max
    exp_x = tl.exp(x_shifted)
    exp_x_masked = tl.where(mask, exp_x, 0.0)
    sum_exp = tl.sum(exp_x_masked, axis=0)
    out = exp_x / sum_exp
    
    tl.store(out_ptr + offs, out, mask=mask)


def softmax_temperature(x: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax with temperature."""
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = triton.next_power_of_2(n)
    softmax_temp_kernel[(1,)](x, out, temperature, n, BLOCK)
    return out


if __name__ == "__main__":
    print("Day 13: Fused Softmax")
    print("Run test_day13.py to verify!")
