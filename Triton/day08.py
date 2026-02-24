"""
Day 8: Vector Operations
========================
Estimated time: 1-2 hours
Prerequisites: Day 7 (memory patterns)

Learning objectives:
- Implement element-wise vector operations
- Handle different data types
- Combine multiple operations (fusion)
- Use mathematical functions from triton.language
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Exercise 1: GELU Activation
# ============================================================================

@triton.jit
def gelu_kernel(
    x_ptr, out_ptr, n,
    BLOCK: tl.constexpr,
):
    """
    GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    Constants: sqrt(2/pi) = 0.7978845608028654, coeff = 0.044715
    """
    # API hints:
    # - tl.program_id(0) -> block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.tanh(x) -> hyperbolic tangent
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement GELU kernel
    pass


def gelu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    gelu_kernel[grid](x, out, n, BLOCK)
    return out


# ============================================================================
# Exercise 2: SiLU (Swish) Activation
# ============================================================================

@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """
    SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    """
    # API hints:
    # - tl.program_id(0) -> block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.exp(x) -> exponential
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement SiLU kernel
    pass


def silu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    silu_kernel[grid](x, out, n, 1024)
    return out


# ============================================================================
# Exercise 3: Fused Linear + GELU
# ============================================================================

@triton.jit
def fused_linear_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N,  # input/output size
    BLOCK: tl.constexpr,
):
    """
    Fused: out = GELU(x * w + b) for element-wise operation.
    (Not a full linear layer, just element-wise for learning)
    """
    # API hints:
    # - tl.program_id(0) -> block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.tanh(x) -> hyperbolic tangent for GELU
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement fused linear+GELU kernel
    pass


def fused_linear_gelu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    fused_linear_gelu_kernel[grid](x, w, b, out, n, 1024)
    return out


# ============================================================================
# Exercise 4: Exponential Moving Average
# ============================================================================

@triton.jit
def ema_update_kernel(
    running_ptr, new_ptr, out_ptr,
    alpha,  # EMA coefficient
    n,
    BLOCK: tl.constexpr,
):
    """
    EMA: out = (1-alpha) * running + alpha * new
    """
    # API hints:
    # - tl.program_id(0) -> block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement EMA kernel
    pass


def ema_update(running: torch.Tensor, new: torch.Tensor, alpha: float) -> torch.Tensor:
    out = torch.empty_like(running)
    n = running.numel()
    grid = (triton.cdiv(n, 1024),)
    ema_update_kernel[grid](running, new, out, alpha, n, 1024)
    return out


# ============================================================================
# Exercise 5: Polynomial Evaluation
# ============================================================================

@triton.jit
def polynomial_kernel(
    x_ptr, out_ptr,
    a0, a1, a2, a3,  # coefficients: a0 + a1*x + a2*x^2 + a3*x^3
    n,
    BLOCK: tl.constexpr,
):
    """
    Evaluate polynomial using Horner's method.
    Result: a0 + x * (a1 + x * (a2 + x * a3))
    """
    # API hints:
    # - tl.program_id(0) -> block index
    # - tl.arange(start, end) -> create range
    # - tl.load(ptr + offsets, mask=mask) -> load from memory
    # - tl.store(ptr + offsets, value, mask=mask) -> store to memory
    
    # TODO: Implement polynomial kernel using Horner's method
    pass


def polynomial(x: torch.Tensor, coeffs: list) -> torch.Tensor:
    """Evaluate polynomial with coefficients [a0, a1, a2, a3]."""
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    polynomial_kernel[grid](x, out, coeffs[0], coeffs[1], coeffs[2], coeffs[3], n, 1024)
    return out


if __name__ == "__main__":
    print("Day 8: Vector Operations")
    print("Run test_day08.py to verify!")
