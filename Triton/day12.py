"""
Day 12: Activation Functions
============================
Estimated time: 1-2 hours
Prerequisites: Day 11 (binary operations)

Learning objectives:
- Implement common activation functions as Triton kernels
- Understand numerical stability in activations
- Implement sigmoid, tanh, leaky ReLU, ELU
- Compare with PyTorch implementations
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# Exercise 1: Sigmoid Activation
# ============================================================================

@triton.jit
def sigmoid_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sigmoid: σ(x) = 1 / (1 + exp(-x))
    
    Numerically stable version handles large negative values.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # API hints:
    # - tl.load(ptr, mask=mask) -> load elements from memory
    # - tl.store(ptr, value, mask=mask) -> store elements to memory
    # - tl.exp(x) -> element-wise exponential
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # TODO: Compute sigmoid: σ(x) = 1 / (1 + exp(-x))
    result = None  # Replace
    
    # TODO: Store result
    pass  # Replace


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for sigmoid_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Exercise 2: Tanh Activation
# ============================================================================

@triton.jit
def tanh_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Tanh: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Or equivalently: 2 * sigmoid(2x) - 1
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # API hints:
    # - tl.load(ptr, mask=mask) -> load elements from memory
    # - tl.store(ptr, value, mask=mask) -> store elements to memory
    # - tl.exp(x) -> element-wise exponential
    # - Can use sigmoid identity: tanh(x) = 2 * sigmoid(2x) - 1
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # TODO: Compute tanh using the sigmoid identity or direct formula
    result = None  # Replace
    
    # TODO: Store result
    pass  # Replace


def tanh(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for tanh_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    tanh_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Exercise 3: Leaky ReLU
# ============================================================================

@triton.jit
def leaky_relu_kernel(
    x_ptr, output_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Leaky ReLU: f(x) = x if x > 0 else negative_slope * x
    
    Allows small gradient for negative inputs (prevents "dying ReLU").
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # API hints:
    # - tl.load(ptr, mask=mask) -> load elements from memory
    # - tl.store(ptr, value, mask=mask) -> store elements to memory
    # - tl.where(condition, x, y) -> select x where condition is true, else y
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # TODO: Compute leaky ReLU: x if x > 0, else negative_slope * x
    result = None  # Replace
    
    # TODO: Store result
    pass  # Replace


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """Wrapper for leaky_relu_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    leaky_relu_kernel[grid](x, output, n_elements, negative_slope, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Exercise 4: ELU (Exponential Linear Unit)
# ============================================================================

@triton.jit
def elu_kernel(
    x_ptr, output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    """
    ELU: f(x) = x if x > 0 else alpha * (exp(x) - 1)
    
    Smooth version of leaky ReLU with exponential negative part.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # API hints:
    # - tl.load(ptr, mask=mask) -> load elements from memory
    # - tl.store(ptr, value, mask=mask) -> store elements to memory
    # - tl.where(condition, x, y) -> select x where condition is true, else y
    # - tl.exp(x) -> element-wise exponential
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # TODO: Compute ELU: x if x > 0, else alpha * (exp(x) - 1)
    result = None  # Replace
    
    # TODO: Store result
    pass  # Replace


def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Wrapper for elu_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    elu_kernel[grid](x, output, n_elements, alpha, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Exercise 5: Softplus
# ============================================================================

@triton.jit
def softplus_kernel(
    x_ptr, output_ptr,
    n_elements,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softplus: f(x) = (1/beta) * log(1 + exp(beta * x))
    
    Smooth approximation to ReLU.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # API hints:
    # - tl.load(ptr, mask=mask) -> load elements from memory
    # - tl.store(ptr, value, mask=mask) -> store elements to memory
    # - tl.exp(x) -> element-wise exponential
    # - tl.log(x) -> element-wise natural logarithm
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # TODO: Compute softplus: (1/beta) * log(1 + exp(beta * x))
    result = None  # Replace
    
    # TODO: Store result
    pass  # Replace


def softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Wrapper for softplus_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    softplus_kernel[grid](x, output, n_elements, beta, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Exercise 6: Mish Activation
# ============================================================================

@triton.jit
def mish_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Mish: f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    
    Modern activation function used in YOLO and other networks.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # API hints:
    # - tl.load(ptr, mask=mask) -> load elements from memory
    # - tl.store(ptr, value, mask=mask) -> store elements to memory
    # - tl.exp(x) -> element-wise exponential
    # - tl.log(x) -> element-wise natural logarithm
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # TODO: Compute mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    # Step 1: Compute softplus(x) = ln(1 + exp(x))
    # Step 2: Compute tanh(softplus(x))
    # Step 3: Compute x * tanh(softplus(x))
    result = None  # Replace
    
    # TODO: Store result
    pass  # Replace


def mish(x: torch.Tensor) -> torch.Tensor:
    """Wrapper for mish_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mish_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    print("Day 12: Activation Functions")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        x = torch.randn(1024, device=device)
        
        print("\nTesting activations:")
        activations = [
            ("sigmoid", sigmoid, torch.sigmoid),
            ("tanh", tanh, torch.tanh),
            ("leaky_relu", lambda t: leaky_relu(t, 0.01), lambda t: torch.nn.functional.leaky_relu(t, 0.01)),
        ]
        
        for name, our_fn, torch_fn in activations:
            result = our_fn(x)
            expected = torch_fn(x)
            err = (result - expected).abs().max().item()
            print(f"  {name}: max error = {err:.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day12.py to verify!")
