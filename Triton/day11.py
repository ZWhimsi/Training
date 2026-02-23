"""
Day 11: Element-wise Binary Operations
======================================
Estimated time: 1-2 hours
Prerequisites: Day 10 (reductions)

Learning objectives:
- Implement binary operations (add, sub, mul, div)
- Handle broadcasting in kernels
- Compare element-wise and broadcasted operations
- Build a library of reusable kernels
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Exercise 1: Vector Addition
# ============================================================================

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise addition: output = x + y
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load x and y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # TODO: Compute sum
    # HINT: result = x + y
    result = None  # Replace
    
    # TODO: Store result
    # HINT: tl.store(output_ptr + offsets, result, mask=mask)
    pass  # Replace


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper for add_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Exercise 2: Vector Subtraction
# ============================================================================

@triton.jit
def sub_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise subtraction: output = x - y
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # TODO: Compute difference
    result = None  # Replace
    
    # TODO: Store
    pass  # Replace


def vector_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper for sub_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    sub_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Exercise 3: Vector Multiplication
# ============================================================================

@triton.jit
def mul_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise multiplication: output = x * y
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # TODO: Compute product
    result = None  # Replace
    
    # TODO: Store
    pass  # Replace


def vector_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper for mul_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Exercise 4: Vector Division
# ============================================================================

@triton.jit
def div_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise division: output = x / y
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # TODO: Compute quotient
    result = None  # Replace
    
    # TODO: Store
    pass  # Replace


def vector_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper for div_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    div_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Exercise 5: Scalar Broadcast Add
# ============================================================================

@triton.jit
def scalar_add_kernel(
    x_ptr, scalar, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Add scalar to vector: output = x + scalar
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # TODO: Add scalar (broadcasts automatically)
    result = None  # Replace: x + scalar
    
    # TODO: Store
    pass  # Replace


def scalar_add(x: torch.Tensor, scalar: float) -> torch.Tensor:
    """Wrapper for scalar_add_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scalar_add_kernel[grid](x, scalar, output, n_elements, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Exercise 6: Maximum of Two Vectors
# ============================================================================

@triton.jit
def maximum_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Element-wise maximum: output = max(x, y)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # TODO: Compute element-wise maximum
    # HINT: result = tl.maximum(x, y)
    result = None  # Replace
    
    # TODO: Store
    pass  # Replace


def vector_maximum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Wrapper for maximum_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    maximum_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    print("Day 11: Element-wise Binary Operations")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        x = torch.randn(1024, device=device)
        y = torch.randn(1024, device=device)
        
        print("\nTesting vector operations:")
        for name, fn, ref in [
            ("add", vector_add, lambda a, b: a + b),
            ("sub", vector_sub, lambda a, b: a - b),
            ("mul", vector_mul, lambda a, b: a * b),
            ("div", vector_div, lambda a, b: a / b),
        ]:
            result = fn(x, y)
            expected = ref(x, y)
            err = (result - expected).abs().max().item()
            print(f"  {name}: max error = {err:.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day11.py to verify!")
