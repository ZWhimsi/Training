"""
Day 9: Kernel Fusion
====================
Estimated time: 1-2 hours
Prerequisites: Day 8 (vector operations)

Learning objectives:
- Understand why kernel fusion matters for performance
- Fuse multiple operations into single kernels
- Reduce memory bandwidth by avoiding intermediate writes
- Implement common fused operation patterns

Kernel fusion is critical for GPU performance - it reduces memory traffic!
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Why Kernel Fusion?
# ============================================================================
# Without fusion: x -> kernel1 -> temp -> kernel2 -> y (2 memory round trips)
# With fusion:    x -> fused_kernel -> y (1 memory round trip)
#
# Memory bandwidth is often the bottleneck, not compute!
# ============================================================================


# ============================================================================
# Exercise 1: Fused Add-Multiply
# ============================================================================

@triton.jit
def fused_add_mul_kernel(
    x_ptr, y_ptr, z_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute output = (x + y) * z in a single kernel.
    
    Without fusion this would be:
    1. temp = x + y (write to memory)
    2. output = temp * z (read temp, write output)
    
    With fusion: single kernel, no intermediate memory access.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load all three inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    
    # TODO: Compute fused operation (x + y) * z
    # HINT: result = (x + y) * z
    result = None  # Replace
    
    # TODO: Store result
    # HINT: tl.store(output_ptr + offsets, result, mask=mask)
    pass  # Replace


def fused_add_mul(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Wrapper for fused_add_mul_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_add_mul_kernel[grid](x, y, z, output, n_elements, BLOCK_SIZE=1024)
    
    return output


# ============================================================================
# Exercise 2: Fused Bias-ReLU
# ============================================================================

@triton.jit
def fused_bias_relu_kernel(
    x_ptr, bias_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute output = ReLU(x + bias) in a single kernel.
    
    Common pattern in neural networks after linear layers.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load x and bias
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    # TODO: Compute x + bias
    # HINT: y = x + bias
    y = None  # Replace
    
    # TODO: Apply ReLU: max(0, y)
    # HINT: result = tl.maximum(y, 0.0)
    result = None  # Replace
    
    # TODO: Store
    pass  # Replace


def fused_bias_relu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Wrapper for fused_bias_relu_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bias_relu_kernel[grid](x, bias, output, n_elements, BLOCK_SIZE=1024)
    
    return output


# ============================================================================
# Exercise 3: Fused Scale-Shift (BatchNorm-like)
# ============================================================================

@triton.jit
def fused_scale_shift_kernel(
    x_ptr, scale_ptr, shift_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute output = x * scale + shift (affine transformation).
    
    This is the final step of batch normalization.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load all inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr + offsets, mask=mask)
    shift = tl.load(shift_ptr + offsets, mask=mask)
    
    # TODO: Compute affine: x * scale + shift
    result = None  # Replace
    
    # TODO: Store
    pass  # Replace


def fused_scale_shift(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Wrapper for fused_scale_shift_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_scale_shift_kernel[grid](x, scale, shift, output, n_elements, BLOCK_SIZE=1024)
    
    return output


# ============================================================================
# Exercise 4: Fused Residual Add
# ============================================================================

@triton.jit
def fused_residual_dropout_kernel(
    x_ptr, residual_ptr, output_ptr,
    n_elements,
    scale: tl.constexpr,  # For dropout scaling
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute output = x * scale + residual.
    
    Common in transformers: add residual connection after scaled output.
    (Simplified - real dropout would need random mask)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    residual = tl.load(residual_ptr + offsets, mask=mask)
    
    # TODO: Compute scaled addition
    # HINT: result = x * scale + residual
    result = None  # Replace
    
    # TODO: Store
    pass  # Replace


def fused_residual_add(x: torch.Tensor, residual: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Wrapper for fused_residual_dropout_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_residual_dropout_kernel[grid](x, residual, output, n_elements, scale, BLOCK_SIZE=1024)
    
    return output


# ============================================================================
# Exercise 5: Triple Fused Operation
# ============================================================================

@triton.jit
def fused_linear_bias_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute output = ReLU(x * weight + bias).
    
    Element-wise version (real linear would be matrix multiply).
    Fuses three operations: multiply, add, activation.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load all inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    # TODO: Compute x * weight + bias
    linear = None  # Replace
    
    # TODO: Apply ReLU
    result = None  # Replace
    
    # TODO: Store
    pass  # Replace


def fused_linear_bias_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Wrapper for fused_linear_bias_relu_kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_linear_bias_relu_kernel[grid](x, weight, bias, output, n_elements, BLOCK_SIZE=1024)
    
    return output


if __name__ == "__main__":
    print("Day 9: Kernel Fusion")
    print("=" * 50)
    
    # Test setup
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        x = torch.randn(1024, device=device)
        y = torch.randn(1024, device=device)
        z = torch.randn(1024, device=device)
        
        print("\nTesting fused_add_mul:")
        result = fused_add_mul(x, y, z)
        expected = (x + y) * z
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day09.py to verify all implementations!")
