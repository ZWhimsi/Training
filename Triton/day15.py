"""
Day 15: Layer Normalization
===========================
Estimated time: 1-2 hours
Prerequisites: Day 14 (cross-entropy)

Learning objectives:
- Understand layer normalization mathematically
- Implement efficient mean and variance computation
- Apply affine transformation (scale and shift)
- Compare with PyTorch LayerNorm
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Layer Normalization
# ============================================================================
# LayerNorm normalizes across features for each sample:
#
# y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
#
# Where:
# - mean and var are computed over the normalized dimensions
# - gamma (weight) and beta (bias) are learnable parameters
# - eps prevents division by zero
#
# Unlike BatchNorm, LayerNorm normalizes each sample independently.
# ============================================================================


# ============================================================================
# Exercise 1: Compute Row Mean
# ============================================================================

@triton.jit
def row_mean_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute mean of each row.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # TODO: Compute mean
    # HINT: mean = tl.sum(x, axis=0) / n_cols
    mean = None  # Replace
    
    # TODO: Store
    # HINT: tl.store(output_ptr + row_idx, mean)
    pass  # Replace


# ============================================================================
# Exercise 2: Compute Row Variance
# ============================================================================

@triton.jit
def row_var_kernel(
    input_ptr, mean_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute variance of each row given the mean.
    var = sum((x - mean)^2) / n
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row and mean
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + row_idx)
    
    # TODO: Compute variance
    # HINT: diff = x - mean
    # HINT: var = tl.sum(diff * diff, axis=0) / n_cols
    diff = None  # Replace
    var = None  # Replace
    
    # TODO: Store
    pass  # Replace


# ============================================================================
# Exercise 3: Full LayerNorm Kernel
# ============================================================================

@triton.jit
def layer_norm_kernel(
    input_ptr, output_ptr,
    weight_ptr, bias_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Complete LayerNorm in a single kernel.
    
    y = (x - mean) / sqrt(var + eps) * weight + bias
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input row
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # TODO: Compute mean
    mean = tl.sum(x, axis=0) / n_cols
    
    # TODO: Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    
    # TODO: Normalize
    # HINT: x_norm = diff / tl.sqrt(var + eps)
    x_norm = None  # Replace
    
    # TODO: Load and apply affine transformation
    # HINT: weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    # HINT: bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    
    # TODO: output = x_norm * weight + bias
    output = None  # Replace
    
    # TODO: Store
    # HINT: tl.store(output_ptr + row_start + col_offsets, output, mask=mask)
    pass  # Replace


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, 
               eps: float = 1e-5) -> torch.Tensor:
    """Apply layer normalization."""
    assert x.dim() == 2
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    layer_norm_kernel[(n_rows,)](
        x, output, weight, bias, n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


# ============================================================================
# Exercise 4: RMSNorm (Root Mean Square Normalization)
# ============================================================================

@triton.jit
def rms_norm_kernel(
    input_ptr, output_ptr,
    weight_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
    
    Simpler than LayerNorm - no mean subtraction, no bias.
    Used in LLaMA and other models.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load input
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)
    
    # TODO: Compute RMS (root mean square)
    # HINT: rms = tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    rms = None  # Replace
    
    # TODO: Normalize
    x_norm = None  # Replace: x / rms
    
    # TODO: Apply weight
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output = None  # Replace: x_norm * weight
    
    # TODO: Store
    pass  # Replace


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply RMS normalization."""
    assert x.dim() == 2
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    rms_norm_kernel[(n_rows,)](
        x, output, weight, n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


# ============================================================================
# Exercise 5: Fused LayerNorm + Residual
# ============================================================================

@triton.jit
def layer_norm_residual_kernel(
    input_ptr, residual_ptr, output_ptr,
    weight_ptr, bias_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused: output = LayerNorm(input + residual)
    
    Common pattern in transformers.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # TODO: Load and add input + residual
    x = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x = x + residual
    
    # Compute LayerNorm
    mean = tl.sum(x, axis=0) / n_cols
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    x_norm = diff / tl.sqrt(var + eps)
    
    # Apply affine
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    output = x_norm * weight + bias
    
    tl.store(output_ptr + row_start + col_offsets, output, mask=mask)


def layer_norm_residual(x: torch.Tensor, residual: torch.Tensor,
                        weight: torch.Tensor, bias: torch.Tensor,
                        eps: float = 1e-5) -> torch.Tensor:
    """Apply LayerNorm(x + residual)."""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    layer_norm_residual_kernel[(n_rows,)](
        x, residual, output, weight, bias, n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


if __name__ == "__main__":
    print("Day 15: Layer Normalization")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        batch_size, hidden = 32, 256
        x = torch.randn(batch_size, hidden, device=device)
        weight = torch.ones(hidden, device=device)
        bias = torch.zeros(hidden, device=device)
        
        print("\nTesting LayerNorm:")
        result = layer_norm(x, weight, bias)
        expected = torch.nn.functional.layer_norm(x, [hidden], weight, bias)
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day15.py to verify!")
