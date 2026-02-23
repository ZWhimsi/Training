"""
Day 24: Online Softmax
======================
Estimated time: 1-2 hours
Prerequisites: Day 23 (attention scores)

Learning objectives:
- Understand online/streaming softmax algorithm
- Implement numerically stable softmax without full row
- Learn the flash attention softmax technique
- Handle running max and sum updates
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: Online Softmax
# ============================================================================
# Standard softmax requires two passes:
# 1. Find max over all elements
# 2. Compute exp(x - max) / sum(exp(x - max))
#
# Online softmax does this in ONE pass using:
# - Running maximum (m)
# - Running sum scaled by max changes (l)
#
# Key insight: when max changes from m_old to m_new:
# sum_new = sum_old * exp(m_old - m_new) + exp(x_new - m_new)
# ============================================================================


# ============================================================================
# Exercise 1: Online Max-Sum Update
# ============================================================================

@triton.jit
def online_softmax_update(
    m_old, l_old,  # Old running max and scaled sum
    x_block,       # New block of values
):
    """
    Update running max and sum for online softmax.
    
    Returns: (m_new, l_new)
    """
    # TODO: Find max of new block
    # HINT: m_block = tl.max(x_block, axis=0)
    m_block = None  # Replace
    
    # TODO: New running max
    # HINT: m_new = tl.maximum(m_old, m_block)
    m_new = None  # Replace
    
    # TODO: Rescale old sum and add new contributions
    # l_new = l_old * exp(m_old - m_new) + sum(exp(x_block - m_new))
    # HINT: l_new = l_old * tl.exp(m_old - m_new) + tl.sum(tl.exp(x_block - m_new), axis=0)
    l_new = None  # Replace
    
    return m_new, l_new


# ============================================================================
# Exercise 2: Row-wise Online Softmax
# ============================================================================

@triton.jit
def online_softmax_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute softmax using online algorithm.
    Processes row in blocks, maintaining running max and sum.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Initialize running statistics
    m = float('-inf')  # Running max
    l = 0.0           # Running sum
    
    # First pass: compute max and sum
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        
        # Load block
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=float('-inf'))
        
        # Update running max
        m_block = tl.max(x, axis=0)
        m_new = tl.maximum(m, m_block)
        
        # Update running sum (rescale old sum for new max)
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    
    # Second pass: compute softmax values
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        
        # Load block
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=float('-inf'))
        
        # TODO: Compute softmax: exp(x - m) / l
        # HINT: softmax = tl.exp(x - m) / l
        softmax = None  # Replace
        
        # TODO: Store
        # HINT: tl.store(output_ptr + row_start + offs, softmax, mask=mask)
        pass  # Replace


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax using online algorithm."""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    online_softmax_kernel[(n_rows,)](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


# ============================================================================
# Exercise 3: Fused Scale + Online Softmax
# ============================================================================

@triton.jit
def scaled_online_softmax_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols, scale,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute softmax(scale * x) using online algorithm.
    Useful for attention where we compute softmax(QK^T / sqrt(d)).
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    m = float('-inf')
    l = 0.0
    
    # First pass with scaling
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=float('-inf'))
        x_scaled = x * scale
        
        m_block = tl.max(x_scaled, axis=0)
        m_new = tl.maximum(m, m_block)
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x_scaled - m_new), axis=0)
        m = m_new
    
    # Second pass
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=float('-inf'))
        x_scaled = x * scale
        
        softmax = tl.exp(x_scaled - m) / l
        tl.store(output_ptr + row_start + offs, softmax, mask=mask)


def scaled_online_softmax(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Compute softmax(scale * x)."""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    scaled_online_softmax_kernel[(n_rows,)](x, output, n_rows, n_cols, scale, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


# ============================================================================
# Exercise 4: Online Softmax with Causal Mask
# ============================================================================

@triton.jit
def causal_online_softmax_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online softmax with causal masking.
    Row i only considers columns 0..i.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Only process up to row_idx + 1 columns
    valid_cols = row_idx + 1
    
    m = float('-inf')
    l = 0.0
    
    # First pass (only valid columns)
    for block_start in range(0, valid_cols, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < valid_cols
        
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=float('-inf'))
        
        m_block = tl.max(x, axis=0)
        m_new = tl.maximum(m, m_block)
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    
    # Second pass: write softmax for valid, 0 for masked
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        is_valid = offs <= row_idx
        
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=float('-inf'))
        
        softmax = tl.where(is_valid, tl.exp(x - m) / l, 0.0)
        tl.store(output_ptr + row_start + offs, softmax, mask=mask)


def causal_online_softmax(x: torch.Tensor) -> torch.Tensor:
    """Online softmax with causal mask."""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    causal_online_softmax_kernel[(n_rows,)](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


if __name__ == "__main__":
    print("Day 24: Online Softmax")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        rows, cols = 64, 256
        x = torch.randn(rows, cols, device=device)
        
        print("\nTesting online softmax:")
        result = online_softmax(x)
        expected = torch.softmax(x, dim=-1)
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        print("\nTesting scaled online softmax:")
        scale = 0.125
        result = scaled_online_softmax(x, scale)
        expected = torch.softmax(x * scale, dim=-1)
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day24.py to verify!")
