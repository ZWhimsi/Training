"""
Day 26: Blocked Softmax
=======================
Estimated time: 1-2 hours
Prerequisites: Day 24 (online softmax), Day 25 (attention output)

Learning objectives:
- Implement block-by-block softmax computation
- Handle inter-block dependencies
- Combine with attention pattern
- Prepare for Flash Attention's memory-efficient approach
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: Blocked Softmax
# ============================================================================
# Process softmax in blocks, tracking:
# - Block maximum
# - Block sum (rescaled when max changes)
# - Partial outputs (rescaled when max changes)
#
# This is the core technique in Flash Attention!
# ============================================================================


# ============================================================================
# Exercise 1: Block-wise Max Reduction
# ============================================================================

@triton.jit
def block_max_kernel(
    input_ptr, max_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute max of each row in blocks.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Initialize max
    row_max = float('-inf')
    
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=float('-inf'))
        block_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, block_max)
    
    # TODO: Store row max
    # HINT: tl.store(max_ptr + row_idx, row_max)
    pass  # Replace


def compute_row_max(x: torch.Tensor) -> torch.Tensor:
    """Compute max of each row."""
    n_rows, n_cols = x.shape
    max_vals = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 1024
    block_max_kernel[(n_rows,)](x, max_vals, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return max_vals


# ============================================================================
# Exercise 2: Blocked Softmax with Output
# ============================================================================

@triton.jit
def blocked_softmax_v_kernel(
    scores_ptr, V_ptr, output_ptr,
    seq_len, head_dim,
    stride_Ss, stride_Sd,
    stride_Vs, stride_Vd,
    stride_Os, stride_Od,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Blocked computation: softmax(scores) @ V
    Process in blocks over sequence dimension.
    
    Key insight: output = sum_j(softmax_weight_j * V_j)
    We can compute this incrementally with rescaling!
    """
    row_idx = tl.program_id(0)  # Query position
    
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Running statistics
    m = float('-inf')  # Running max
    l = 0.0           # Running sum
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)  # Running weighted sum
    
    # Process blocks
    for block_start in range(0, seq_len, BLOCK_N):
        n_offs = block_start + offs_n
        n_mask = n_offs < seq_len
        
        # Load scores for this block
        s_offs = row_idx * stride_Ss + n_offs * stride_Sd
        scores = tl.load(scores_ptr + s_offs, mask=n_mask, other=float('-inf'))
        
        # Update running max
        m_block = tl.max(scores, axis=0)
        m_new = tl.maximum(m, m_block)
        
        # Rescale old accumulator
        scale = tl.exp(m - m_new)
        acc = acc * scale
        l = l * scale
        
        # Compute new weights
        weights = tl.exp(scores - m_new)
        l = l + tl.sum(weights, axis=0)
        
        # Load and accumulate V
        for d_start in range(0, head_dim, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < head_dim
            
            v_offs = n_offs[:, None] * stride_Vs + d_offs[None, :] * stride_Vd
            v_mask = n_mask[:, None] & d_mask[None, :]
            v = tl.load(V_ptr + v_offs, mask=v_mask, other=0.0)
            
            # weights: [BLOCK_N], v: [BLOCK_N, BLOCK_D]
            # weighted_v = weights[:, None] * v -> [BLOCK_N, BLOCK_D]
            # sum over BLOCK_N
            acc += tl.sum(weights[:, None] * v, axis=0)
        
        m = m_new
    
    # Normalize by sum
    output = acc / l
    
    # TODO: Store output
    o_offs = row_idx * stride_Os + offs_d * stride_Od
    o_mask = offs_d < head_dim
    # HINT: tl.store(output_ptr + o_offs, output, mask=o_mask)
    pass  # Replace


def blocked_attention(scores: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Compute softmax(scores) @ V using blocked algorithm."""
    seq_len = scores.shape[0]
    head_dim = V.shape[1]
    output = torch.empty((seq_len, head_dim), device=scores.device, dtype=scores.dtype)
    
    BLOCK_N = 32
    BLOCK_D = min(32, head_dim)
    
    blocked_softmax_v_kernel[(seq_len,)](
        scores, V, output,
        seq_len, head_dim,
        scores.stride(0), scores.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_N, BLOCK_D
    )
    
    return output


# ============================================================================
# Exercise 3: Safe Blocked Softmax (Numerically Stable)
# ============================================================================

@triton.jit
def safe_blocked_softmax_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Two-pass blocked softmax for numerical stability.
    Pass 1: Compute max and sum
    Pass 2: Compute softmax values
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Pass 1: Find max
    row_max = float('-inf')
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Pass 1b: Compute sum
    row_sum = 0.0
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=float('-inf'))
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)
    
    # Pass 2: Compute softmax
    for block_start in range(0, n_cols, BLOCK_SIZE):
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        x = tl.load(input_ptr + row_start + offs, mask=mask, other=float('-inf'))
        softmax = tl.exp(x - row_max) / row_sum
        tl.store(output_ptr + row_start + offs, softmax, mask=mask)


def safe_blocked_softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable blocked softmax."""
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    safe_blocked_softmax_kernel[(n_rows,)](x, output, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    
    return output


if __name__ == "__main__":
    print("Day 26: Blocked Softmax")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        seq_len = 64
        x = torch.randn(seq_len, seq_len, device=device)
        
        print("\nTesting safe blocked softmax:")
        result = safe_blocked_softmax(x)
        expected = torch.softmax(x, dim=-1)
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        print("\nTesting blocked attention:")
        V = torch.randn(seq_len, 32, device=device)
        result = blocked_attention(x, V)
        expected = torch.softmax(x, dim=-1) @ V
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day26.py to verify!")
