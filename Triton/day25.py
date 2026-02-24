"""
Day 25: Attention Output Computation
====================================
Estimated time: 1-2 hours
Prerequisites: Day 24 (online softmax)

Learning objectives:
- Compute attention output: softmax(scores) @ V
- Combine scores computation and output in one pass
- Understand the full attention pipeline
- Optimize memory access patterns
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: Attention Output
# ============================================================================
# Full attention: output = softmax(Q @ K^T / sqrt(d_k)) @ V
#
# After computing attention weights (softmax), we multiply by V:
# output[i] = sum_j(attention_weights[i, j] * V[j])
#
# Each output position is a weighted combination of value vectors.
# ============================================================================


# ============================================================================
# Exercise 1: Attention Weights @ V
# ============================================================================

@triton.jit
def attn_output_kernel(
    weights_ptr, V_ptr, output_ptr,
    seq_len, head_dim,
    stride_Ws, stride_Wd,
    stride_Vs, stride_Vd,
    stride_Os, stride_Od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute output = attention_weights @ V
    weights: (seq_len, seq_len) - softmax attention weights
    V: (seq_len, head_dim)
    output: (seq_len, head_dim)
    """
    pid_m = tl.program_id(0)  # Output row
    pid_d = tl.program_id(1)  # Output dim block
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # Loop over sequence positions
    for n_start in range(0, seq_len, BLOCK_N):
        n_offs = n_start + offs_n
        
        # Load weights block [BLOCK_M, BLOCK_N]
        w_offs = offs_m[:, None] * stride_Ws + n_offs[None, :] * stride_Wd
        w_mask = (offs_m[:, None] < seq_len) & (n_offs[None, :] < seq_len)
        weights = tl.load(weights_ptr + w_offs, mask=w_mask, other=0.0)
        
        # Load V block [BLOCK_N, BLOCK_D]
        v_offs = n_offs[:, None] * stride_Vs + offs_d[None, :] * stride_Vd
        v_mask = (n_offs[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(V_ptr + v_offs, mask=v_mask, other=0.0)
        
        # Accumulate: [BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D]
        acc += tl.dot(weights, v)
    
    # TODO: Store accumulated output to memory
    # API hints:
    # - tl.store(ptr + offsets, values, mask=mask) -> store with bounds checking
    o_offs = offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    pass


def attention_output(weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Compute attention output = weights @ V."""
    seq_len = weights.shape[0]
    head_dim = V.shape[1]
    output = torch.empty((seq_len, head_dim), device=weights.device, dtype=weights.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = min(32, head_dim)
    
    grid = (triton.cdiv(seq_len, BLOCK_M), triton.cdiv(head_dim, BLOCK_D))
    
    attn_output_kernel[grid](
        weights, V, output,
        seq_len, head_dim,
        weights.stride(0), weights.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_D
    )
    
    return output


# ============================================================================
# Exercise 2: Multi-Head Attention Output
# ============================================================================

@triton.jit
def mha_output_kernel(
    weights_ptr, V_ptr, output_ptr,
    n_heads, seq_len, head_dim,
    stride_Wh, stride_Ws, stride_Wd,
    stride_Vh, stride_Vs, stride_Vd,
    stride_Oh, stride_Os, stride_Od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Multi-head attention output.
    weights: (n_heads, seq_len, seq_len)
    V: (n_heads, seq_len, head_dim)
    output: (n_heads, seq_len, head_dim)
    """
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_d = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Head offsets
    W_head = weights_ptr + pid_h * stride_Wh
    V_head = V_ptr + pid_h * stride_Vh
    O_head = output_ptr + pid_h * stride_Oh
    
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    for n_start in range(0, seq_len, BLOCK_N):
        n_offs = n_start + offs_n
        
        w_offs = offs_m[:, None] * stride_Ws + n_offs[None, :] * stride_Wd
        w_mask = (offs_m[:, None] < seq_len) & (n_offs[None, :] < seq_len)
        weights = tl.load(W_head + w_offs, mask=w_mask, other=0.0)
        
        v_offs = n_offs[:, None] * stride_Vs + offs_d[None, :] * stride_Vd
        v_mask = (n_offs[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(V_head + v_offs, mask=v_mask, other=0.0)
        
        acc += tl.dot(weights, v)
    
    o_offs = offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(O_head + o_offs, acc, mask=o_mask)


def mha_attention_output(weights: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Multi-head attention output."""
    n_heads, seq_len, _ = weights.shape
    head_dim = V.shape[2]
    output = torch.empty((n_heads, seq_len, head_dim), device=weights.device, dtype=weights.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = min(32, head_dim)
    
    grid = (n_heads, triton.cdiv(seq_len, BLOCK_M), triton.cdiv(head_dim, BLOCK_D))
    
    mha_output_kernel[grid](
        weights, V, output,
        n_heads, seq_len, head_dim,
        weights.stride(0), weights.stride(1), weights.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_D
    )
    
    return output


# ============================================================================
# Exercise 3: Full Single-Head Attention
# ============================================================================

def full_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Complete single-head attention.
    
    TODO: Combine score computation, softmax, and output.
    
    This is a stepping stone to Flash Attention!
    """
    seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    # Step 1: Compute scores
    scores = (Q @ K.T) * scale
    
    # Step 2: Softmax
    weights = torch.softmax(scores, dim=-1)
    
    # Step 3: Output
    # TODO: Compute attention output using weights and V
    # API hints:
    # - attention_output(weights, V) -> multiply attention weights by values
    output = None
    
    return output


# ============================================================================
# Exercise 4: Full Multi-Head Attention
# ============================================================================

def full_mha(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Complete multi-head attention.
    Q, K, V: (n_heads, seq_len, head_dim)
    """
    n_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    # Compute scores for all heads
    scores = torch.bmm(Q, K.transpose(-2, -1)) * scale
    
    # Softmax per head
    weights = torch.softmax(scores, dim=-1)
    
    # TODO: Compute multi-head attention output
    # API hints:
    # - mha_attention_output(weights, V) -> batched attention output for all heads
    output = None
    
    return output


if __name__ == "__main__":
    print("Day 25: Attention Output Computation")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        seq_len, head_dim = 64, 32
        weights = torch.softmax(torch.randn(seq_len, seq_len, device=device), dim=-1)
        V = torch.randn(seq_len, head_dim, device=device)
        
        print("\nTesting attention output:")
        result = attention_output(weights, V)
        expected = weights @ V
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        print("\nTesting multi-head attention output:")
        n_heads = 8
        weights_mh = torch.softmax(torch.randn(n_heads, seq_len, seq_len, device=device), dim=-1)
        V_mh = torch.randn(n_heads, seq_len, head_dim, device=device)
        result = mha_attention_output(weights_mh, V_mh)
        expected = torch.bmm(weights_mh, V_mh)
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day25.py to verify!")
