"""
Day 28: Causal Attention
========================
Estimated time: 1-2 hours
Prerequisites: Day 27 (memory-efficient attention)

Learning objectives:
- Implement causal masking in blocked attention
- Handle triangular attention pattern efficiently
- Optimize for autoregressive models
- Skip unnecessary computations
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: Causal Attention
# ============================================================================
# In autoregressive models (GPT), position i can only attend to j <= i.
# This creates a lower triangular attention pattern.
#
# Optimization: skip blocks entirely above the diagonal!
# ============================================================================


# ============================================================================
# Exercise 1: Causal Block Skip
# ============================================================================

@triton.jit
def causal_attention_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    seq_len, head_dim, scale,
    stride_Qs, stride_Qd,
    stride_Ks, stride_Kd,
    stride_Vs, stride_Vd,
    stride_Os, stride_Od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Causal attention with block-level skipping.
    
    For query block starting at m, only process key blocks where
    the block contains at least one position <= max(m block positions).
    """
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    m_mask = offs_m < seq_len
    d_mask = offs_d < head_dim
    
    # Load query block
    q_offs = offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd
    q_mask = m_mask[:, None] & d_mask[None, :]
    q = tl.load(Q_ptr + q_offs, mask=q_mask, other=0.0)
    
    # Running statistics
    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # Maximum position in this query block
    max_m_pos = tl.min((pid_m + 1) * BLOCK_M - 1, seq_len - 1)
    
    # Only process key blocks up to max_m_pos
    for n_start in range(0, max_m_pos + 1, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        # Load K
        k_offs = offs_n[:, None] * stride_Ks + offs_d[None, :] * stride_Kd
        k_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(K_ptr + k_offs, mask=k_mask, other=0.0)
        
        # Compute scores
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # TODO: Apply causal mask within block
        # scores[i, j] = -inf if offs_n[j] > offs_m[i]
        # HINT: causal_mask = offs_n[None, :] > offs_m[:, None]
        causal_mask = None  # Replace
        scores = tl.where(causal_mask, float('-inf'), scores)
        
        # Update running max
        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m, m_block)
        
        # Rescale
        rescale = tl.exp(m - m_new)[:, None]
        acc = acc * rescale
        l = l * tl.exp(m - m_new)
        
        # Weights
        weights = tl.exp(scores - m_new[:, None])
        l = l + tl.sum(weights, axis=1)
        
        # Load V and accumulate
        v_offs = offs_n[:, None] * stride_Vs + offs_d[None, :] * stride_Vd
        v = tl.load(V_ptr + v_offs, mask=k_mask, other=0.0)
        acc += tl.dot(weights, v)
        m = m_new
    
    # Normalize
    output = acc / l[:, None]
    
    # TODO: Store
    o_offs = offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od
    # HINT: tl.store(output_ptr + o_offs, output, mask=q_mask)
    pass  # Replace


def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Causal attention with block skipping."""
    seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.empty_like(Q)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = min(64, head_dim)
    
    grid = (triton.cdiv(seq_len, BLOCK_M),)
    
    causal_attention_kernel[grid](
        Q, K, V, output,
        seq_len, head_dim, scale,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_D
    )
    
    return output


# ============================================================================
# Exercise 2: Multi-Head Causal Attention
# ============================================================================

@triton.jit
def mh_causal_attention_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    n_heads, seq_len, head_dim, scale,
    stride_Qh, stride_Qs, stride_Qd,
    stride_Kh, stride_Ks, stride_Kd,
    stride_Vh, stride_Vs, stride_Vd,
    stride_Oh, stride_Os, stride_Od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Multi-head causal attention."""
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    m_mask = offs_m < seq_len
    d_mask = offs_d < head_dim
    
    # Head pointers
    Q_head = Q_ptr + pid_h * stride_Qh
    K_head = K_ptr + pid_h * stride_Kh
    V_head = V_ptr + pid_h * stride_Vh
    O_head = output_ptr + pid_h * stride_Oh
    
    # Load query
    q_offs = offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd
    q_mask = m_mask[:, None] & d_mask[None, :]
    q = tl.load(Q_head + q_offs, mask=q_mask, other=0.0)
    
    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    max_m_pos = tl.min((pid_m + 1) * BLOCK_M - 1, seq_len - 1)
    
    for n_start in range(0, max_m_pos + 1, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        k_offs = offs_n[:, None] * stride_Ks + offs_d[None, :] * stride_Kd
        k_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(K_head + k_offs, mask=k_mask, other=0.0)
        
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # Causal mask
        causal_mask = offs_n[None, :] > offs_m[:, None]
        scores = tl.where(causal_mask, float('-inf'), scores)
        
        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m, m_block)
        
        rescale = tl.exp(m - m_new)[:, None]
        acc = acc * rescale
        l = l * tl.exp(m - m_new)
        
        weights = tl.exp(scores - m_new[:, None])
        l = l + tl.sum(weights, axis=1)
        
        v_offs = offs_n[:, None] * stride_Vs + offs_d[None, :] * stride_Vd
        v = tl.load(V_head + v_offs, mask=k_mask, other=0.0)
        acc += tl.dot(weights, v)
        m = m_new
    
    output = acc / l[:, None]
    o_offs = offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od
    tl.store(O_head + o_offs, output, mask=q_mask)


def mh_causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Multi-head causal attention."""
    n_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.empty_like(Q)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = min(64, head_dim)
    
    grid = (n_heads, triton.cdiv(seq_len, BLOCK_M))
    
    mh_causal_attention_kernel[grid](
        Q, K, V, output,
        n_heads, seq_len, head_dim, scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_D
    )
    
    return output


if __name__ == "__main__":
    print("Day 28: Causal Attention")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        seq_len, head_dim = 64, 32
        Q = torch.randn(seq_len, head_dim, device=device)
        K = torch.randn(seq_len, head_dim, device=device)
        V = torch.randn(seq_len, head_dim, device=device)
        
        print("\nTesting causal attention:")
        result = causal_attention(Q, K, V)
        
        # Reference
        scale = 1.0 / math.sqrt(head_dim)
        scores = (Q @ K.T) * scale
        mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        expected = torch.softmax(scores, dim=-1) @ V
        
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day28.py to verify!")
