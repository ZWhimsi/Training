"""
Day 27: Memory-Efficient Attention Pattern
==========================================
Estimated time: 1-2 hours
Prerequisites: Day 26 (blocked softmax)

Learning objectives:
- Understand memory bottlenecks in standard attention
- Implement query-by-query attention computation
- Avoid materializing full attention matrix
- Build toward Flash Attention's IO optimization
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: Memory-Efficient Attention
# ============================================================================
# Standard attention materializes (seq_len, seq_len) attention matrix.
# For seq_len=4096, this is 64MB per head in fp32!
#
# Memory-efficient approach: compute one query's output at a time,
# never storing the full attention matrix.
# ============================================================================


# ============================================================================
# Exercise 1: Single Query Attention
# ============================================================================

@triton.jit
def single_query_attention_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    query_idx,
    seq_len, head_dim, scale,
    stride_Qs, stride_Qd,
    stride_Ks, stride_Kd,
    stride_Vs, stride_Vd,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute attention output for a single query position.
    
    Avoids materializing full attention row - processes K/V in blocks.
    """
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load query vector
    q_offs = query_idx * stride_Qs + offs_d * stride_Qd
    q_mask = offs_d < head_dim
    q = tl.load(Q_ptr + q_offs, mask=q_mask, other=0.0)  # [BLOCK_D]
    
    # Running statistics for online softmax
    m = float('-inf')
    l = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    
    # Process K, V in blocks
    for n_start in range(0, seq_len, BLOCK_N):
        n_offs = n_start + offs_n
        n_mask = n_offs < seq_len
        
        # Load K block [BLOCK_N, BLOCK_D]
        k_offs = n_offs[:, None] * stride_Ks + offs_d[None, :] * stride_Kd
        k_mask = n_mask[:, None] & q_mask[None, :]
        k = tl.load(K_ptr + k_offs, mask=k_mask, other=0.0)
        
        # Compute attention scores: q @ k^T -> [BLOCK_N]
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        
        # Update running max
        m_block = tl.max(scores, axis=0)
        m_new = tl.maximum(m, m_block)
        
        # Rescale accumulator
        rescale = tl.exp(m - m_new)
        acc = acc * rescale
        l = l * rescale
        
        # Compute weights
        weights = tl.exp(scores - m_new)
        l = l + tl.sum(weights, axis=0)
        
        # Load V and accumulate
        v_offs = n_offs[:, None] * stride_Vs + offs_d[None, :] * stride_Vd
        v = tl.load(V_ptr + v_offs, mask=k_mask, other=0.0)
        
        acc += tl.sum(weights[:, None] * v, axis=0)
        m = m_new
    
    # Normalize
    output = acc / l
    
    # TODO: Store
    o_offs = query_idx * stride_Qs + offs_d * stride_Qd
    # HINT: tl.store(output_ptr + o_offs, output, mask=q_mask)
    pass  # Replace


def single_query_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                           query_idx: int) -> torch.Tensor:
    """Compute attention output for a single query."""
    seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.empty(head_dim, device=Q.device, dtype=Q.dtype)
    
    # Create output tensor for single query result
    full_output = torch.zeros_like(Q)
    
    BLOCK_N = 32
    BLOCK_D = min(64, head_dim)
    
    single_query_attention_kernel[(1,)](
        Q, K, V, full_output,
        query_idx,
        seq_len, head_dim, scale,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        BLOCK_N, BLOCK_D
    )
    
    return full_output[query_idx]


# ============================================================================
# Exercise 2: Full Memory-Efficient Attention
# ============================================================================

@triton.jit
def memory_efficient_attention_kernel(
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
    Memory-efficient attention for a block of queries.
    
    Each program handles BLOCK_M queries.
    Never materializes full (seq_len, seq_len) attention matrix.
    """
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    m_mask = offs_m < seq_len
    d_mask = offs_d < head_dim
    
    # Load query block [BLOCK_M, BLOCK_D]
    q_offs = offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd
    q_mask = m_mask[:, None] & d_mask[None, :]
    q = tl.load(Q_ptr + q_offs, mask=q_mask, other=0.0)
    
    # Running statistics per query
    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # Process K, V in blocks
    for n_start in range(0, seq_len, BLOCK_N):
        n_offs = n_start + offs_n
        n_mask = n_offs < seq_len
        
        # Load K block [BLOCK_N, BLOCK_D]
        k_offs = n_offs[:, None] * stride_Ks + offs_d[None, :] * stride_Kd
        k_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(K_ptr + k_offs, mask=k_mask, other=0.0)
        
        # Compute scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k)) * scale
        
        # Update max per query
        m_block = tl.max(scores, axis=1)  # [BLOCK_M]
        m_new = tl.maximum(m, m_block)
        
        # Rescale
        rescale = tl.exp(m - m_new)[:, None]
        acc = acc * rescale
        l = l * tl.exp(m - m_new)
        
        # New weights
        weights = tl.exp(scores - m_new[:, None])  # [BLOCK_M, BLOCK_N]
        l = l + tl.sum(weights, axis=1)
        
        # Load V and accumulate
        v_offs = n_offs[:, None] * stride_Vs + offs_d[None, :] * stride_Vd
        v = tl.load(V_ptr + v_offs, mask=k_mask, other=0.0)
        
        acc += tl.dot(weights, v)
        m = m_new
    
    # Normalize
    output = acc / l[:, None]
    
    # Store
    o_offs = offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od
    tl.store(output_ptr + o_offs, output, mask=q_mask)


def memory_efficient_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Full memory-efficient attention."""
    seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.empty_like(Q)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = min(64, head_dim)
    
    grid = (triton.cdiv(seq_len, BLOCK_M),)
    
    memory_efficient_attention_kernel[grid](
        Q, K, V, output,
        seq_len, head_dim, scale,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_D
    )
    
    return output


if __name__ == "__main__":
    print("Day 27: Memory-Efficient Attention")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        seq_len, head_dim = 64, 32
        Q = torch.randn(seq_len, head_dim, device=device)
        K = torch.randn(seq_len, head_dim, device=device)
        V = torch.randn(seq_len, head_dim, device=device)
        
        print("\nTesting memory-efficient attention:")
        result = memory_efficient_attention(Q, K, V)
        
        scale = 1.0 / math.sqrt(head_dim)
        scores = (Q @ K.T) * scale
        expected = torch.softmax(scores, dim=-1) @ V
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day27.py to verify!")
