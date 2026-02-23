"""
Day 33: Flash Attention Forward - Production Ready
==================================================
Estimated time: 1-2 hours
Prerequisites: Day 32 (Flash Attention forward basic)

Learning objectives:
- Optimize Flash Attention for production
- Add dropout support
- Handle variable sequence lengths
- Tune block sizes for different GPUs
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: Production Flash Attention
# ============================================================================
# Additional features for real-world use:
# 1. Dropout in attention weights
# 2. Attention bias/mask support
# 3. Variable sequence lengths (padding)
# 4. Mixed precision (fp16/bf16)
# ============================================================================


@triton.jit
def flash_attn_fwd_v2_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    L_ptr, M_ptr,  # Store both for backward
    bias_ptr,  # Optional attention bias
    dropout_mask_ptr,  # Optional dropout mask
    n_heads, seq_len, head_dim, scale,
    dropout_p,
    has_bias: tl.constexpr,
    has_dropout: tl.constexpr,
    stride_Qb, stride_Qh, stride_Qs, stride_Qd,
    stride_Kb, stride_Kh, stride_Ks, stride_Kd,
    stride_Vb, stride_Vh, stride_Vs, stride_Vd,
    stride_Ob, stride_Oh, stride_Os, stride_Od,
    stride_Lb, stride_Lh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Production-ready Flash Attention with:
    - Optional bias
    - Optional dropout
    - Causal masking
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    m_mask = offs_m < seq_len
    d_mask = offs_d < head_dim
    
    # Base pointers
    Q_base = Q_ptr + pid_batch * stride_Qb + pid_head * stride_Qh
    K_base = K_ptr + pid_batch * stride_Kb + pid_head * stride_Kh
    V_base = V_ptr + pid_batch * stride_Vb + pid_head * stride_Vh
    O_base = output_ptr + pid_batch * stride_Ob + pid_head * stride_Oh
    L_base = L_ptr + pid_batch * stride_Lb + pid_head * stride_Lh
    M_base = M_ptr + pid_batch * stride_Lb + pid_head * stride_Lh
    
    # Load Q
    q_offs = offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd
    q_mask = m_mask[:, None] & d_mask[None, :]
    q = tl.load(Q_base + q_offs, mask=q_mask, other=0.0)
    
    # Initialize
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # Loop bounds
    if CAUSAL:
        n_end = min((pid_m + 1) * BLOCK_M, seq_len)
    else:
        n_end = seq_len
    
    for n_start in range(0, n_end, BLOCK_N):
        n_offs = n_start + offs_n
        n_mask = n_offs < seq_len
        
        # Load K
        k_offs = n_offs[:, None] * stride_Ks + offs_d[None, :] * stride_Kd
        k_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(K_base + k_offs, mask=k_mask, other=0.0)
        
        # Compute scores
        qk = tl.dot(q, tl.trans(k)) * scale
        
        # Add bias if present
        if has_bias:
            # Simplified: assume bias is (seq_len, seq_len)
            pass  # TODO: Load and add bias
        
        # Causal mask
        if CAUSAL:
            causal_mask = n_offs[None, :] > offs_m[:, None]
            qk = tl.where(causal_mask, float('-inf'), qk)
        
        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        
        p = tl.exp(qk - m_new[:, None])
        
        # Apply dropout if present
        if has_dropout:
            # Simplified: would need proper random mask
            p = p * (1.0 / (1.0 - dropout_p))
        
        l_i = l_i + tl.sum(p, axis=1)
        
        # Load V and accumulate
        v_offs = n_offs[:, None] * stride_Vs + offs_d[None, :] * stride_Vd
        v = tl.load(V_base + v_offs, mask=k_mask, other=0.0)
        acc += tl.dot(p, v)
        
        m_i = m_new
    
    # Normalize
    output = acc / l_i[:, None]
    
    # Store output
    o_offs = offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od
    tl.store(O_base + o_offs, output, mask=q_mask)
    
    # Store statistics
    tl.store(L_base + offs_m, l_i, mask=m_mask)
    tl.store(M_base + offs_m, m_i, mask=m_mask)


def flash_attention_v2(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    causal: bool = False,
    dropout_p: float = 0.0,
    attention_bias: torch.Tensor = None
) -> tuple:
    """
    Production Flash Attention.
    
    Args:
        Q, K, V: (batch, heads, seq_len, head_dim)
        causal: Apply causal mask
        dropout_p: Dropout probability
        attention_bias: Optional (batch, heads, seq_len, seq_len) or broadcastable
    
    Returns:
        output, L, M
    """
    batch, n_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    output = torch.empty_like(Q)
    L = torch.empty((batch, n_heads, seq_len), device=Q.device, dtype=Q.dtype)
    M = torch.empty((batch, n_heads, seq_len), device=Q.device, dtype=Q.dtype)
    
    has_bias = attention_bias is not None
    has_dropout = dropout_p > 0
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = min(64, head_dim)
    
    grid = (batch, n_heads, triton.cdiv(seq_len, BLOCK_M))
    
    # Placeholder pointers for optional tensors
    bias_ptr = attention_bias if has_bias else Q
    dropout_ptr = Q  # Placeholder
    
    flash_attn_fwd_v2_kernel[grid](
        Q, K, V, output, L, M,
        bias_ptr, dropout_ptr,
        n_heads, seq_len, head_dim, scale,
        dropout_p,
        has_bias, has_dropout,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        L.stride(0), L.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_D,
        CAUSAL=causal
    )
    
    return output, L, M


if __name__ == "__main__":
    print("Day 33: Flash Attention v2 - Production Ready")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        batch, n_heads, seq_len, head_dim = 2, 8, 256, 64
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        K = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        V = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        
        print("\nTesting Flash Attention v2:")
        output, L, M = flash_attention_v2(Q, K, V, causal=False)
        
        scale = 1.0 / math.sqrt(head_dim)
        expected = torch.softmax((Q @ K.transpose(-2, -1)) * scale, dim=-1) @ V
        print(f"  Max error: {(output - expected).abs().max().item():.6f}")
        
        print("\nTesting causal:")
        output, _, _ = flash_attention_v2(Q, K, V, causal=True)
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        scores = (Q @ K.transpose(-2, -1)) * scale
        scores.masked_fill_(mask, float('-inf'))
        expected = torch.softmax(scores, dim=-1) @ V
        print(f"  Max error: {(output - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day33.py to verify!")
