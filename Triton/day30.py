"""
Day 30: Flash Attention Forward - Optimized
============================================
Estimated time: 1-2 hours
Prerequisites: Day 29 (IO-aware attention)

Learning objectives:
- Combine all optimizations into Flash Attention
- Multi-head batched implementation
- Optimal block sizes
- Handle edge cases
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: Flash Attention
# ============================================================================
# Flash Attention combines:
# 1. Tiled computation (fits in SRAM)
# 2. Online softmax (single pass)
# 3. No intermediate N² matrix storage
# 4. Recomputation in backward (not stored)
#
# Result: O(N) memory instead of O(N²), faster than naive attention!
# ============================================================================


@triton.jit
def flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    L_ptr,  # Log-sum-exp for backward
    n_heads, seq_len, head_dim, scale,
    stride_Qb, stride_Qh, stride_Qs, stride_Qd,
    stride_Kb, stride_Kh, stride_Ks, stride_Kd,
    stride_Vb, stride_Vh, stride_Vs, stride_Vd,
    stride_Ob, stride_Oh, stride_Os, stride_Od,
    stride_Lb, stride_Lh, stride_Ls,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Optimized Flash Attention forward pass.
    
    Handles: batched, multi-head, optional causal mask.
    """
    # Program indices
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    m_mask = offs_m < seq_len
    d_mask = offs_d < head_dim
    
    # Base pointers for this batch and head
    Q_base = Q_ptr + pid_batch * stride_Qb + pid_head * stride_Qh
    K_base = K_ptr + pid_batch * stride_Kb + pid_head * stride_Kh
    V_base = V_ptr + pid_batch * stride_Vb + pid_head * stride_Vh
    O_base = output_ptr + pid_batch * stride_Ob + pid_head * stride_Oh
    L_base = L_ptr + pid_batch * stride_Lb + pid_head * stride_Lh
    
    # Load Q block
    q_offs = offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd
    q_mask = m_mask[:, None] & d_mask[None, :]
    q = tl.load(Q_base + q_offs, mask=q_mask, other=0.0)
    
    # Initialize accumulators
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # Determine loop bounds for causal
    if CAUSAL:
        n_end = min((pid_m + 1) * BLOCK_M, seq_len)
    else:
        n_end = seq_len
    
    # Main loop over K, V blocks
    for n_start in range(0, n_end, BLOCK_N):
        n_offs = n_start + offs_n
        n_mask = n_offs < seq_len
        
        # Load K block
        k_offs = n_offs[:, None] * stride_Ks + offs_d[None, :] * stride_Kd
        k_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(K_base + k_offs, mask=k_mask, other=0.0)
        
        # Compute attention scores
        qk = tl.dot(q, tl.trans(k)) * scale
        
        # Apply causal mask if needed
        if CAUSAL:
            causal_mask = n_offs[None, :] > offs_m[:, None]
            qk = tl.where(causal_mask, float('-inf'), qk)
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i + tl.sum(p, axis=1)
        
        # Load V and accumulate
        v_offs = n_offs[:, None] * stride_Vs + offs_d[None, :] * stride_Vd
        v = tl.load(V_base + v_offs, mask=k_mask, other=0.0)
        acc += tl.dot(p, v)
        
        m_i = m_new
    
    # Final normalization
    output = acc / l_i[:, None]
    
    # Store output
    o_offs = offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od
    tl.store(O_base + o_offs, output, mask=q_mask)
    
    # Store log-sum-exp for backward
    lse = m_i + tl.log(l_i)
    tl.store(L_base + offs_m * stride_Ls, lse, mask=m_mask)


def flash_attention_forward(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    causal: bool = False
) -> tuple:
    """
    Flash Attention forward pass.
    
    Args:
        Q, K, V: (batch, heads, seq_len, head_dim)
        causal: Whether to apply causal mask
    
    Returns:
        output: (batch, heads, seq_len, head_dim)
        L: log-sum-exp for backward (batch, heads, seq_len)
    """
    batch, n_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    output = torch.empty_like(Q)
    L = torch.empty((batch, n_heads, seq_len), device=Q.device, dtype=Q.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = min(64, head_dim)
    
    grid = (batch, n_heads, triton.cdiv(seq_len, BLOCK_M))
    
    flash_attention_fwd_kernel[grid](
        Q, K, V, output, L,
        n_heads, seq_len, head_dim, scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_D,
        CAUSAL=causal
    )
    
    return output, L


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    causal: bool = False) -> torch.Tensor:
    """Simplified interface returning just output."""
    output, _ = flash_attention_forward(Q, K, V, causal)
    return output


if __name__ == "__main__":
    print("Day 30: Flash Attention Forward - Optimized")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        K = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        V = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        
        print("\nTesting Flash Attention (non-causal):")
        result = flash_attention(Q, K, V, causal=False)
        
        scale = 1.0 / math.sqrt(head_dim)
        expected = torch.softmax((Q @ K.transpose(-2, -1)) * scale, dim=-1) @ V
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        print("\nTesting Flash Attention (causal):")
        result = flash_attention(Q, K, V, causal=True)
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        scores = (Q @ K.transpose(-2, -1)) * scale
        scores.masked_fill_(mask, float('-inf'))
        expected = torch.softmax(scores, dim=-1) @ V
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day30.py to verify!")
