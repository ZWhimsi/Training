"""
Day 31: Flash Attention Backward - dV, dK
=========================================
Estimated time: 1-2 hours
Prerequisites: Day 30 (Flash Attention forward)

Learning objectives:
- Understand backward pass for attention
- Compute gradients for V and K
- Recompute attention weights during backward
- Handle online algorithm in reverse
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: Attention Backward
# ============================================================================
# Given: dO (gradient of output), Q, K, V, stored L (log-sum-exp)
# Compute: dQ, dK, dV
#
# Key insight: don't store attention matrix!
# Recompute P = softmax(QK^T/√d) during backward.
#
# dV = P^T @ dO
# dK = (dP @ Q) where dP = P ⊙ (dO @ V^T - D), D = rowsum(dO ⊙ O)
# dQ = dP @ K (handled in day 35)
# ============================================================================


@triton.jit
def flash_attention_bwd_dv_dk_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    dO_ptr, dK_ptr, dV_ptr,
    L_ptr,  # Log-sum-exp from forward
    n_heads, seq_len, head_dim, scale,
    stride_Qb, stride_Qh, stride_Qs, stride_Qd,
    stride_Kb, stride_Kh, stride_Ks, stride_Kd,
    stride_Vb, stride_Vh, stride_Vs, stride_Vd,
    stride_Ob, stride_Oh, stride_Os, stride_Od,
    stride_dKb, stride_dKh, stride_dKs, stride_dKd,
    stride_dVb, stride_dVh, stride_dVs, stride_dVd,
    stride_Lb, stride_Lh, stride_Ls,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute dK and dV for Flash Attention backward.
    
    Each program handles one block of K, V positions.
    Iterates over all Q positions to accumulate gradients.
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_n = tl.program_id(2)  # K/V block
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_m = tl.arange(0, BLOCK_M)
    
    n_mask = offs_n < seq_len
    d_mask = offs_d < head_dim
    
    # Base pointers
    Q_base = Q_ptr + pid_batch * stride_Qb + pid_head * stride_Qh
    K_base = K_ptr + pid_batch * stride_Kb + pid_head * stride_Kh
    V_base = V_ptr + pid_batch * stride_Vb + pid_head * stride_Vh
    O_base = output_ptr + pid_batch * stride_Ob + pid_head * stride_Oh
    dO_base = dO_ptr + pid_batch * stride_Ob + pid_head * stride_Oh
    dK_base = dK_ptr + pid_batch * stride_dKb + pid_head * stride_dKh
    dV_base = dV_ptr + pid_batch * stride_dVb + pid_head * stride_dVh
    L_base = L_ptr + pid_batch * stride_Lb + pid_head * stride_Lh
    
    # Load K, V blocks
    k_offs = offs_n[:, None] * stride_Ks + offs_d[None, :] * stride_Kd
    k_mask = n_mask[:, None] & d_mask[None, :]
    k = tl.load(K_base + k_offs, mask=k_mask, other=0.0)
    v = tl.load(V_base + k_offs, mask=k_mask, other=0.0)
    
    # Initialize gradient accumulators
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    
    # Iterate over Q blocks
    for m_start in range(0, seq_len, BLOCK_M):
        m_offs = m_start + offs_m
        m_mask = m_offs < seq_len
        
        # Load Q block
        q_offs = m_offs[:, None] * stride_Qs + offs_d[None, :] * stride_Qd
        q_mask = m_mask[:, None] & d_mask[None, :]
        q = tl.load(Q_base + q_offs, mask=q_mask, other=0.0)
        
        # Load output and dO
        o = tl.load(O_base + q_offs, mask=q_mask, other=0.0)
        do = tl.load(dO_base + q_offs, mask=q_mask, other=0.0)
        
        # Load log-sum-exp
        l = tl.load(L_base + m_offs * stride_Ls, mask=m_mask, other=0.0)
        
        # Recompute attention weights P
        qk = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]
        p = tl.exp(qk - l[:, None])  # Softmax using stored L
        
        # Compute D = rowsum(dO * O)
        d = tl.sum(do * o, axis=1)  # [BLOCK_M]
        
        # Compute dP = P * (dO @ V^T - D)
        do_v = tl.dot(do, tl.trans(v))  # [BLOCK_M, BLOCK_N]
        dp = p * (do_v - d[:, None])
        
        # Accumulate dV: dV += P^T @ dO
        dv += tl.dot(tl.trans(p), do)
        
        # Accumulate dK: dK += dP^T @ Q * scale
        dk += tl.dot(tl.trans(dp), q) * scale
    
    # Store gradients
    dk_offs = offs_n[:, None] * stride_dKs + offs_d[None, :] * stride_dKd
    dv_offs = offs_n[:, None] * stride_dVs + offs_d[None, :] * stride_dVd
    
    tl.store(dK_base + dk_offs, dk, mask=k_mask)
    tl.store(dV_base + dv_offs, dv, mask=k_mask)


def flash_attention_backward_dv_dk(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    output: torch.Tensor, dO: torch.Tensor, L: torch.Tensor
) -> tuple:
    """
    Compute dK and dV for Flash Attention backward.
    
    Returns:
        dK, dV
    """
    batch, n_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = min(64, head_dim)
    
    grid = (batch, n_heads, triton.cdiv(seq_len, BLOCK_N))
    
    flash_attention_bwd_dv_dk_kernel[grid](
        Q, K, V, output, dO, dK, dV, L,
        n_heads, seq_len, head_dim, scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_D
    )
    
    return dK, dV


if __name__ == "__main__":
    print("Day 31: Flash Attention Backward - dV, dK")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        from day30 import flash_attention_forward
        
        batch, n_heads, seq_len, head_dim = 1, 4, 64, 32
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        K = torch.randn(batch, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        V = torch.randn(batch, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        
        # Forward
        output, L = flash_attention_forward(Q, K, V)
        dO = torch.randn_like(output)
        
        # Our backward
        dK, dV = flash_attention_backward_dv_dk(Q, K, V, output, dO, L)
        
        # Reference backward
        scale = 1.0 / math.sqrt(head_dim)
        scores = (Q @ K.transpose(-2, -1)) * scale
        P = torch.softmax(scores, dim=-1)
        ref_output = P @ V
        ref_output.backward(dO)
        
        print("\nComparing gradients:")
        print(f"  dK max error: {(dK - K.grad).abs().max().item():.6f}")
        print(f"  dV max error: {(dV - V.grad).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day31.py to verify!")
