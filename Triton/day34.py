"""
Day 34: Flash Attention Backward - dQ
=====================================
Estimated time: 1-2 hours
Prerequisites: Day 31 (dV, dK backward)

Learning objectives:
- Complete the backward pass with dQ computation
- Understand the full backward algorithm
- Combine all gradients efficiently
- Handle numerical stability
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: Computing dQ
# ============================================================================
# dQ = dP @ K  (where dP is the gradient of the attention matrix)
# dP = P ⊙ (dO @ V^T - D), D = rowsum(dO ⊙ O)
#
# Key: recompute P during backward to avoid storing O(N²) matrix
# ============================================================================


@triton.jit
def flash_attention_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    dO_ptr, dQ_ptr,
    L_ptr, M_ptr,
    n_heads, seq_len, head_dim, scale,
    stride_Qb, stride_Qh, stride_Qs, stride_Qd,
    stride_Kb, stride_Kh, stride_Ks, stride_Kd,
    stride_Vb, stride_Vh, stride_Vs, stride_Vd,
    stride_Ob, stride_Oh, stride_Os, stride_Od,
    stride_dQb, stride_dQh, stride_dQs, stride_dQd,
    stride_Lb, stride_Lh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Compute dQ for Flash Attention backward.
    
    Each program handles one block of Q positions.
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)  # Q block
    
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
    dO_base = dO_ptr + pid_batch * stride_Ob + pid_head * stride_Oh
    dQ_base = dQ_ptr + pid_batch * stride_dQb + pid_head * stride_dQh
    L_base = L_ptr + pid_batch * stride_Lb + pid_head * stride_Lh
    M_base = M_ptr + pid_batch * stride_Lb + pid_head * stride_Lh
    
    # Load Q, O, dO for this block
    q_offs = offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd
    q_mask = m_mask[:, None] & d_mask[None, :]
    q = tl.load(Q_base + q_offs, mask=q_mask, other=0.0)
    o = tl.load(O_base + q_offs, mask=q_mask, other=0.0)
    do = tl.load(dO_base + q_offs, mask=q_mask, other=0.0)
    
    # Load L, M for normalization
    l = tl.load(L_base + offs_m, mask=m_mask, other=1.0)
    m = tl.load(M_base + offs_m, mask=m_mask, other=0.0)
    
    # Compute D = rowsum(dO * O)
    d_sum = tl.sum(do * o, axis=1)  # [BLOCK_M]
    
    # Initialize dQ accumulator
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # Iterate over K, V blocks
    for n_start in range(0, seq_len, BLOCK_N):
        n_offs = n_start + offs_n
        n_mask = n_offs < seq_len
        
        # Load K, V
        k_offs = n_offs[:, None] * stride_Ks + offs_d[None, :] * stride_Kd
        k_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(K_base + k_offs, mask=k_mask, other=0.0)
        v = tl.load(V_base + k_offs, mask=k_mask, other=0.0)
        
        # Recompute attention scores and weights
        qk = tl.dot(q, tl.trans(k)) * scale  # [BLOCK_M, BLOCK_N]
        p = tl.exp(qk - m[:, None]) / l[:, None]  # Normalized softmax
        
        # Compute dP = P * (dO @ V^T - D)
        do_v = tl.dot(do, tl.trans(v))  # [BLOCK_M, BLOCK_N]
        dp = p * (do_v - d_sum[:, None])
        
        # Accumulate dQ: dQ += dP @ K * scale
        dq += tl.dot(dp, k) * scale
    
    # Store dQ
    dq_offs = offs_m[:, None] * stride_dQs + offs_d[None, :] * stride_dQd
    tl.store(dQ_base + dq_offs, dq, mask=q_mask)


def flash_attention_backward_dq(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    output: torch.Tensor, dO: torch.Tensor,
    L: torch.Tensor, M: torch.Tensor
) -> torch.Tensor:
    """
    Compute dQ for Flash Attention backward.
    """
    batch, n_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    dQ = torch.empty_like(Q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = min(64, head_dim)
    
    grid = (batch, n_heads, triton.cdiv(seq_len, BLOCK_M))
    
    flash_attention_bwd_dq_kernel[grid](
        Q, K, V, output, dO, dQ, L, M,
        n_heads, seq_len, head_dim, scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        L.stride(0), L.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_D
    )
    
    return dQ


if __name__ == "__main__":
    print("Day 34: Flash Attention Backward - dQ")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        from day33 import flash_attention_v2
        
        batch, n_heads, seq_len, head_dim = 1, 4, 64, 32
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device=device, requires_grad=True)
        K = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        V = torch.randn(batch, n_heads, seq_len, head_dim, device=device)
        
        # Forward
        output, L, M = flash_attention_v2(Q, K, V)
        dO = torch.randn_like(output)
        
        # Our dQ
        dQ = flash_attention_backward_dq(Q.detach(), K, V, output, dO, L, M)
        
        # Reference
        Q_ref = Q.detach().requires_grad_(True)
        scale = 1.0 / math.sqrt(head_dim)
        ref_out = torch.softmax((Q_ref @ K.transpose(-2, -1)) * scale, dim=-1) @ V
        ref_out.backward(dO)
        
        print(f"\ndQ max error: {(dQ - Q_ref.grad).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day34.py to verify!")
