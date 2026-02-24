"""
Day 29: IO-Aware Attention
==========================
Estimated time: 1-2 hours
Prerequisites: Day 28 (causal attention)

Learning objectives:
- Understand memory hierarchy (SRAM vs HBM)
- Minimize HBM reads/writes
- Tile computation for SRAM reuse
- Core concepts behind Flash Attention
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: IO Complexity
# ============================================================================
# Standard attention:
# - Q, K, V: O(N * d) reads from HBM
# - Scores (N, N): O(N²) write + read
# - Output: O(N * d) write
# Total: O(N * d + N²) memory accesses
#
# Flash Attention:
# - Q, K, V: O(N * d) reads (same)
# - No intermediate N² storage!
# - Output: O(N * d) write
# Total: O(N * d) memory accesses
#
# The key: recompute attention during backward instead of storing!
# ============================================================================


# ============================================================================
# Exercise 1: Tiled Q-K-V Attention
# ============================================================================

@triton.jit
def io_aware_attention_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    seq_len, head_dim, scale,
    stride_Qs, stride_Qd,
    stride_Ks, stride_Kd,
    stride_Vs, stride_Vd,
    stride_Os, stride_Od,
    BLOCK_M: tl.constexpr,  # Query block size
    BLOCK_N: tl.constexpr,  # Key/Value block size
    BLOCK_D: tl.constexpr,  # Head dim block size
):
    """
    IO-aware attention that minimizes HBM accesses.
    
    Each thread block:
    1. Loads a tile of Q into SRAM
    2. Iterates over K, V tiles
    3. Accumulates output in registers
    4. Writes output once to HBM
    """
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    m_mask = offs_m < seq_len
    d_mask = offs_d < head_dim
    
    # Load Q tile ONCE into SRAM/registers
    q_offs = offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd
    q_mask = m_mask[:, None] & d_mask[None, :]
    q = tl.load(Q_ptr + q_offs, mask=q_mask, other=0.0)
    
    # Running statistics
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # Iterate over K, V tiles
    for n_start in range(0, seq_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        # Load K tile
        k_offs = offs_n[:, None] * stride_Ks + offs_d[None, :] * stride_Kd
        k_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(K_ptr + k_offs, mask=k_mask, other=0.0)
        
        # Compute QK^T for this tile
        qk = tl.dot(q, tl.trans(k)) * scale
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        # Rescale previous accumulator
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        
        # Compute new softmax weights (unnormalized)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i + tl.sum(p, axis=1)
        
        # Load V tile and accumulate
        v_offs = offs_n[:, None] * stride_Vs + offs_d[None, :] * stride_Vd
        v = tl.load(V_ptr + v_offs, mask=k_mask, other=0.0)
        acc += tl.dot(p, v)
        
        m_i = m_new
    
    # Normalize
    output = acc / l_i[:, None]
    
    # Single write to HBM
    o_offs = offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od
    tl.store(output_ptr + o_offs, output, mask=q_mask)


def io_aware_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """IO-optimized attention."""
    seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.empty_like(Q)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = min(64, head_dim)
    
    grid = (triton.cdiv(seq_len, BLOCK_M),)
    
    io_aware_attention_kernel[grid](
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
# Exercise 2: Store Statistics for Backward
# ============================================================================

@triton.jit
def attention_with_stats_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    L_ptr, M_ptr,  # Store logsumexp and max for backward
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
    Attention that stores statistics needed for backward pass.
    
    Stores:
    - M[i]: row max (for numerical stability in backward)
    - L[i]: row sum (for normalization in backward)
    """
    pid_m = tl.program_id(0)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    m_mask = offs_m < seq_len
    d_mask = offs_d < head_dim
    
    q_offs = offs_m[:, None] * stride_Qs + offs_d[None, :] * stride_Qd
    q_mask = m_mask[:, None] & d_mask[None, :]
    q = tl.load(Q_ptr + q_offs, mask=q_mask, other=0.0)
    
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    for n_start in range(0, seq_len, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq_len
        
        k_offs = offs_n[:, None] * stride_Ks + offs_d[None, :] * stride_Kd
        k_mask = n_mask[:, None] & d_mask[None, :]
        k = tl.load(K_ptr + k_offs, mask=k_mask, other=0.0)
        
        qk = tl.dot(q, tl.trans(k)) * scale
        
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha
        
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i + tl.sum(p, axis=1)
        
        v_offs = offs_n[:, None] * stride_Vs + offs_d[None, :] * stride_Vd
        v = tl.load(V_ptr + v_offs, mask=k_mask, other=0.0)
        acc += tl.dot(p, v)
        
        m_i = m_new
    
    output = acc / l_i[:, None]
    
    # Store output
    o_offs = offs_m[:, None] * stride_Os + offs_d[None, :] * stride_Od
    tl.store(output_ptr + o_offs, output, mask=q_mask)
    
    # TODO: Store statistics (max and sum) needed for backward pass
    # API hints:
    # - tl.store(ptr + offsets, values, mask=mask) -> store with bounds checking
    # - Store m_i (row maxes) to M_ptr
    # - Store l_i (row sums) to L_ptr
    pass


def attention_with_stats(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """Attention that returns statistics for backward."""
    seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    output = torch.empty_like(Q)
    L = torch.empty(seq_len, device=Q.device, dtype=Q.dtype)
    M = torch.empty(seq_len, device=Q.device, dtype=Q.dtype)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = min(64, head_dim)
    
    grid = (triton.cdiv(seq_len, BLOCK_M),)
    
    attention_with_stats_kernel[grid](
        Q, K, V, output, L, M,
        seq_len, head_dim, scale,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_D
    )
    
    return output, L, M


if __name__ == "__main__":
    print("Day 29: IO-Aware Attention")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        seq_len, head_dim = 128, 64
        Q = torch.randn(seq_len, head_dim, device=device)
        K = torch.randn(seq_len, head_dim, device=device)
        V = torch.randn(seq_len, head_dim, device=device)
        
        print("\nTesting IO-aware attention:")
        result = io_aware_attention(Q, K, V)
        
        scale = 1.0 / math.sqrt(head_dim)
        expected = torch.softmax((Q @ K.T) * scale, dim=-1) @ V
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day29.py to verify!")
