"""
Day 23: Attention Score Computation
===================================
Estimated time: 1-2 hours
Prerequisites: Day 22 (batch matmul)

Learning objectives:
- Compute Q @ K^T attention scores
- Apply scaling factor
- Understand attention score matrix structure
- Prepare for softmax and masking
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: Attention Scores
# ============================================================================
# Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
#
# Step 1: Compute scores = Q @ K^T
# Step 2: Scale by 1/sqrt(d_k) 
# Step 3: Apply mask (optional, for causal attention)
# Step 4: Apply softmax
# Step 5: Multiply by V
#
# This day focuses on Steps 1-3: computing scaled, masked attention scores.
# ============================================================================


# ============================================================================
# Exercise 1: Basic Attention Scores
# ============================================================================

@triton.jit
def attention_scores_kernel(
    Q_ptr, K_ptr, scores_ptr,
    seq_len, head_dim, scale,
    stride_Qs, stride_Qd,
    stride_Ks, stride_Kd,
    stride_Ss, stride_Sd,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Compute attention scores for single head.
    Q: (seq_len, head_dim)
    K: (seq_len, head_dim)
    scores: (seq_len, seq_len)
    
    scores = (Q @ K^T) * scale
    """
    pid_m = tl.program_id(0)  # Query position
    pid_n = tl.program_id(1)  # Key position
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute Q @ K^T tile
    for k_start in range(0, head_dim, BLOCK_K):
        k_offs = k_start + offs_k
        
        # Load Q block [BLOCK_M, BLOCK_K]
        q_offs = offs_m[:, None] * stride_Qs + k_offs[None, :] * stride_Qd
        q_mask = (offs_m[:, None] < seq_len) & (k_offs[None, :] < head_dim)
        q = tl.load(Q_ptr + q_offs, mask=q_mask, other=0.0)
        
        # Load K block [BLOCK_N, BLOCK_K], then transpose for K^T
        k_offs_2d = offs_n[:, None] * stride_Ks + k_offs[None, :] * stride_Kd
        k_mask = (offs_n[:, None] < seq_len) & (k_offs[None, :] < head_dim)
        k = tl.load(K_ptr + k_offs_2d, mask=k_mask, other=0.0)  # [BLOCK_N, BLOCK_K]
        
        # Q @ K^T: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(q, tl.trans(k))
    
    # TODO: Apply scale to accumulated scores and store result
    # API hints:
    # - Multiply accumulator by scale factor
    # - tl.store(ptr + offsets, values, mask=mask) -> store with mask
    s_offs = offs_m[:, None] * stride_Ss + offs_n[None, :] * stride_Sd
    s_mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
    pass


def attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Compute scaled attention scores."""
    seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    scores = torch.empty((seq_len, seq_len), device=Q.device, dtype=Q.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = min(32, head_dim)
    
    grid = (triton.cdiv(seq_len, BLOCK_M), triton.cdiv(seq_len, BLOCK_N))
    
    attention_scores_kernel[grid](
        Q, K, scores,
        seq_len, head_dim, scale,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        scores.stride(0), scores.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return scores


# ============================================================================
# Exercise 2: Causal Masking
# ============================================================================

@triton.jit
def causal_mask_kernel(
    scores_ptr,
    seq_len,
    stride_s, stride_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Apply causal mask in-place.
    scores[i, j] = -inf if j > i (can't attend to future)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # TODO: Create causal mask, apply it to scores, and store
    # Causal mask: position i can only attend to positions j <= i
    # API hints:
    # - Create boolean mask comparing column indices to row indices
    # - tl.where(condition, true_val, false_val) -> select based on condition
    # - Use float('-inf') for masked positions
    # - tl.load(ptr, mask=mask, other=val) -> load with mask
    # - tl.store(ptr, values, mask=mask) -> store with mask
    valid = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
    s_offs = offs_m[:, None] * stride_s + offs_n[None, :] * stride_d
    pass


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """Apply causal mask in-place."""
    seq_len = scores.shape[0]
    
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(seq_len, BLOCK_M), triton.cdiv(seq_len, BLOCK_N))
    
    causal_mask_kernel[grid](
        scores, seq_len,
        scores.stride(0), scores.stride(1),
        BLOCK_M, BLOCK_N
    )
    
    return scores


# ============================================================================
# Exercise 3: Multi-Head Attention Scores
# ============================================================================

@triton.jit
def mha_scores_kernel(
    Q_ptr, K_ptr, scores_ptr,
    n_heads, seq_len, head_dim, scale,
    stride_Qh, stride_Qs, stride_Qd,
    stride_Kh, stride_Ks, stride_Kd,
    stride_Sh, stride_Ss, stride_Sd,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Multi-head attention scores.
    Q, K: (n_heads, seq_len, head_dim)
    scores: (n_heads, seq_len, seq_len)
    """
    pid_h = tl.program_id(0)  # Head
    pid_m = tl.program_id(1)  # Query position
    pid_n = tl.program_id(2)  # Key position
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Offset for this head
    Q_head = Q_ptr + pid_h * stride_Qh
    K_head = K_ptr + pid_h * stride_Kh
    S_head = scores_ptr + pid_h * stride_Sh
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, head_dim, BLOCK_K):
        k_offs = k_start + offs_k
        
        q_offs = offs_m[:, None] * stride_Qs + k_offs[None, :] * stride_Qd
        q_mask = (offs_m[:, None] < seq_len) & (k_offs[None, :] < head_dim)
        q = tl.load(Q_head + q_offs, mask=q_mask, other=0.0)
        
        k_offs_2d = offs_n[:, None] * stride_Ks + k_offs[None, :] * stride_Kd
        k_mask = (offs_n[:, None] < seq_len) & (k_offs[None, :] < head_dim)
        k = tl.load(K_head + k_offs_2d, mask=k_mask, other=0.0)
        
        acc += tl.dot(q, tl.trans(k))
    
    scores = acc * scale
    
    s_offs = offs_m[:, None] * stride_Ss + offs_n[None, :] * stride_Sd
    s_mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
    tl.store(S_head + s_offs, scores, mask=s_mask)


def mha_attention_scores(Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Multi-head attention scores."""
    n_heads, seq_len, head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)
    
    scores = torch.empty((n_heads, seq_len, seq_len), device=Q.device, dtype=Q.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = min(32, head_dim)
    
    grid = (n_heads, triton.cdiv(seq_len, BLOCK_M), triton.cdiv(seq_len, BLOCK_N))
    
    mha_scores_kernel[grid](
        Q, K, scores,
        n_heads, seq_len, head_dim, scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        scores.stride(0), scores.stride(1), scores.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    return scores


if __name__ == "__main__":
    print("Day 23: Attention Score Computation")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        seq_len, head_dim = 64, 32
        Q = torch.randn(seq_len, head_dim, device=device)
        K = torch.randn(seq_len, head_dim, device=device)
        
        print("\nTesting attention scores:")
        result = attention_scores(Q, K)
        scale = 1.0 / math.sqrt(head_dim)
        expected = (Q @ K.T) * scale
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        print("\nTesting multi-head scores:")
        n_heads = 8
        Q_mh = torch.randn(n_heads, seq_len, head_dim, device=device)
        K_mh = torch.randn(n_heads, seq_len, head_dim, device=device)
        result = mha_attention_scores(Q_mh, K_mh)
        expected = torch.bmm(Q_mh, K_mh.transpose(-2, -1)) * scale
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day23.py to verify!")
