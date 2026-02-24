"""
Day 32: Flash Attention - Forward Pass
======================================
Estimated time: 2+ hours (challenging!)
Prerequisites: Days 27-31 (attention basics, tiling)

Learning objectives:
- Understand the Flash Attention algorithm
- Implement tiled attention computation
- Use online softmax normalization
- Reduce memory from O(N^2) to O(N)

This is the capstone of the Triton track!

Flash Attention paper: https://arxiv.org/abs/2205.14135
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# CONCEPT: Flash Attention
# ============================================================================
# Standard attention materializes the N x N attention matrix, using O(N^2) memory.
# Flash Attention computes attention in tiles, using only O(N) memory.
#
# Key insight: Use "online softmax" to incrementally compute softmax
# without storing the full attention matrix.
#
# Online softmax:
# - Keep running max and sum of exponentials
# - Update output incrementally as we process each tile
# ============================================================================


# ============================================================================
# Exercise 1: Online Softmax Helper
# ============================================================================

@triton.jit
def _online_softmax_update(
    m_prev,    # Previous max
    l_prev,    # Previous sum of exp
    m_new,     # New max (from current block)
    l_new,     # New sum of exp (from current block)
):
    """
    Update online softmax statistics.
    
    Compute the updated max and sum-of-exponentials when processing a new block.
    When the max changes, previous exponential sums must be rescaled.
    
    Returns: (m_updated, l_updated, scale_prev, scale_new)
    """
    # API hints:
    # - tl.maximum(a, b) -> element-wise maximum
    # - tl.exp(x) -> element-wise exponential
    # - Online softmax update formula:
    #   m_updated = max(m_prev, m_new)
    #   scale_prev = exp(m_prev - m_updated)  # rescale old sum
    #   scale_new = exp(m_new - m_updated)    # rescale new sum
    #   l_updated = scale_prev * l_prev + scale_new * l_new
    
    # TODO: Implement online softmax update
    # 1. Compute updated max from previous and new block maxes
    # 2. Compute scaling factors for both old and new sums
    # 3. Compute updated sum using the scaling factors
    
    return None, None, None, None


# ============================================================================
# Exercise 2: Flash Attention Forward Kernel
# ============================================================================

@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, N, D,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Flash Attention forward pass.
    
    Q: [B, H, M, D] - queries
    K: [B, H, N, D] - keys  
    V: [B, H, N, D] - values
    O: [B, H, M, D] - output
    
    For each query block:
    1. Load Q block
    2. Iterate over K, V blocks
    3. Compute attention scores incrementally
    4. Update output using online softmax
    """
    # Which batch, head, and query block are we processing?
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_m = tl.program_id(2)
    
    # Pointers for this batch and head
    Q_block_ptr = Q_ptr + off_b * stride_qb + off_h * stride_qh
    K_block_ptr = K_ptr + off_b * stride_kb + off_h * stride_kh
    V_block_ptr = V_ptr + off_b * stride_vb + off_h * stride_vh
    O_block_ptr = O_ptr + off_b * stride_ob + off_h * stride_oh
    
    # Query positions for this block
    offs_m = off_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    # TODO: Load Q block [BLOCK_M, BLOCK_D]
    q_ptrs = Q_block_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Initialize online softmax state
    # m: running max, l: running sum of exp, o: running output
    m = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    o = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # TODO: Iterate over K, V blocks
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Load K block [BLOCK_N, BLOCK_D]
        k_ptrs = K_block_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # Load V block [BLOCK_N, BLOCK_D]
        v_ptrs = V_block_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        
        # TODO: Compute attention scores: S = Q @ K^T * scale
        # API hints:
        # - tl.dot(a, b) -> matrix multiply two blocks
        # - tl.trans(x) -> transpose a block
        # - Scores shape should be [BLOCK_M, BLOCK_N]
        
        s = None  # Exercise: compute scaled attention scores
        
        # TODO: Mask out invalid positions (set invalid to -inf)
        # API hints:
        # - tl.where(condition, x, y) -> select x where true, y where false
        # - Create mask: (offs_m[:, None] < M) & (offs_n[None, :] < N)
        
        pass  # Exercise: apply mask to scores
        
        # TODO: Compute block-wise max and sum for softmax
        # API hints:
        # - tl.max(x, axis=1) -> row-wise maximum
        # - tl.exp(x) -> element-wise exponential
        # - tl.sum(x, axis=1) -> row-wise sum
        # - Shifted softmax: exp(s - max(s)) for numerical stability
        
        pass  # Exercise: compute m_block, p (attention weights), l_block
        
        # TODO: Update online softmax state
        # API hints:
        # - tl.maximum(a, b) -> element-wise max
        # - tl.exp(x) -> exponential for rescaling
        # - Update formula: m_new = max(m, m_block)
        #   scale_old = exp(m - m_new), scale_new = exp(m_block - m_new)
        #   l_new = scale_old * l + scale_new * l_block
        
        pass  # Exercise: update m, l using online softmax
        
        # TODO: Update output accumulator
        # API hints:
        # - tl.dot(a, b) -> matrix multiply
        # - Accumulate: o = (scale_old * l * o + scale_new * p @ v) / l_new
        # - Use broadcasting: scale[:, None] for [BLOCK_M] -> [BLOCK_M, 1]
        
        pass  # Exercise: update output accumulator o
    
    # TODO: Store output to global memory
    # API hints:
    # - tl.store(ptr, value, mask=mask) -> store with masking
    # - Compute output pointers similar to Q loading
    # - Convert output dtype: o.to(O_ptr.dtype.element_ty)
    
    pass  # Exercise: store output block to O_ptr


def flash_attention_forward(Q, K, V):
    """
    Flash Attention forward pass.
    
    Args:
        Q: [B, H, M, D] queries
        K: [B, H, N, D] keys
        V: [B, H, N, D] values
    
    Returns:
        O: [B, H, M, D] output
    """
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    
    O = torch.empty_like(Q)
    
    scale = 1.0 / math.sqrt(D)
    
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = min(64, D)
    
    grid = (B, H, triton.cdiv(M, BLOCK_M))
    
    flash_attn_fwd_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        B, H, M, N, D,
        scale,
        BLOCK_M, BLOCK_N, BLOCK_D,
    )
    
    return O


# ============================================================================
# Exercise 3: Standard Attention (Reference)
# ============================================================================

def standard_attention(Q, K, V):
    """
    Standard attention for comparison.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    """
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Softmax
    attn = torch.softmax(scores, dim=-1)
    
    # Apply to values
    output = torch.matmul(attn, V)
    
    return output


if __name__ == "__main__":
    print("Day 32: Flash Attention Forward")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print("\nThis is a challenging exercise!")
        print("Flash Attention reduces memory from O(N^2) to O(N)")
        print("\nRun test_day32.py to verify your implementation.")
    else:
        print("CUDA not available")
