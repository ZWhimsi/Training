"""
Day 25: Advanced Attention Patterns
===================================
Estimated time: 1-2 hours
Prerequisites: Day 16 (multi-head attention), Day 22 (GQA)

Learning objectives:
- Understand the quadratic memory problem with long sequences
- Implement sliding window attention for local context
- Build dilated/strided attention patterns
- Implement block-sparse attention
- Understand and implement attention with sinks (StreamingLLM)
- Combine patterns for efficient long-context attention

Key Concepts:
-------------
Standard Attention Complexity:
    Memory: O(n²) for attention matrix
    Compute: O(n²) for each attention computation
    Problem: 32K tokens = 1B attention entries per head!

Sliding Window Attention:
    Each token attends only to w previous tokens
    Memory: O(n * w) instead of O(n²)
    Used in: Mistral, Longformer (local attention)

Dilated/Strided Attention:
    Attend to every k-th token within a larger window
    Captures longer-range dependencies with same compute

Block Sparse Attention:
    Divide sequence into blocks, attend within/between blocks
    More flexible patterns with block-level granularity

Attention Sinks (StreamingLLM):
    First few tokens act as "attention sinks"
    Keep sink tokens + recent window for infinite context
    Critical insight: removing sink tokens hurts quality

Combining Patterns:
    Many models combine local + global patterns
    Example: Local window + global tokens (Longformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


# ============================================================================
# Exercise 1: Sliding Window Attention Mask
# ============================================================================

def create_sliding_window_mask(seq_len: int, window_size: int, 
                                causal: bool = True) -> torch.Tensor:
    """
    Create attention mask for sliding window attention.
    
    Each position attends to at most window_size tokens before it
    (and optionally after, if not causal).
    
    Args:
        seq_len: Sequence length
        window_size: Number of positions to attend to on each side
        causal: If True, only attend to past positions
    
    Returns:
        Mask tensor of shape (seq_len, seq_len)
        1 = attend, 0 = mask out
    
    Example (seq_len=6, window_size=2, causal=True):
        [1, 0, 0, 0, 0, 0]  # Position 0 only sees itself
        [1, 1, 0, 0, 0, 0]  # Position 1 sees 0-1
        [1, 1, 1, 0, 0, 0]  # Position 2 sees 0-2
        [0, 1, 1, 1, 0, 0]  # Position 3 sees 1-3 (window of 2)
        [0, 0, 1, 1, 1, 0]  # Position 4 sees 2-4
        [0, 0, 0, 1, 1, 1]  # Position 5 sees 3-5
    """
    # API hints:
    # - torch.arange(n) -> 1D tensor [0, 1, ..., n-1]
    # - tensor.unsqueeze(dim) -> add dimension at position dim
    # - torch.abs(tensor) -> element-wise absolute value
    # - (condition1) & (condition2) -> element-wise logical AND
    # - tensor.float() -> convert to float tensor
    
    return None


# ============================================================================
# Exercise 2: Sliding Window Attention
# ============================================================================

class SlidingWindowAttention(nn.Module):
    """
    Multi-head attention with sliding window pattern.
    
    More memory efficient for long sequences since each position
    only attends to a local window of tokens.
    """
    
    def __init__(self, d_model: int, num_heads: int, window_size: int,
                 dropout: float = 0.0):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        # TODO: Initialize projection layers
        # API hints:
        # - nn.Linear(in_features, out_features) -> linear transformation
        # - nn.Dropout(p) -> dropout layer with probability p
        
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor, 
                causal: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sliding window attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            causal: Whether to use causal masking
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # API hints:
        # - tensor.view(shape) -> reshape tensor
        # - tensor.transpose(dim0, dim1) -> swap dimensions
        # - torch.matmul(a, b) -> matrix multiplication
        # - tensor.masked_fill(mask, value) -> fill where mask is True
        # - F.softmax(tensor, dim=-1) -> softmax over last dimension
        # - create_sliding_window_mask(seq_len, window_size, causal) -> get mask
        
        return None


# ============================================================================
# Exercise 3: Dilated Attention Mask
# ============================================================================

def create_dilated_mask(seq_len: int, window_size: int, 
                        dilation: int, causal: bool = True) -> torch.Tensor:
    """
    Create attention mask for dilated/strided attention.
    
    Instead of attending to consecutive tokens, attend to every dilation-th token.
    This increases the effective receptive field without increasing compute.
    
    Args:
        seq_len: Sequence length
        window_size: Number of tokens to attend to
        dilation: Stride between attended tokens
        causal: If True, only attend to past positions
    
    Returns:
        Mask tensor (seq_len, seq_len)
    
    Example (seq_len=10, window_size=3, dilation=2, causal=True):
        Position 6 would attend to positions 6, 4, 2 (every 2nd position, 3 total)
    """
    # API hints:
    # - torch.arange(n).unsqueeze(dim) -> create position indices
    # - (distance % dilation) == 0 -> check if distance is multiple of dilation
    # - (distance // dilation) < window_size -> check if within window
    # - condition1 & condition2 & condition3 -> combine boolean conditions
    # - tensor.float() -> convert boolean mask to float
    
    return None


# ============================================================================
# Exercise 4: Block Sparse Attention
# ============================================================================

def create_block_sparse_mask(seq_len: int, block_size: int,
                              num_global_blocks: int = 1,
                              num_random_blocks: int = 1) -> torch.Tensor:
    """
    Create block-sparse attention mask.
    
    Attention is computed at block granularity:
    - Each block attends to itself (local)
    - First N blocks are global (attend to/from all)
    - Random blocks for longer-range connections
    
    Args:
        seq_len: Sequence length
        block_size: Size of each block
        num_global_blocks: Number of blocks that attend globally
        num_random_blocks: Number of random blocks to attend to
    
    Returns:
        Mask tensor (seq_len, seq_len)
    """
    # API hints:
    # - torch.zeros(seq_len, seq_len) -> initialize empty mask
    # - num_blocks = (seq_len + block_size - 1) // block_size -> ceiling division
    # - mask[start:end, start:end] = 1 -> set block region to 1
    # - min((i + 1) * block_size, seq_len) -> handle last block boundary
    
    return None


class BlockSparseAttention(nn.Module):
    """
    Block-sparse attention for efficient long-range attention.
    """
    
    def __init__(self, d_model: int, num_heads: int, block_size: int = 64,
                 num_global_blocks: int = 1, dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.num_global_blocks = num_global_blocks
        self.scale = self.head_dim ** -0.5
        
        # TODO: Initialize layers
        # API hints:
        # - nn.Linear(in_features, out_features) -> linear projection
        # - nn.Dropout(p) -> dropout layer
        
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implement block sparse attention (similar to sliding window but with block sparse mask).
        """
        # API hints:
        # - create_block_sparse_mask(seq_len, block_size, num_global_blocks) -> get mask
        # - Same attention pattern as SlidingWindowAttention
        
        return None


# ============================================================================
# Exercise 5: Attention Sinks (StreamingLLM)
# ============================================================================

class AttentionWithSinks(nn.Module):
    """
    Attention with attention sinks for streaming/infinite context.
    
    Key insight from StreamingLLM paper:
    - First few tokens act as "attention sinks" that absorb attention mass
    - Removing them causes quality degradation
    - Keep: [sink tokens] + [recent window] for stable generation
    
    This enables processing arbitrarily long sequences with fixed memory.
    """
    
    def __init__(self, d_model: int, num_heads: int, 
                 num_sink_tokens: int = 4, window_size: int = 512,
                 dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_sink_tokens = num_sink_tokens
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        # TODO: Initialize projection layers (same as standard attention)
        # API hints:
        # - nn.Linear(d_model, d_model) -> Q, K, V, O projections
        # - nn.Dropout(dropout) -> dropout layer
        
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None
        self.dropout = None
    
    def create_sink_window_mask(self, seq_len: int, 
                                 current_pos: int = None) -> torch.Tensor:
        """
        Create mask that attends to sinks + recent window.
        
        Args:
            seq_len: Current sequence length
            current_pos: Current position (for generation)
        
        Returns:
            Mask (seq_len, seq_len) or (1, attended_len) for generation
        """
        # API hints:
        # - torch.zeros(seq_len, seq_len) -> initialize mask
        # - mask[i, :num_sink_tokens] = 1 -> attend to sink tokens
        # - max(num_sink_tokens, i - window_size + 1) -> window start
        # - mask[i, start:i+1] = 1 -> attend to recent window
        
        return None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sink attention (standard attention with custom mask).
        """
        # API hints:
        # - self.create_sink_window_mask(seq_len) -> get attention mask
        # - Standard attention computation with mask
        
        return None


# ============================================================================
# Exercise 6: Combined Local + Global Attention (Longformer-style)
# ============================================================================

class LocalGlobalAttention(nn.Module):
    """
    Combines local sliding window attention with global attention tokens.
    
    Similar to Longformer:
    - Most tokens use local sliding window attention
    - Special "global" tokens attend to/from all positions
    - Useful for tasks needing both local context and global aggregation
    """
    
    def __init__(self, d_model: int, num_heads: int, 
                 local_window: int = 256, dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.local_window = local_window
        self.scale = self.head_dim ** -0.5
        
        # TODO: Initialize layers
        # API hints:
        # - nn.Linear(d_model, d_model) -> Q, K, V, O projections
        # - nn.Dropout(dropout) -> dropout layer
        
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None
        self.dropout = None
    
    def create_local_global_mask(self, seq_len: int, 
                                  global_indices: torch.Tensor) -> torch.Tensor:
        """
        Create mask combining local window and global tokens.
        
        Args:
            seq_len: Sequence length
            global_indices: Indices of global tokens (e.g., [0] for CLS token)
        
        Returns:
            Attention mask (seq_len, seq_len)
        """
        # API hints:
        # - create_sliding_window_mask(seq_len, window, causal=False) -> local mask
        # - mask[idx, :] = 1 -> global token attends to all
        # - mask[:, idx] = 1 -> all attend to global token
        
        return None
    
    def forward(self, x: torch.Tensor, 
                global_indices: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with local + global attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            global_indices: Indices of global tokens (default: first token)
        """
        # API hints:
        # - self.create_local_global_mask(seq_len, global_indices) -> get mask
        # - Standard attention with combined mask
        
        return None


# ============================================================================
# Exercise 7: Memory Efficiency Comparison
# ============================================================================

def compute_attention_memory(seq_len: int, num_heads: int, head_dim: int,
                              batch_size: int = 1, dtype_bytes: int = 2) -> dict:
    """
    Compare memory requirements for different attention patterns.
    
    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads
        head_dim: Dimension per head
        batch_size: Batch size
        dtype_bytes: Bytes per element
    
    Returns:
        Dictionary comparing memory usage
    """
    # API hints:
    # - standard_attn_matrix = batch * heads * seq * seq * dtype_bytes
    # - sliding_attn_matrix = batch * heads * seq * min(window, seq) * dtype_bytes
    # - block_sparse uses sparse_ratio (~0.1) of standard
    # - Compute ratios: standard / sliding, standard / block_sparse
    
    return None


def analyze_attention_patterns(attention_weights: torch.Tensor) -> dict:
    """
    Analyze attention weight distribution.
    
    Args:
        attention_weights: (batch, heads, seq, seq)
    
    Returns:
        Statistics about attention patterns
    """
    # Average attention entropy (higher = more uniform attention)
    entropy = -torch.sum(
        attention_weights * torch.log(attention_weights + 1e-9), 
        dim=-1
    ).mean()
    
    # Attention to first tokens (sink behavior)
    sink_attention = attention_weights[:, :, :, :4].sum(dim=-1).mean()
    
    # Local vs global attention ratio
    batch, heads, seq_len, _ = attention_weights.shape
    local_window = 64
    
    # Create local mask
    rows = torch.arange(seq_len).unsqueeze(1)
    cols = torch.arange(seq_len).unsqueeze(0)
    local_mask = (torch.abs(rows - cols) <= local_window).float()
    
    local_attention = (attention_weights * local_mask.to(attention_weights.device)).sum(dim=-1).mean()
    
    return {
        'entropy': entropy.item(),
        'sink_attention': sink_attention.item(),
        'local_attention_ratio': local_attention.item()
    }


if __name__ == "__main__":
    print("Day 25: Advanced Attention Patterns")
    print("=" * 50)
    
    # Configuration
    d_model = 256
    num_heads = 8
    batch_size = 2
    seq_len = 32
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  seq_len: {seq_len}")
    
    # Test sliding window mask
    print("\nSliding Window Mask (window=4):")
    mask = create_sliding_window_mask(8, window_size=4, causal=True)
    print(mask.int())
    
    # Test dilated mask
    print("\nDilated Mask (window=3, dilation=2):")
    dilated = create_dilated_mask(10, window_size=3, dilation=2, causal=True)
    print(dilated[:6, :6].int())
    
    # Test sliding window attention
    print("\nTesting SlidingWindowAttention:")
    sw_attn = SlidingWindowAttention(d_model, num_heads, window_size=8)
    if sw_attn.W_q is not None:
        x = torch.randn(batch_size, seq_len, d_model)
        output, attn = sw_attn(x)
        print(f"  Output: {output.shape}")
        print(f"  Attention sparsity: {(attn == 0).float().mean():.2%}")
    
    # Test attention with sinks
    print("\nTesting AttentionWithSinks:")
    sink_attn = AttentionWithSinks(d_model, num_heads, num_sink_tokens=4, window_size=16)
    if sink_attn.W_q is not None:
        x = torch.randn(batch_size, seq_len, d_model)
        output, attn = sink_attn(x)
        print(f"  Output: {output.shape}")
    
    # Memory comparison
    print("\nMemory Comparison (seq_len=8192):")
    mem = compute_attention_memory(8192, num_heads, d_model // num_heads)
    for k, v in mem.items():
        if 'bytes' in k:
            print(f"  {k}: {v / (1024**2):.1f} MB")
        else:
            print(f"  {k}: {v:.1f}x")
    
    print("\nRun test_day25.py to verify your implementations!")
