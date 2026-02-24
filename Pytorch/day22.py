"""
Day 22: Grouped Query Attention (GQA)
=====================================
Estimated time: 1-2 hours
Prerequisites: Day 16 (multi-head attention), Day 17 (transformer blocks)

Learning objectives:
- Understand the memory efficiency problem with multi-head attention
- Learn how Grouped Query Attention reduces KV memory
- Implement GQA with key-value head grouping
- Compare MHA, MQA, and GQA architectures
- Understand the trade-offs between memory and quality

Key Concepts:
-------------
Standard Multi-Head Attention (MHA):
    - Q, K, V each have num_heads heads
    - KV cache size: 2 * num_heads * head_dim * seq_len * batch
    - Full expressiveness but high memory cost

Multi-Query Attention (MQA):
    - Q has num_heads heads, K and V share 1 head
    - KV cache size: 2 * 1 * head_dim * seq_len * batch
    - Much smaller cache but quality degradation

Grouped Query Attention (GQA):
    - Q has num_heads heads, K and V have num_kv_heads (< num_heads)
    - Each KV head is shared by (num_heads / num_kv_heads) query heads
    - Balance between memory and quality

Example: 32 query heads, 8 KV heads
    - Each KV head serves 4 query heads
    - 4x reduction in KV cache vs MHA
    - Better quality than MQA (1 KV head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================================
# Exercise 1: Understanding Head Grouping
# ============================================================================

def compute_kv_memory_savings(num_heads: int, num_kv_heads: int) -> dict:
    """
    Compute memory savings from using GQA vs MHA.
    
    Args:
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads
    
    Returns:
        Dictionary with memory comparison metrics
    """
    # TODO: Calculate memory savings metrics
    # API hints:
    # - mha_kv_heads = num_heads (MHA has equal Q and KV heads)
    # - gqa_kv_heads = num_kv_heads
    # - savings_ratio = mha_kv_heads / gqa_kv_heads
    # - heads_per_group = num_heads // num_kv_heads
    return {
        'mha_kv_heads': 0,
        'gqa_kv_heads': 0,
        'savings_ratio': 0.0,
        'heads_per_group': 0
    }


def repeat_kv(x: torch.Tensor, num_repeats: int) -> torch.Tensor:
    """
    Repeat key/value heads to match the number of query heads.
    
    This is the core operation in GQA - we replicate KV heads so
    each query head has a corresponding KV head to attend to.
    
    Args:
        x: Tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        num_repeats: Number of times to repeat each KV head
    
    Returns:
        Tensor of shape (batch, num_kv_heads * num_repeats, seq_len, head_dim)
    
    Example:
        If x has shape (2, 4, 10, 64) and num_repeats=2:
        Output shape: (2, 8, 10, 64)
        KV head 0 is used by query heads 0, 1
        KV head 1 is used by query heads 2, 3
        etc.
    """
    # TODO: Repeat KV heads to match query heads
    # API hints:
    # - if num_repeats == 1: return x (no repetition needed)
    # - x.unsqueeze(2) -> add dimension for repetition
    # - x.expand(batch, num_kv_heads, num_repeats, seq_len, head_dim) -> expand
    # - x.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim) -> flatten
    return x


# ============================================================================
# Exercise 2: GQA Projection Layers
# ============================================================================

class GQAProjection(nn.Module):
    """
    Separate projection layers for Q, K, V with different head counts.
    """
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key-value heads (must divide num_heads evenly)
        """
        super().__init__()
        
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_kv_groups = num_heads // num_kv_heads  # How many Q heads per KV head
        
        # TODO: Create Q, K, V projection layers
        # API hints:
        # - nn.Linear(d_model, num_heads * head_dim, bias=False) -> Q projection
        # - nn.Linear(d_model, num_kv_heads * head_dim, bias=False) -> K, V projections
        self.W_q = None
        self.W_k = None
        self.W_v = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project input to Q, K, V with appropriate head counts.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_kv_heads, seq_len, head_dim)
            v: (batch, num_kv_heads, seq_len, head_dim)
        """
        # TODO: Project and reshape to multi-head format
        # API hints:
        # - self.W_q(x), self.W_k(x), self.W_v(x) -> project inputs
        # - q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2) -> reshape Q
        # - k.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2) -> reshape K
        # - v.view(batch, seq_len, num_kv_heads, head_dim).transpose(1, 2) -> reshape V
        batch, seq_len, _ = x.shape
        q = torch.zeros(batch, self.num_heads, seq_len, self.head_dim)
        k = torch.zeros(batch, self.num_kv_heads, seq_len, self.head_dim)
        v = torch.zeros(batch, self.num_kv_heads, seq_len, self.head_dim)
        return q, k, v


# ============================================================================
# Exercise 3: Grouped Query Attention
# ============================================================================

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) as used in Llama 2 70B, Mistral, etc.
    
    Key insight: We repeat KV heads to match Q heads before computing attention.
    """
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, 
                 dropout: float = 0.0):
        super().__init__()
        
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        # TODO: Create projection layers and output projection
        # API hints:
        # - GQAProjection(d_model, num_heads, num_kv_heads) -> Q, K, V projections
        # - nn.Linear(d_model, d_model, bias=False) -> output projection
        # - nn.Dropout(dropout) -> dropout layer
        self.projection = None
        self.W_o = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute grouped query attention.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # TODO: Implement GQA forward pass
        # API hints:
        # - q, k, v = self.projection(x) -> get Q, K, V projections
        # - repeat_kv(k, self.num_kv_groups) -> repeat K to match Q heads
        # - repeat_kv(v, self.num_kv_groups) -> repeat V to match Q heads
        # - torch.matmul(q, k.transpose(-2, -1)) * self.scale -> attention scores
        # - scores.masked_fill(mask == 0, float('-inf')) -> apply mask
        # - F.softmax(scores, dim=-1) -> attention weights
        # - torch.matmul(attn_weights, v) -> weighted sum of values
        # - output.transpose(1, 2).reshape(batch, seq_len, d_model) -> reshape
        # - self.W_o(output) -> output projection
        batch, seq_len, _ = x.shape
        return torch.zeros_like(x), torch.zeros(batch, self.num_heads, seq_len, seq_len)


# ============================================================================
# Exercise 4: Multi-Query Attention (MQA) - Special Case
# ============================================================================

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) - a special case of GQA with 1 KV head.
    
    Used in PaLM, Falcon, and other models for maximum KV cache efficiency.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        # TODO: Initialize MQA (GQA with num_kv_heads=1)
        # API hints:
        # - GroupedQueryAttention(d_model, num_heads, num_kv_heads=1, dropout=dropout) -> MQA
        self.attention = None
        self.num_heads = num_heads
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MQA.
        """
        # TODO: Delegate to GQA attention
        # API hints:
        # - return self.attention(x, mask)
        return x, torch.zeros(1)


# ============================================================================
# Exercise 5: Attention Comparison
# ============================================================================

class StandardMultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention for comparison.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Standard MHA has equal Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        
        q = self.W_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output, attn_weights


def compare_attention_memory(d_model: int, num_heads: int, 
                              seq_len: int, batch_size: int) -> dict:
    """
    Compare memory requirements for MHA, GQA, and MQA.
    
    Args:
        d_model: Model dimension
        num_heads: Number of query heads
        seq_len: Sequence length
        batch_size: Batch size
    
    Returns:
        Dictionary with memory comparisons
    """
    # TODO: Compute KV cache memory for each attention type
    # API hints:
    # - head_dim = d_model // num_heads
    # - bytes_per_element = 4 (for float32)
    # - MHA: 2 * batch_size * num_heads * seq_len * head_dim * bytes_per_element
    # - GQA: 2 * batch_size * (num_heads // 4) * seq_len * head_dim * bytes_per_element
    # - MQA: 2 * batch_size * 1 * seq_len * head_dim * bytes_per_element
    # - savings_ratio = mha_memory / gqa_memory
    return {
        'mha_kv_bytes': 0,
        'gqa_kv_bytes': 0,
        'mqa_kv_bytes': 0,
        'gqa_savings_vs_mha': 0.0,
        'mqa_savings_vs_mha': 0.0
    }


# ============================================================================
# Exercise 6: GQA Transformer Block
# ============================================================================

class GQATransformerBlock(nn.Module):
    """
    Transformer block using Grouped Query Attention.
    """
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 d_ff: int = None, dropout: float = 0.0):
        super().__init__()
        
        d_ff = d_ff or d_model * 4
        
        # TODO: Initialize GQA attention, FFN, and layer norms
        # API hints:
        # - GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout) -> GQA
        # - nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout)) -> FFN
        # - nn.LayerNorm(d_model) -> layer norm
        # - nn.Dropout(dropout) -> dropout layer
        self.attention = None
        self.ffn = None
        self.norm1 = None
        self.norm2 = None
        self.dropout = None
    
    def forward(self, x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with pre-norm architecture.
        """
        # TODO: Implement pre-norm forward pass
        # API hints:
        # - self.norm1(x) -> pre-norm before attention
        # - self.attention(normed, mask) -> returns (output, attn_weights)
        # - x + self.dropout(attn_out) -> residual connection
        # - self.norm2(x) -> pre-norm before FFN
        # - self.ffn(normed) -> FFN output
        return x


if __name__ == "__main__":
    print("Day 22: Grouped Query Attention (GQA)")
    print("=" * 50)
    
    # Demo configuration (similar to Llama 2 70B)
    d_model = 256
    num_heads = 8
    num_kv_heads = 2  # 4x reduction in KV cache
    batch_size = 2
    seq_len = 16
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_query_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  heads_per_kv_group: {num_heads // num_kv_heads}")
    
    # Test memory savings calculation
    savings = compute_kv_memory_savings(num_heads, num_kv_heads)
    print(f"\nMemory savings: {savings}")
    
    # Test repeat_kv
    print("\nTesting repeat_kv:")
    kv = torch.randn(batch_size, num_kv_heads, seq_len, d_model // num_heads)
    print(f"  Original KV shape: {kv.shape}")
    repeated = repeat_kv(kv, num_heads // num_kv_heads)
    print(f"  Repeated KV shape: {repeated.shape}")
    
    # Test GQA
    print("\nTesting GroupedQueryAttention:")
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    if gqa.projection is not None:
        output, attn = gqa(x)
        print(f"  Input: {x.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Attention: {attn.shape}")
    
    # Compare memory requirements
    print("\nMemory comparison:")
    mem = compare_attention_memory(d_model, num_heads, seq_len=1024, batch_size=32)
    for k, v in mem.items():
        print(f"  {k}: {v}")
    
    print("\nRun test_day22.py to verify your implementations!")
