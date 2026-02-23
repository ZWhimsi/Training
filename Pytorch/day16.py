"""
Day 16: Multi-Head Attention from Scratch
=========================================
Estimated time: 1-2 hours
Prerequisites: Day 15 (self-attention)

Learning objectives:
- Understand why multiple attention heads are beneficial
- Implement separate Q, K, V projections for multi-head attention
- Learn head splitting and concatenation
- Build complete multi-head attention module
- Compare with PyTorch's nn.MultiheadAttention

Key Concepts:
-------------
Multi-head attention allows the model to jointly attend to information
from different representation subspaces at different positions.

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
where head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)

Instead of separate projections per head, we typically:
1. Project to full dimension: Q, K, V = x @ W_q, x @ W_k, x @ W_v
2. Reshape to split heads: [batch, seq, d_model] -> [batch, heads, seq, d_k]
3. Apply attention per head in parallel
4. Concatenate and project output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Exercise 1: Head Splitting and Merging
# ============================================================================

def split_heads(x, num_heads):
    """
    Split the last dimension into (num_heads, d_k).
    
    Args:
        x: [batch, seq, d_model]
        num_heads: Number of attention heads
    
    Returns:
        [batch, num_heads, seq, d_k] where d_k = d_model // num_heads
    """
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    
    # TODO: Reshape x to [batch, seq, num_heads, d_k]
    # HINT: x = x.view(batch_size, seq_len, num_heads, d_k)
    x = None  # Replace
    
    # TODO: Transpose to [batch, num_heads, seq, d_k]
    # HINT: x = x.transpose(1, 2)
    x = None  # Replace
    
    return x


def merge_heads(x):
    """
    Inverse of split_heads.
    
    Args:
        x: [batch, num_heads, seq, d_k]
    
    Returns:
        [batch, seq, d_model] where d_model = num_heads * d_k
    """
    batch_size, num_heads, seq_len, d_k = x.shape
    
    # TODO: Transpose to [batch, seq, num_heads, d_k]
    # HINT: x = x.transpose(1, 2)
    x = None  # Replace
    
    # TODO: Reshape to [batch, seq, d_model]
    # HINT: x = x.contiguous().view(batch_size, seq_len, num_heads * d_k)
    x = None  # Replace
    
    return x


# ============================================================================
# Exercise 2: Multi-Head Scaled Dot-Product Attention
# ============================================================================

def multi_head_attention_scores(Q, K, V, mask=None):
    """
    Compute attention with multi-head tensors.
    
    This is the same as single-head attention but operates on 4D tensors.
    
    Args:
        Q: [batch, num_heads, seq_q, d_k]
        K: [batch, num_heads, seq_k, d_k]
        V: [batch, num_heads, seq_k, d_v]
        mask: Optional [batch, 1, seq_q, seq_k] or [batch, 1, 1, seq_k]
    
    Returns:
        output: [batch, num_heads, seq_q, d_v]
        attention_weights: [batch, num_heads, seq_q, seq_k]
    """
    d_k = Q.shape[-1]
    
    # TODO: Compute attention scores: Q @ K^T / sqrt(d_k)
    # HINT: scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    scores = None  # Replace
    
    # TODO: Apply mask if provided
    if mask is not None:
        # HINT: scores = scores.masked_fill(mask == 0, float('-inf'))
        pass  # Replace
    
    # TODO: Apply softmax over last dimension
    attention_weights = None  # Replace: F.softmax(scores, dim=-1)
    
    # TODO: Compute weighted sum with V
    output = None  # Replace: torch.matmul(attention_weights, V)
    
    return output, attention_weights


# ============================================================================
# Exercise 3: Complete Multi-Head Attention Module
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    
    Projects input to Q, K, V, splits into heads, applies attention,
    concatenates heads, and projects output.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.0):
        """
        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # TODO: Create linear projections for Q, K, V, and output
        # Each should be nn.Linear(d_model, d_model)
        self.W_q = None  # Replace: nn.Linear(d_model, d_model)
        self.W_k = None  # Replace: nn.Linear(d_model, d_model)
        self.W_v = None  # Replace: nn.Linear(d_model, d_model)
        self.W_o = None  # Replace: nn.Linear(d_model, d_model)
        
        # TODO: Create dropout layer
        self.dropout = None  # Replace: nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, seq_q, d_model]
            key: [batch, seq_k, d_model]
            value: [batch, seq_k, d_model]
            mask: Optional attention mask
        
        Returns:
            output: [batch, seq_q, d_model]
            attention_weights: [batch, num_heads, seq_q, seq_k]
        """
        batch_size = query.shape[0]
        
        # TODO: Step 1 - Project inputs to Q, K, V
        # HINT: Q = self.W_q(query)
        Q = None  # Replace
        K = None  # Replace
        V = None  # Replace
        
        # TODO: Step 2 - Split into multiple heads
        # HINT: Q = split_heads(Q, self.num_heads)
        Q = None  # Replace
        K = None  # Replace
        V = None  # Replace
        
        # TODO: Step 3 - Apply attention
        attn_output, attention_weights = multi_head_attention_scores(Q, K, V, mask)
        
        # TODO: Step 4 - Apply dropout to attention output
        attn_output = None  # Replace: self.dropout(attn_output)
        
        # TODO: Step 5 - Merge heads back
        attn_output = None  # Replace: merge_heads(attn_output)
        
        # TODO: Step 6 - Final output projection
        output = None  # Replace: self.W_o(attn_output)
        
        return output, attention_weights


# ============================================================================
# Exercise 4: Self-Attention with Multi-Head Attention
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Convenience wrapper for self-attention using MultiHeadAttention.
    In self-attention, Q, K, V all come from the same input.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        # TODO: Initialize the MultiHeadAttention module
        self.attention = None  # Replace: MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq, d_model]
            mask: Optional mask
        
        Returns:
            output: [batch, seq, d_model]
        """
        # TODO: Call attention with x as query, key, and value
        output, weights = None, None  # Replace: self.attention(x, x, x, mask)
        return output, weights


# ============================================================================
# Exercise 5: Cross-Attention
# ============================================================================

class CrossAttention(nn.Module):
    """
    Cross-attention where queries come from one sequence and 
    keys/values come from another.
    
    Used in encoder-decoder architectures (decoder attends to encoder output).
    """
    
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(self, query_seq, kv_seq, mask=None):
        """
        Args:
            query_seq: [batch, seq_q, d_model] - typically decoder hidden states
            kv_seq: [batch, seq_k, d_model] - typically encoder output
            mask: Optional mask for padding in kv_seq
        
        Returns:
            output: [batch, seq_q, d_model]
        """
        # TODO: Apply attention with separate query and key/value sources
        output, weights = None, None  # Replace: self.attention(query_seq, kv_seq, kv_seq, mask)
        return output, weights


# ============================================================================
# Exercise 6: Comparing with PyTorch's MultiheadAttention
# ============================================================================

def compare_with_pytorch_mha():
    """
    Compare our implementation with PyTorch's nn.MultiheadAttention.
    
    Note: PyTorch's MHA expects inputs in (seq, batch, d_model) format by default,
    unless batch_first=True is specified.
    """
    d_model = 64
    num_heads = 4
    batch_size = 2
    seq_len = 8
    
    # Create both modules
    our_mha = MultiHeadAttention(d_model, num_heads)
    
    # PyTorch's implementation (batch_first=True for same format)
    pytorch_mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    
    # Copy weights to make comparison fair
    with torch.no_grad():
        # PyTorch combines Q, K, V projections into in_proj_weight
        pytorch_mha.in_proj_weight.copy_(
            torch.cat([our_mha.W_q.weight, our_mha.W_k.weight, our_mha.W_v.weight], dim=0)
        )
        pytorch_mha.in_proj_bias.copy_(
            torch.cat([our_mha.W_q.bias, our_mha.W_k.bias, our_mha.W_v.bias], dim=0)
        )
        pytorch_mha.out_proj.weight.copy_(our_mha.W_o.weight)
        pytorch_mha.out_proj.bias.copy_(our_mha.W_o.bias)
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model)
    
    our_output, _ = our_mha(x, x, x)
    pytorch_output, _ = pytorch_mha(x, x, x)
    
    match = torch.allclose(our_output, pytorch_output, atol=1e-5)
    return match, our_output, pytorch_output


if __name__ == "__main__":
    print("Day 16: Multi-Head Attention from Scratch")
    print("=" * 50)
    
    # Demo
    print("\nDemo: Multi-head attention")
    d_model, num_heads = 64, 4
    batch, seq = 2, 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq, d_model)
    
    output, weights = mha(x, x, x)
    if output is not None:
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Attention weights shape: {weights.shape}")
        print(f"  (batch, heads, seq_q, seq_k)")
    
    print("\nRun test_day16.py to verify your implementations!")
