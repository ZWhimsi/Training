"""
Day 15: Self-Attention from Scratch
===================================
Estimated time: 1-2 hours
Prerequisites: Day 14 (custom modules)

Learning objectives:
- Understand Query, Key, Value concept
- Implement scaled dot-product attention
- Build self-attention from scratch
- Compare with PyTorch's implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Exercise 1: Scaled Dot-Product Attention
# ============================================================================

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Args:
        Q: Queries [batch, seq_q, d_k]
        K: Keys [batch, seq_k, d_k]
        V: Values [batch, seq_k, d_v]
        mask: Optional mask [batch, seq_q, seq_k]
    
    Returns:
        output: [batch, seq_q, d_v]
        attention_weights: [batch, seq_q, seq_k]
    """
    # API hints:
    # - Q.shape[-1] -> d_k dimension
    # - Q @ K.transpose(-2, -1) -> attention scores (batch, seq_q, seq_k)
    # - scores / math.sqrt(d_k) -> scale by sqrt(d_k)
    # - scores.masked_fill(mask == 0, float('-inf')) -> mask invalid positions
    # - F.softmax(scores, dim=-1) -> normalize to attention weights
    # - attention_weights @ V -> weighted sum of values
    return None


# ============================================================================
# Exercise 2: Self-Attention Module
# ============================================================================

class SelfAttention(nn.Module):
    """
    Self-attention module where Q, K, V come from the same input.
    """
    
    def __init__(self, d_model, d_k=None, d_v=None):
        """
        Args:
            d_model: Input/output dimension
            d_k: Key dimension (default: d_model)
            d_v: Value dimension (default: d_model)
        """
        super().__init__()
        
        d_k = d_k or d_model
        d_v = d_v or d_model
        
        # API hints:
        # - nn.Linear(d_model, d_k) -> query projection
        # - nn.Linear(d_model, d_k) -> key projection
        # - nn.Linear(d_model, d_v) -> value projection
        # - nn.Linear(d_v, d_model) -> output projection
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None
        
        self.d_k = d_k
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input [batch, seq, d_model]
            mask: Optional mask
        
        Returns:
            output: [batch, seq, d_model]
        """
        # API hints:
        # - self.W_q(x), self.W_k(x), self.W_v(x) -> project to Q, K, V
        # - scaled_dot_product_attention(Q, K, V, mask) -> compute attention
        # - self.W_o(output) -> project output back to d_model
        return None


# ============================================================================
# Exercise 3: Causal (Masked) Attention
# ============================================================================

def create_causal_mask(seq_len, device='cpu'):
    """
    Create a causal mask to prevent attending to future positions.
    
    Returns:
        mask: [1, seq_len, seq_len] where mask[i,j]=1 if j<=i else 0
    """
    # API hints:
    # - torch.ones(seq_len, seq_len) -> all ones matrix
    # - torch.tril(matrix) -> lower triangular (zeros above diagonal)
    # - mask.unsqueeze(0) -> add batch dimension
    # - mask.to(device) -> move to specified device
    return None


def causal_attention(Q, K, V):
    """
    Attention with causal masking (for autoregressive models).
    """
    seq_len = Q.shape[1]
    mask = create_causal_mask(seq_len, Q.device)
    return scaled_dot_product_attention(Q, K, V, mask)


# ============================================================================
# Exercise 4: Attention Visualization
# ============================================================================

def visualize_attention(attention_weights, tokens=None):
    """
    Create a simple text visualization of attention weights.
    
    Args:
        attention_weights: [seq_q, seq_k]
        tokens: Optional list of token strings
    """
    seq_q, seq_k = attention_weights.shape
    
    if tokens is None:
        tokens = [f"t{i}" for i in range(seq_k)]
    
    print("\nAttention Weights:")
    print("-" * (seq_k * 8 + 6))
    
    # Header
    print("     ", end="")
    for t in tokens[:seq_k]:
        print(f"{t:>6}", end=" ")
    print()
    
    # Weights
    for i in range(seq_q):
        print(f"{tokens[i] if i < len(tokens) else f't{i}':>4} ", end="")
        for j in range(seq_k):
            w = attention_weights[i, j].item()
            print(f"{w:>6.3f}", end=" ")
        print()


# ============================================================================
# Exercise 5: Compare with PyTorch
# ============================================================================

def compare_with_pytorch():
    """
    Compare our implementation with PyTorch's F.scaled_dot_product_attention.
    """
    batch, seq, d = 2, 8, 16
    
    Q = torch.randn(batch, seq, d)
    K = torch.randn(batch, seq, d)
    V = torch.randn(batch, seq, d)
    
    # Our implementation
    our_output, _ = scaled_dot_product_attention(Q, K, V)
    
    # PyTorch's implementation (if available)
    if hasattr(F, 'scaled_dot_product_attention'):
        pytorch_output = F.scaled_dot_product_attention(Q, K, V)
        
        match = torch.allclose(our_output, pytorch_output, atol=1e-5)
        return match, our_output, pytorch_output
    
    return None, our_output, None


if __name__ == "__main__":
    print("Day 15: Self-Attention from Scratch")
    print("=" * 50)
    
    # Demo
    print("\nDemo: Attention on small sequence")
    Q = torch.randn(1, 4, 8)
    K = torch.randn(1, 4, 8)
    V = torch.randn(1, 4, 8)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    print(f"Input shape: {Q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    
    print("\nRun test_day15.py to verify your implementations!")
