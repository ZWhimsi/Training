"""
Day 17: Transformer Encoder Block
=================================
Estimated time: 1-2 hours
Prerequisites: Day 16 (multi-head attention)

Learning objectives:
- Understand the Transformer encoder architecture
- Implement residual connections correctly
- Implement layer normalization
- Compare pre-norm vs post-norm architectures
- Build a complete encoder block and stack

Key Concepts:
-------------
The Transformer encoder block consists of:
1. Multi-Head Self-Attention
2. Feed-Forward Network (FFN)
3. Residual connections (add input to output)
4. Layer Normalization

Two common normalization patterns:

Post-Norm (original "Attention is All You Need"):
    output = LayerNorm(x + Sublayer(x))

Pre-Norm (used in GPT-2, more stable training):
    output = x + Sublayer(LayerNorm(x))

The FFN typically expands dimension by 4x:
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    Where W1: [d_model, d_ff], W2: [d_ff, d_model]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Exercise 1: Layer Normalization
# ============================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization normalizes over the last dimension (features).
    
    For input [batch, seq, d_model], normalizes over d_model.
    
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        
        # TODO: Create learnable scale (gamma) and shift (beta) parameters
        # Both should be of shape [d_model] and initialized to ones/zeros
        # API hints:
        # - nn.Parameter(tensor) -> creates learnable parameter
        # - torch.ones(size) -> tensor of ones
        # - torch.zeros(size) -> tensor of zeros
        self.gamma = None
        self.beta = None
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            normalized: [batch, seq, d_model]
        """
        # TODO: Compute mean and variance over last dimension, normalize, apply scale and shift
        # API hints:
        # - tensor.mean(dim=-1, keepdim=True) -> mean over last dim
        # - torch.sqrt(tensor) -> element-wise square root
        # - Broadcasting: gamma * x_norm + beta works with [d_model] params
        return None


# ============================================================================
# Exercise 2: Position-wise Feed-Forward Network
# ============================================================================

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = activation(x @ W1 + b1) @ W2 + b2
    
    Typically d_ff = 4 * d_model
    """
    
    def __init__(self, d_model, d_ff=None, dropout=0.0, activation='gelu'):
        super().__init__()
        
        d_ff = d_ff or d_model * 4
        
        # TODO: Create two linear layers and dropout
        # API hints:
        # - nn.Linear(in_features, out_features) -> linear layer
        # - nn.Dropout(p) -> dropout layer
        self.linear1 = None
        self.linear2 = None
        self.dropout = None
        self.activation = activation
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            [batch, seq, d_model]
        """
        # TODO: Apply linear1 -> GELU -> dropout -> linear2
        # API hints:
        # - F.gelu(tensor) -> GELU activation
        # - self.dropout(tensor) -> apply dropout
        # - self.linear1(x), self.linear2(x) -> apply linear layers
        return None


# ============================================================================
# Exercise 3: Post-Norm Encoder Block (Original Transformer)
# ============================================================================

class PostNormEncoderBlock(nn.Module):
    """
    Post-Norm Encoder Block (as in "Attention is All You Need").
    
    Structure:
        x = LayerNorm(x + MultiHeadAttention(x))
        x = LayerNorm(x + FeedForward(x))
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0):
        super().__init__()
        
        from day16 import MultiHeadAttention
        
        # TODO: Create self-attention, FFN, layer norms, and dropout
        # API hints:
        # - MultiHeadAttention(d_model, num_heads, dropout) -> from day16
        # - FeedForward(d_model, d_ff, dropout) -> defined above
        # - LayerNorm(d_model) -> defined above
        # - nn.Dropout(dropout) -> dropout layer
        self.self_attn = None
        self.ffn = None
        self.norm1 = None
        self.norm2 = None
        self.dropout = None
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq, d_model]
            mask: Optional attention mask
        
        Returns:
            [batch, seq, d_model]
        """
        # TODO: Implement post-norm encoder block
        # Post-norm pattern: output = LayerNorm(x + Sublayer(x))
        # API hints:
        # - self.self_attn(q, k, v, mask) -> returns (output, attn_weights)
        # - self.ffn(x) -> feed-forward output
        # - self.norm1(tensor), self.norm2(tensor) -> layer norm
        # - self.dropout(tensor) -> apply dropout
        return None


# ============================================================================
# Exercise 4: Pre-Norm Encoder Block (GPT-2 style)
# ============================================================================

class PreNormEncoderBlock(nn.Module):
    """
    Pre-Norm Encoder Block (as in GPT-2).
    
    Structure:
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FeedForward(LayerNorm(x))
    
    Benefits:
    - More stable gradient flow
    - Can train deeper networks without warmup
    - Generally preferred in modern architectures
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0):
        super().__init__()
        
        from day16 import MultiHeadAttention
        
        # TODO: Create self-attention, FFN, layer norms, and dropout
        # API hints:
        # - MultiHeadAttention(d_model, num_heads, dropout) -> from day16
        # - FeedForward(d_model, d_ff, dropout) -> defined above
        # - LayerNorm(d_model) -> defined above
        # - nn.Dropout(dropout) -> dropout layer
        self.self_attn = None
        self.ffn = None
        self.norm1 = None
        self.norm2 = None
        self.dropout = None
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq, d_model]
            mask: Optional attention mask
        
        Returns:
            [batch, seq, d_model]
        """
        # TODO: Implement pre-norm encoder block
        # Pre-norm pattern: output = x + Sublayer(LayerNorm(x))
        # API hints:
        # - self.norm1(x), self.norm2(x) -> normalize BEFORE sublayer
        # - self.self_attn(q, k, v, mask) -> returns (output, attn_weights)
        # - self.ffn(x) -> feed-forward output
        # - self.dropout(tensor) -> apply dropout
        return None


# ============================================================================
# Exercise 5: Transformer Encoder Stack
# ============================================================================

class TransformerEncoder(nn.Module):
    """
    Stack of N encoder blocks.
    
    For pre-norm, we need a final layer norm after all blocks.
    """
    
    def __init__(self, d_model, num_heads, num_layers, d_ff=None, 
                 dropout=0.0, pre_norm=True):
        super().__init__()
        
        self.pre_norm = pre_norm
        
        # TODO: Create stack of encoder blocks and final norm
        # API hints:
        # - nn.ModuleList([...]) -> list of modules for iteration
        # - PreNormEncoderBlock(d_model, num_heads, d_ff, dropout) -> for pre_norm=True
        # - PostNormEncoderBlock(d_model, num_heads, d_ff, dropout) -> for pre_norm=False
        # - LayerNorm(d_model) -> final norm (only for pre-norm)
        self.layers = None
        self.final_norm = None
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq, d_model]
            mask: Optional attention mask
        
        Returns:
            [batch, seq, d_model]
        """
        # TODO: Pass through all layers, apply final norm if pre-norm
        # API hints:
        # - for layer in self.layers: x = layer(x, mask)
        # - self.final_norm(x) -> apply final normalization
        return None


# ============================================================================
# Exercise 6: Complete Encoder with Embeddings
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (from original Transformer).
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but should be saved)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x):
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoderWithEmbedding(nn.Module):
    """
    Complete Transformer encoder with token and positional embeddings.
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, 
                 d_ff=None, dropout=0.0, max_len=5000, pre_norm=True):
        super().__init__()
        
        # TODO: Create token embedding, positional encoding, and encoder
        # API hints:
        # - nn.Embedding(vocab_size, d_model) -> token embedding
        # - PositionalEncoding(d_model, max_len, dropout) -> defined above
        # - TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout, pre_norm)
        self.token_emb = None
        self.pos_enc = None
        self.encoder = None
        self.d_model = d_model
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Token indices [batch, seq]
            mask: Optional attention mask
        
        Returns:
            [batch, seq, d_model]
        """
        # TODO: Embed tokens, scale by sqrt(d_model), add positional encoding, encode
        # API hints:
        # - self.token_emb(x) -> token embeddings
        # - math.sqrt(self.d_model) -> scaling factor
        # - self.pos_enc(x) -> add positional encoding
        # - self.encoder(x, mask) -> pass through encoder stack
        return None


# ============================================================================
# Comparing Pre-Norm vs Post-Norm
# ============================================================================

def compare_norm_architectures():
    """
    Demonstrate the difference in gradient flow between architectures.
    """
    d_model, num_heads, num_layers = 64, 4, 6
    batch, seq = 2, 16
    
    x = torch.randn(batch, seq, d_model, requires_grad=True)
    
    # Build both architectures
    pre_norm = TransformerEncoder(d_model, num_heads, num_layers, pre_norm=True)
    post_norm = TransformerEncoder(d_model, num_heads, num_layers, pre_norm=False)
    
    # Forward pass
    pre_out = pre_norm(x)
    post_out = post_norm(x.clone().detach().requires_grad_(True))
    
    if pre_out is not None and post_out is not None:
        # Compute gradients
        pre_out.sum().backward()
        post_out.sum().backward()
        
        return pre_out, post_out
    
    return None, None


if __name__ == "__main__":
    print("Day 17: Transformer Encoder Block")
    print("=" * 50)
    
    # Demo
    d_model, num_heads = 64, 4
    batch, seq = 2, 16
    
    print("\nDemo: Single encoder block (pre-norm)")
    block = PreNormEncoderBlock(d_model, num_heads)
    x = torch.randn(batch, seq, d_model)
    
    output = block(x)
    if output is not None:
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
    
    print("\nDemo: Full encoder stack")
    encoder = TransformerEncoder(d_model, num_heads, num_layers=4)
    output = encoder(x)
    if output is not None:
        print(f"4-layer encoder output: {output.shape}")
    
    print("\nRun test_day17.py to verify your implementations!")
