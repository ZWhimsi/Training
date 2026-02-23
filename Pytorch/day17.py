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
        # HINT: self.gamma = nn.Parameter(torch.ones(d_model))
        self.gamma = None  # Replace
        self.beta = None   # Replace: nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            normalized: [batch, seq, d_model]
        """
        # TODO: Compute mean over last dimension (keepdim=True)
        mean = None  # Replace: x.mean(dim=-1, keepdim=True)
        
        # TODO: Compute variance over last dimension (keepdim=True)
        # HINT: var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        var = None  # Replace
        
        # TODO: Normalize: (x - mean) / sqrt(var + eps)
        x_norm = None  # Replace
        
        # TODO: Apply scale and shift: gamma * x_norm + beta
        output = None  # Replace
        
        return output


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
        
        # TODO: First linear layer (expansion)
        self.linear1 = None  # Replace: nn.Linear(d_model, d_ff)
        
        # TODO: Second linear layer (projection back)
        self.linear2 = None  # Replace: nn.Linear(d_ff, d_model)
        
        # TODO: Dropout layer
        self.dropout = None  # Replace: nn.Dropout(dropout)
        
        # Store activation function
        self.activation = activation
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            [batch, seq, d_model]
        """
        # TODO: Apply first linear, then activation, then dropout
        # HINT: x = self.dropout(F.gelu(self.linear1(x)))
        x = None  # Replace
        
        # TODO: Apply second linear
        x = None  # Replace: self.linear2(x)
        
        return x


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
        
        # Import MultiHeadAttention from day16
        from day16 import MultiHeadAttention
        
        # TODO: Self-attention sublayer
        self.self_attn = None  # Replace: MultiHeadAttention(d_model, num_heads, dropout)
        
        # TODO: Feed-forward sublayer
        self.ffn = None  # Replace: FeedForward(d_model, d_ff, dropout)
        
        # TODO: Layer norms (one for each sublayer)
        self.norm1 = None  # Replace: LayerNorm(d_model)
        self.norm2 = None  # Replace: LayerNorm(d_model)
        
        # TODO: Dropout for residual connections
        self.dropout = None  # Replace: nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq, d_model]
            mask: Optional attention mask
        
        Returns:
            [batch, seq, d_model]
        """
        # TODO: Self-attention with residual connection
        # Step 1: Apply self-attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        
        # Step 2: Apply dropout
        attn_output = None  # Replace: self.dropout(attn_output)
        
        # Step 3: Add residual and normalize (POST-NORM: norm AFTER adding)
        x = None  # Replace: self.norm1(x + attn_output)
        
        # TODO: Feed-forward with residual connection
        # Step 1: Apply FFN
        ffn_output = None  # Replace: self.ffn(x)
        
        # Step 2: Apply dropout
        ffn_output = None  # Replace: self.dropout(ffn_output)
        
        # Step 3: Add residual and normalize
        x = None  # Replace: self.norm2(x + ffn_output)
        
        return x


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
        
        # TODO: Initialize same components as post-norm
        self.self_attn = None  # Replace: MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = None        # Replace: FeedForward(d_model, d_ff, dropout)
        self.norm1 = None      # Replace: LayerNorm(d_model)
        self.norm2 = None      # Replace: LayerNorm(d_model)
        self.dropout = None    # Replace: nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq, d_model]
            mask: Optional attention mask
        
        Returns:
            [batch, seq, d_model]
        """
        # TODO: Self-attention with PRE-NORM
        # Step 1: Normalize FIRST
        normed = None  # Replace: self.norm1(x)
        
        # Step 2: Apply self-attention
        attn_output, _ = self.self_attn(normed, normed, normed, mask)
        
        # Step 3: Apply dropout and add residual (no norm here!)
        x = None  # Replace: x + self.dropout(attn_output)
        
        # TODO: Feed-forward with PRE-NORM
        # Step 1: Normalize FIRST  
        normed = None  # Replace: self.norm2(x)
        
        # Step 2: Apply FFN
        ffn_output = None  # Replace: self.ffn(normed)
        
        # Step 3: Apply dropout and add residual
        x = None  # Replace: x + self.dropout(ffn_output)
        
        return x


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
        
        # TODO: Create stack of encoder blocks
        # HINT: Use nn.ModuleList and a for loop
        if pre_norm:
            self.layers = None  # Replace with ModuleList of PreNormEncoderBlock
        else:
            self.layers = None  # Replace with ModuleList of PostNormEncoderBlock
        
        # TODO: Final layer norm (only needed for pre-norm architecture)
        # In pre-norm, the last block's output isn't normalized
        self.final_norm = None  # Replace: LayerNorm(d_model) if pre_norm else None
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq, d_model]
            mask: Optional attention mask
        
        Returns:
            [batch, seq, d_model]
        """
        # TODO: Pass through all layers
        if self.layers is not None:
            for layer in self.layers:
                x = layer(x, mask)
        
        # TODO: Apply final norm for pre-norm architecture
        if self.pre_norm and self.final_norm is not None:
            x = self.final_norm(x)
        
        return x


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
        
        # TODO: Token embedding
        self.token_emb = None  # Replace: nn.Embedding(vocab_size, d_model)
        
        # TODO: Positional encoding
        self.pos_enc = None  # Replace: PositionalEncoding(d_model, max_len, dropout)
        
        # TODO: Transformer encoder
        self.encoder = None  # Replace: TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout, pre_norm)
        
        self.d_model = d_model
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Token indices [batch, seq]
            mask: Optional attention mask
        
        Returns:
            [batch, seq, d_model]
        """
        # TODO: Get token embeddings and scale
        # HINT: Scaling by sqrt(d_model) as in original paper
        x = None  # Replace: self.token_emb(x) * math.sqrt(self.d_model)
        
        # TODO: Add positional encoding
        x = None  # Replace: self.pos_enc(x)
        
        # TODO: Pass through encoder
        x = None  # Replace: self.encoder(x, mask)
        
        return x


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
