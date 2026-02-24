"""
Day 14: Positional Encoding
===========================
Estimated time: 1-2 hours
Prerequisites: Day 13 (embedding layers)

Learning objectives:
- Understand why position information is needed
- Implement sinusoidal positional encoding
- Implement learned positional encoding
- Understand Rotary Position Embeddings (RoPE) basics
- Apply positional encoding to transformer-style models
"""

import torch
import torch.nn as nn
import math
from typing import Optional


# ============================================================================
# CONCEPT: Positional Encoding
# ============================================================================
"""
Transformers process all positions in parallel, losing sequential information.
Positional encodings add position information to token embeddings.

Three main approaches:
1. Sinusoidal (fixed): PE(pos, 2i) = sin(pos / 10000^(2i/d))
                       PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
   - No learned parameters
   - Can generalize to longer sequences

2. Learned: Simple embedding lookup for positions
   - More flexible
   - Limited to trained sequence length

3. RoPE (Rotary Position Embedding): Applies rotation based on position
   - Encodes relative position in attention
   - Better for long sequences
"""


# ============================================================================
# Exercise 1: Sinusoidal Positional Encoding
# ============================================================================

def create_sinusoidal_encoding(max_seq_len: int, d_model: int) -> torch.Tensor:
    """
    Create sinusoidal positional encoding matrix.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        max_seq_len: Maximum sequence length
        d_model: Embedding dimension (must be even)
    
    Returns:
        Tensor of shape (max_seq_len, d_model)
    """
    # API hints:
    # - torch.zeros(max_seq_len, d_model) -> initialize output
    # - torch.arange(0, max_seq_len).unsqueeze(1) -> position column vector
    # - torch.arange(0, d_model, 2) -> even dimension indices
    # - torch.exp(x * (-math.log(10000.0) / d_model)) -> compute div_term
    # - pe[:, 0::2] = torch.sin(...) -> fill even indices
    # - pe[:, 1::2] = torch.cos(...) -> fill odd indices
    return None


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding module.
    """
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        # API hints:
        # - nn.Dropout(p=dropout) -> dropout layer
        # - create_sinusoidal_encoding(max_seq_len, d_model) -> get encoding matrix
        # - pe.unsqueeze(0) -> add batch dimension (1, max_seq_len, d_model)
        # - self.register_buffer('pe', pe) -> register as non-trainable buffer
        self.dropout = nn.Dropout(p=dropout)
        self.pe = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        # API hints:
        # - self.pe[:, :x.size(1), :] -> slice to match input seq_len
        # - x + self.pe[...] -> add positional encoding
        # - self.dropout(x) -> apply dropout
        return None


# ============================================================================
# Exercise 2: Learned Positional Encoding
# ============================================================================

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding using nn.Embedding.
    """
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        # API hints:
        # - nn.Embedding(max_seq_len, d_model) -> learnable position embeddings
        # - nn.Dropout(p=dropout) -> dropout layer
        self.position_embedding = None
        self.dropout = nn.Dropout(p=dropout)
        self.max_seq_len = max_seq_len
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input.
        
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        # API hints:
        # - x.size(1) -> sequence length
        # - torch.arange(seq_len, device=x.device) -> position indices
        # - self.position_embedding(positions) -> (seq_len, d_model)
        # - pos_embed.unsqueeze(0) -> add batch dim for broadcasting
        # - x + pos_embed -> add to input
        # - self.dropout(x) -> apply dropout
        return None


# ============================================================================
# Exercise 3: Relative Positional Bias (Simplified)
# ============================================================================

def create_relative_position_bias(seq_len: int, num_heads: int) -> nn.Parameter:
    """
    Create learnable relative position bias matrix.
    Simplified version of T5-style relative attention bias.
    
    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads
    
    Returns:
        Parameter of shape (num_heads, seq_len, seq_len)
    """
    # API hints:
    # - torch.zeros(num_heads, seq_len, seq_len) -> initialize bias
    # - nn.Parameter(tensor) -> wrap as learnable parameter
    return None


def compute_relative_positions(seq_len: int) -> torch.Tensor:
    """
    Compute relative position matrix where entry (i,j) = j - i.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Tensor of shape (seq_len, seq_len) with relative positions
    """
    # API hints:
    # - torch.arange(seq_len) -> [0, 1, 2, ..., seq_len-1]
    # - positions.unsqueeze(0) - positions.unsqueeze(1) -> broadcasting
    # - Result: row indices - column indices
    return None


# ============================================================================
# Exercise 4: RoPE (Rotary Position Embeddings) Basics
# ============================================================================

def compute_rope_frequencies(dim: int, max_seq_len: int, 
                              base: float = 10000.0) -> torch.Tensor:
    """
    Compute frequency bases for RoPE.
    
    theta_i = base^(-2i/dim) for i in [0, 1, ..., dim/2 - 1]
    
    Args:
        dim: Embedding dimension (must be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation
    
    Returns:
        Tensor of shape (max_seq_len, dim/2) containing position * frequency
    """
    # API hints:
    # - torch.arange(0, dim, 2).float() -> even dimension indices
    # - 1.0 / (base ** (indices / dim)) -> frequency per dimension
    # - torch.arange(max_seq_len).float() -> position indices
    # - torch.outer(positions, freq) -> outer product (max_seq_len, dim/2)
    return None


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to input.
    
    RoPE rotates pairs of dimensions based on position.
    For each pair (x_2i, x_2i+1):
        x'_2i = x_2i * cos(theta) - x_2i+1 * sin(theta)
        x'_2i+1 = x_2i * sin(theta) + x_2i+1 * cos(theta)
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        freqs: Frequency tensor of shape (seq_len, dim/2)
    
    Returns:
        Rotated tensor of same shape
    """
    # API hints:
    # - x.view(*x.shape[:-1], -1, 2) -> reshape to pairs (..., dim/2, 2)
    # - freqs[:x.size(1)] -> slice freqs to actual sequence length
    # - torch.cos(freqs), torch.sin(freqs) -> compute rotation components
    # - Apply rotation: x' = x * cos - x_rotated * sin, x_rotated' = x * sin + x_rotated * cos
    # - torch.stack([...], dim=-1) -> combine rotated pairs
    # - result.view(*x.shape) -> reshape back
    return None


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding module.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        # API hints:
        # - compute_rope_frequencies(dim, max_seq_len, base) -> precompute freqs
        # - self.register_buffer('freqs', freqs) -> register as non-trainable buffer
        self.freqs = None
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor."""
        # API hints:
        # - apply_rope(x, self.freqs) -> apply rotary encoding
        return None


# ============================================================================
# Exercise 5: Compare Positional Encoding Methods
# ============================================================================

def compare_encoding_properties():
    """
    Compare properties of different positional encoding methods.
    Returns dict with parameter counts and config.
    """
    # API hints:
    # - SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    # - LearnedPositionalEncoding(d_model, max_len, dropout=0.0)
    # - sum(p.numel() for p in module.parameters()) -> count parameters
    # - Sinusoidal has 0 learnable params (buffer only)
    # - Learned has max_len * d_model params
    return None


# ============================================================================
# Exercise 6: Transformer Embedding Layer
# ============================================================================

class TransformerEmbedding(nn.Module):
    """
    Complete transformer embedding: Token + Positional encoding.
    """
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int,
                 dropout: float = 0.1, use_learned_pos: bool = False):
        super().__init__()
        # API hints:
        # - nn.Embedding(vocab_size, d_model) -> token embedding
        # - If use_learned_pos: LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        # - Else: SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.token_embedding = None
        self.pos_encoding = None
        self.d_model = d_model
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens and add positional encoding.
        
        Args:
            token_ids: Integer tensor of shape (batch, seq_len)
        
        Returns:
            Embeddings of shape (batch, seq_len, d_model)
        """
        # API hints:
        # - self.token_embedding(token_ids) -> get token embeddings
        # - Multiply by math.sqrt(self.d_model) -> scaling (common in transformers)
        # - self.pos_encoding(x) -> add positional encoding
        return None


if __name__ == "__main__":
    print("Day 14: Positional Encoding")
    print("=" * 50)
    
    d_model = 64
    max_seq_len = 100
    
    # Test sinusoidal encoding
    pe = create_sinusoidal_encoding(max_seq_len, d_model)
    print(f"\nSinusoidal encoding shape: {pe.shape}")
    
    # Test sinusoidal module
    sin_pe = SinusoidalPositionalEncoding(d_model, max_seq_len)
    x = torch.randn(4, 20, d_model)
    if sin_pe.pe is not None:
        out = sin_pe(x)
        print(f"Sinusoidal module: {x.shape} -> {out.shape}")
    
    # Test learned encoding
    learn_pe = LearnedPositionalEncoding(d_model, max_seq_len)
    if learn_pe.position_embedding is not None:
        out = learn_pe(x)
        print(f"Learned module: {x.shape} -> {out.shape}")
    
    # Test RoPE frequencies
    freqs = compute_rope_frequencies(d_model, max_seq_len)
    print(f"\nRoPE frequencies shape: {freqs.shape}")
    
    # Test relative positions
    rel_pos = compute_relative_positions(5)
    print(f"\nRelative positions (5x5):\n{rel_pos}")
    
    # Test TransformerEmbedding
    trans_emb = TransformerEmbedding(vocab_size=10000, d_model=d_model, 
                                      max_seq_len=max_seq_len)
    if trans_emb.token_embedding is not None:
        tokens = torch.randint(0, 10000, (4, 30))
        out = trans_emb(tokens)
        print(f"\nTransformerEmbedding: {tokens.shape} -> {out.shape}")
    
    print("\nRun test_day14.py to verify all implementations!")
