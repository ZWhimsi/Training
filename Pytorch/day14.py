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
    
    TODO: Implement sinusoidal encoding
    HINT:
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create division term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        return pe
    """
    return torch.zeros(max_seq_len, d_model)  # Replace


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding module.
    """
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        """
        TODO: Create and register positional encoding buffer
        HINT:
            self.dropout = nn.Dropout(p=dropout)
            
            # Create encoding and register as buffer (not a parameter)
            pe = create_sinusoidal_encoding(max_seq_len, d_model)
            pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model) for broadcasting
            self.register_buffer('pe', pe)
        """
        self.dropout = nn.Dropout(p=dropout)
        self.pe = None  # Replace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        
        TODO: Add positional encoding
        HINT:
            # x has shape (batch, seq_len, d_model)
            # self.pe has shape (1, max_seq_len, d_model)
            # Slice pe to match input sequence length
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)
        """
        return x  # Replace


# ============================================================================
# Exercise 2: Learned Positional Encoding
# ============================================================================

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding using nn.Embedding.
    """
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        """
        TODO: Create learned position embeddings
        HINT:
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
            self.dropout = nn.Dropout(p=dropout)
        """
        self.position_embedding = None  # Replace
        self.dropout = nn.Dropout(p=dropout)
        self.max_seq_len = max_seq_len
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input.
        
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        
        TODO: Add learned positions
        HINT:
            seq_len = x.size(1)
            positions = torch.arange(seq_len, device=x.device)
            pos_embed = self.position_embedding(positions)  # (seq_len, d_model)
            x = x + pos_embed.unsqueeze(0)  # Broadcast over batch
            return self.dropout(x)
        """
        return x  # Replace


# ============================================================================
# Exercise 3: Relative Positional Bias (Simplified)
# ============================================================================

def create_relative_position_bias(seq_len: int, num_heads: int) -> nn.Parameter:
    """
    Create learnable relative position bias matrix.
    
    This is a simplified version of what's used in models like T5.
    The bias is added to attention scores based on relative positions.
    
    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads
    
    Returns:
        Parameter of shape (num_heads, seq_len, seq_len)
    
    TODO: Create relative position bias
    HINT:
        # Initialize with zeros or small random values
        bias = torch.zeros(num_heads, seq_len, seq_len)
        return nn.Parameter(bias)
    """
    return nn.Parameter(torch.zeros(1))  # Replace


def compute_relative_positions(seq_len: int) -> torch.Tensor:
    """
    Compute relative position matrix.
    
    For positions i and j, relative position is j - i.
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Tensor of shape (seq_len, seq_len) with relative positions
    
    Example for seq_len=4:
        [[ 0,  1,  2,  3],
         [-1,  0,  1,  2],
         [-2, -1,  0,  1],
         [-3, -2, -1,  0]]
    
    TODO: Compute relative positions
    HINT:
        positions = torch.arange(seq_len)
        # Broadcasting: positions.unsqueeze(0) - positions.unsqueeze(1)
        return positions.unsqueeze(0) - positions.unsqueeze(1)
    """
    return torch.zeros(seq_len, seq_len)  # Replace


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
    
    TODO: Compute RoPE frequencies
    HINT:
        # Compute frequencies
        freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # Compute positions * frequencies
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, freq)  # (max_seq_len, dim/2)
        
        return freqs
    """
    return torch.zeros(max_seq_len, dim // 2)  # Replace


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
    
    TODO: Apply RoPE
    HINT:
        # Split into pairs
        x_pairs = x.view(*x.shape[:-1], -1, 2)  # (..., dim/2, 2)
        
        # Get sin and cos
        freqs = freqs[:x.size(1)]  # Slice to actual seq_len
        cos_freq = torch.cos(freqs).unsqueeze(0).unsqueeze(-1)
        sin_freq = torch.sin(freqs).unsqueeze(0).unsqueeze(-1)
        
        # Rotate
        x_rotated = torch.stack([
            x_pairs[..., 0] * cos_freq.squeeze(-1) - x_pairs[..., 1] * sin_freq.squeeze(-1),
            x_pairs[..., 0] * sin_freq.squeeze(-1) + x_pairs[..., 1] * cos_freq.squeeze(-1)
        ], dim=-1)
        
        return x_rotated.view(*x.shape)
    """
    return x  # Replace


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding module.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        """
        TODO: Precompute and register frequencies
        HINT:
            freqs = compute_rope_frequencies(dim, max_seq_len, base)
            self.register_buffer('freqs', freqs)
        """
        self.freqs = None  # Replace
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input.
        
        TODO: Apply rotary encoding
        HINT: return apply_rope(x, self.freqs)
        """
        return x  # Replace


# ============================================================================
# Exercise 5: Compare Positional Encoding Methods
# ============================================================================

def compare_encoding_properties():
    """
    Compare properties of different positional encoding methods.
    
    TODO: Create instances and compare
    HINT:
        d_model = 64
        max_len = 100
        
        sinusoidal = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
        learned = LearnedPositionalEncoding(d_model, max_len, dropout=0.0)
        
        # Sinusoidal: check parameter count (should be 0 for encoding)
        sin_params = sum(p.numel() for p in sinusoidal.parameters())
        
        # Learned: check parameter count
        learn_params = sum(p.numel() for p in learned.parameters())
        
        return {
            'sinusoidal_params': sin_params,
            'learned_params': learn_params,
            'd_model': d_model,
            'max_len': max_len
        }
    """
    return {
        'sinusoidal_params': 0,
        'learned_params': 0,
        'd_model': 64,
        'max_len': 100
    }  # Replace


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
        """
        TODO: Create token embedding and positional encoding
        HINT:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            
            if use_learned_pos:
                self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
            else:
                self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
            
            self.d_model = d_model
        """
        self.token_embedding = None  # Replace
        self.pos_encoding = None     # Replace
        self.d_model = d_model
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed tokens and add positional encoding.
        
        Args:
            token_ids: Integer tensor of shape (batch, seq_len)
        
        Returns:
            Embeddings of shape (batch, seq_len, d_model)
        
        TODO: Apply token embedding and positional encoding
        HINT:
            # Scale embeddings (common practice in transformers)
            x = self.token_embedding(token_ids) * math.sqrt(self.d_model)
            return self.pos_encoding(x)
        """
        return torch.zeros(token_ids.shape[0], token_ids.shape[1], self.d_model)  # Replace


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
