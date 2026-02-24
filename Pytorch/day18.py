"""
Day 18: Transformer Decoder Block
=================================
Estimated time: 1-2 hours
Prerequisites: Day 16 (multi-head attention), Day 17 (encoder block)

Learning objectives:
- Understand masked self-attention for autoregressive generation
- Implement cross-attention (decoder attending to encoder)
- Build complete decoder block with all three sublayers
- Understand the difference between encoder and decoder architectures

Key Concepts:
-------------
The Transformer decoder block has THREE sublayers (vs encoder's two):

1. Masked Self-Attention:
   - Prevents positions from attending to future positions
   - Uses a causal mask: lower triangular matrix
   - Essential for autoregressive (left-to-right) generation

2. Cross-Attention (Encoder-Decoder Attention):
   - Queries come from decoder
   - Keys and Values come from encoder output
   - Allows decoder to "look at" the source sequence

3. Feed-Forward Network:
   - Same as encoder FFN

Structure (Pre-Norm):
    x = x + MaskedSelfAttention(LayerNorm(x))
    x = x + CrossAttention(LayerNorm(x), encoder_output)
    x = x + FFN(LayerNorm(x))

Causal Mask Example (seq_len=4):
    [[1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 1, 1, 0],
     [1, 1, 1, 1]]
    Position i can only attend to positions <= i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Exercise 1: Causal (Look-Ahead) Mask
# ============================================================================

def create_causal_mask(seq_len, device=None):
    """
    Create a causal mask for masked self-attention.
    
    The mask prevents positions from attending to future positions.
    Position i can only attend to positions 0, 1, ..., i
    
    Args:
        seq_len: Length of the sequence
        device: Device to create tensor on
    
    Returns:
        mask: [seq_len, seq_len] lower triangular matrix of 1s
              (1 = attend, 0 = mask out)
    """
    # TODO: Create a lower triangular matrix of ones
    # API hints:
    # - torch.ones(seq_len, seq_len, device=device) -> matrix of ones
    # - torch.tril(tensor) -> lower triangular part of matrix
    return None


def create_causal_mask_batched(batch_size, num_heads, seq_len, device=None):
    """
    Create a batched causal mask for multi-head attention.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        device: Device to create tensor on
    
    Returns:
        mask: [batch_size, num_heads, seq_len, seq_len]
              (can also be [1, 1, seq_len, seq_len] for broadcasting)
    """
    # TODO: Create causal mask and expand for batch/heads
    # API hints:
    # - create_causal_mask(seq_len, device) -> base [seq_len, seq_len] mask
    # - tensor.unsqueeze(0).unsqueeze(0) -> add batch and head dims
    # - tensor.expand(batch_size, num_heads, -1, -1) -> expand to full size
    return None


# ============================================================================
# Exercise 2: Masked Multi-Head Self-Attention
# ============================================================================

class MaskedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with optional causal masking.
    
    This is used as the first sublayer in the decoder.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # TODO: Create Q, K, V, output projections, and dropout
        # API hints:
        # - nn.Linear(d_model, d_model) -> projection layer
        # - nn.Dropout(dropout) -> dropout layer
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None
        self.dropout = None
    
    def forward(self, x, mask=None, use_causal_mask=True):
        """
        Args:
            x: [batch, seq, d_model]
            mask: Optional additional mask (e.g., padding mask)
            use_causal_mask: Whether to apply causal masking
        
        Returns:
            output: [batch, seq, d_model]
            attention_weights: [batch, num_heads, seq, seq]
        """
        # TODO: Implement masked multi-head attention
        # API hints:
        # - self.W_q(x), self.W_k(x), self.W_v(x) -> project to Q, K, V
        # - tensor.view(batch, seq, num_heads, d_k).transpose(1, 2) -> reshape for multi-head
        # - torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) -> attention scores
        # - create_causal_mask(seq_len, device) -> causal mask
        # - scores.masked_fill(mask == 0, float('-inf')) -> apply mask
        # - F.softmax(scores, dim=-1) -> attention weights
        # - torch.matmul(attn_weights, V) -> weighted sum of values
        # - tensor.transpose(1, 2).contiguous().view(batch, seq, d_model) -> reshape back
        # - self.W_o(attn_output) -> final projection
        return None, None


# ============================================================================
# Exercise 3: Cross-Attention (Encoder-Decoder Attention)
# ============================================================================

class CrossAttention(nn.Module):
    """
    Cross-attention where decoder attends to encoder output.
    
    - Queries (Q): from decoder hidden states
    - Keys (K) and Values (V): from encoder output
    
    No causal mask is needed because we want the decoder to see
    the entire encoder output at every position.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # TODO: Create Q, K, V, output projections, and dropout
        # API hints:
        # - nn.Linear(d_model, d_model) -> projection layer
        # - nn.Dropout(dropout) -> dropout layer
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None
        self.dropout = None
    
    def forward(self, decoder_hidden, encoder_output, encoder_mask=None):
        """
        Args:
            decoder_hidden: [batch, tgt_seq, d_model] - decoder's hidden states
            encoder_output: [batch, src_seq, d_model] - encoder's output
            encoder_mask: Optional [batch, 1, 1, src_seq] padding mask for source
        
        Returns:
            output: [batch, tgt_seq, d_model]
            attention_weights: [batch, num_heads, tgt_seq, src_seq]
        """
        # TODO: Implement cross-attention
        # API hints:
        # - self.W_q(decoder_hidden) -> Q from decoder
        # - self.W_k(encoder_output), self.W_v(encoder_output) -> K, V from encoder
        # - tensor.view(batch, seq, num_heads, d_k).transpose(1, 2) -> reshape for multi-head
        # - torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) -> attention scores
        # - scores.masked_fill(encoder_mask == 0, float('-inf')) -> apply mask
        # - F.softmax(scores, dim=-1) -> attention weights
        # - torch.matmul(attn_weights, V) -> weighted sum of values
        # - tensor.transpose(1, 2).contiguous().view(batch, tgt_seq, d_model) -> reshape back
        # - self.W_o(attn_output) -> final projection
        return None, None


# ============================================================================
# Exercise 4: Feed-Forward Network (reuse from Day 17)
# ============================================================================

class FeedForward(nn.Module):
    """Position-wise FFN (same as encoder)."""
    
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        
        d_ff = d_ff or d_model * 4
        
        # TODO: Create two linear layers and dropout
        # API hints:
        # - nn.Linear(d_model, d_ff) -> expansion layer
        # - nn.Linear(d_ff, d_model) -> projection layer
        # - nn.Dropout(dropout) -> dropout layer
        self.linear1 = None
        self.linear2 = None
        self.dropout = None
    
    def forward(self, x):
        # TODO: linear1 -> GELU -> dropout -> linear2
        # API hints:
        # - F.gelu(tensor) -> GELU activation
        # - self.dropout(tensor) -> apply dropout
        return None


# ============================================================================
# Exercise 5: Layer Normalization (reuse from Day 17)
# ============================================================================

class LayerNorm(nn.Module):
    """Layer Normalization."""
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        
        # TODO: Create learnable scale and shift parameters
        # API hints:
        # - nn.Parameter(torch.ones(d_model)) -> learnable scale (gamma)
        # - nn.Parameter(torch.zeros(d_model)) -> learnable shift (beta)
        self.gamma = None
        self.beta = None
    
    def forward(self, x):
        # TODO: Normalize over last dimension, apply scale and shift
        # API hints:
        # - x.mean(dim=-1, keepdim=True) -> mean over last dim
        # - torch.sqrt(var + self.eps) -> standard deviation
        # - self.gamma * x_norm + self.beta -> scale and shift
        return None


# ============================================================================
# Exercise 6: Complete Pre-Norm Decoder Block
# ============================================================================

class PreNormDecoderBlock(nn.Module):
    """
    Pre-Norm Transformer Decoder Block.
    
    Structure:
        x = x + MaskedSelfAttention(LayerNorm(x))
        x = x + CrossAttention(LayerNorm(x), encoder_output)
        x = x + FFN(LayerNorm(x))
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0):
        super().__init__()
        
        # TODO: Create masked self-attention, cross-attention, FFN, layer norms, dropout
        # API hints:
        # - MaskedMultiHeadAttention(d_model, num_heads, dropout) -> masked self-attention
        # - CrossAttention(d_model, num_heads, dropout) -> cross-attention
        # - FeedForward(d_model, d_ff, dropout) -> feed-forward
        # - LayerNorm(d_model) -> layer norm (need 3 for 3 sublayers)
        # - nn.Dropout(dropout) -> dropout layer
        self.self_attn = None
        self.cross_attn = None
        self.ffn = None
        self.norm1 = None
        self.norm2 = None
        self.norm3 = None
        self.dropout = None
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch, tgt_seq, d_model] - decoder input
            encoder_output: [batch, src_seq, d_model] - encoder output
            src_mask: Optional padding mask for source (encoder output)
            tgt_mask: Optional additional mask for target (beyond causal)
        
        Returns:
            [batch, tgt_seq, d_model]
        """
        # TODO: Implement pre-norm decoder block with 3 sublayers
        # API hints:
        # - Sublayer 1: x = x + dropout(self_attn(norm1(x), mask=tgt_mask))
        # - Sublayer 2: x = x + dropout(cross_attn(norm2(x), encoder_output, src_mask))
        # - Sublayer 3: x = x + dropout(ffn(norm3(x)))
        # - self.self_attn(x, mask) returns (output, attn_weights)
        # - self.cross_attn(x, encoder_output, mask) returns (output, attn_weights)
        return None


# ============================================================================
# Exercise 7: Transformer Decoder Stack
# ============================================================================

class TransformerDecoder(nn.Module):
    """
    Stack of N decoder blocks with final layer norm.
    """
    
    def __init__(self, d_model, num_heads, num_layers, d_ff=None, dropout=0.0):
        super().__init__()
        
        # TODO: Create stack of decoder blocks and final norm
        # API hints:
        # - nn.ModuleList([...]) -> list of modules for iteration
        # - PreNormDecoderBlock(d_model, num_heads, d_ff, dropout) -> decoder block
        # - LayerNorm(d_model) -> final layer norm
        self.layers = None
        self.final_norm = None
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: [batch, tgt_seq, d_model]
            encoder_output: [batch, src_seq, d_model]
            src_mask: Source padding mask
            tgt_mask: Target additional mask
        
        Returns:
            [batch, tgt_seq, d_model]
        """
        # TODO: Pass through all layers, apply final norm
        # API hints:
        # - for layer in self.layers: x = layer(x, encoder_output, src_mask, tgt_mask)
        # - self.final_norm(x) -> apply final normalization
        return None


# ============================================================================
# Visualization: Attention Patterns
# ============================================================================

def visualize_causal_mask():
    """Show the structure of a causal attention mask."""
    seq_len = 6
    mask = create_causal_mask(seq_len)
    
    if mask is None:
        print("Causal mask not implemented yet")
        return
    
    print("Causal Attention Mask (1=attend, 0=block):")
    print(mask.int())
    print("\nToken 0 can see: only itself")
    print("Token 1 can see: tokens 0, 1")
    print("Token 5 can see: tokens 0, 1, 2, 3, 4, 5 (all)")


def visualize_decoder_attention():
    """Demonstrate the three types of attention in decoder."""
    print("\n" + "="*50)
    print("Decoder Attention Types:")
    print("="*50)
    
    print("\n1. Masked Self-Attention:")
    print("   - Decoder tokens attend to previous decoder tokens")
    print("   - Uses causal mask to prevent looking ahead")
    print("   - Query, Key, Value all from decoder")
    
    print("\n2. Cross-Attention:")
    print("   - Decoder tokens attend to encoder output")
    print("   - No causal mask (see entire source)")
    print("   - Query from decoder, Key/Value from encoder")
    
    print("\n3. Feed-Forward:")
    print("   - Position-wise transformation")
    print("   - No attention involved")


if __name__ == "__main__":
    print("Day 18: Transformer Decoder Block")
    print("=" * 50)
    
    # Demo: Causal mask
    print("\nDemo: Causal Mask")
    visualize_causal_mask()
    
    # Demo: Decoder block
    print("\n\nDemo: Decoder Block")
    d_model, num_heads = 64, 4
    batch, tgt_seq, src_seq = 2, 8, 12
    
    decoder_block = PreNormDecoderBlock(d_model, num_heads)
    
    # Simulated inputs
    decoder_input = torch.randn(batch, tgt_seq, d_model)
    encoder_output = torch.randn(batch, src_seq, d_model)
    
    output = decoder_block(decoder_input, encoder_output)
    if output is not None:
        print(f"Decoder input: {decoder_input.shape}")
        print(f"Encoder output: {encoder_output.shape}")
        print(f"Decoder block output: {output.shape}")
    
    visualize_decoder_attention()
    
    print("\nRun test_day18.py to verify your implementations!")
