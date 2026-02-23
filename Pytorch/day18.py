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
    # HINT: torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask = None  # Replace
    
    return mask


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
    # HINT: Create [seq_len, seq_len] mask and use .unsqueeze() or .expand()
    mask = None  # Replace
    
    return mask


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
        
        # TODO: Create Q, K, V, and output projections
        self.W_q = None  # Replace: nn.Linear(d_model, d_model)
        self.W_k = None  # Replace: nn.Linear(d_model, d_model)
        self.W_v = None  # Replace: nn.Linear(d_model, d_model)
        self.W_o = None  # Replace: nn.Linear(d_model, d_model)
        
        self.dropout = None  # Replace: nn.Dropout(dropout)
    
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
        batch_size, seq_len, _ = x.shape
        
        # TODO: Project to Q, K, V
        Q = None  # Replace: self.W_q(x)
        K = None  # Replace: self.W_k(x)
        V = None  # Replace: self.W_v(x)
        
        # TODO: Reshape for multi-head: [batch, seq, d_model] -> [batch, heads, seq, d_k]
        # HINT: Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        Q = None  # Replace
        K = None  # Replace
        V = None  # Replace
        
        # TODO: Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = None  # Replace: torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # TODO: Apply causal mask if requested
        if use_causal_mask:
            # Create causal mask [1, 1, seq, seq]
            causal_mask = create_causal_mask(seq_len, device=x.device)
            if causal_mask is not None:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                # Mask out future positions with -inf
                # HINT: scores = scores.masked_fill(causal_mask == 0, float('-inf'))
                pass  # Replace with masking
        
        # TODO: Apply additional mask if provided (e.g., padding mask)
        if mask is not None:
            # HINT: scores = scores.masked_fill(mask == 0, float('-inf'))
            pass  # Replace
        
        # TODO: Softmax and dropout
        attention_weights = None  # Replace: F.softmax(scores, dim=-1)
        attention_weights = None  # Replace: self.dropout(attention_weights)
        
        # TODO: Apply attention to V
        attn_output = None  # Replace: torch.matmul(attention_weights, V)
        
        # TODO: Reshape back: [batch, heads, seq, d_k] -> [batch, seq, d_model]
        # HINT: attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = None  # Replace
        
        # TODO: Final projection
        output = None  # Replace: self.W_o(attn_output)
        
        return output, attention_weights


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
        
        # TODO: Create projections
        # Note: Q projection for decoder, K and V projections for encoder output
        self.W_q = None  # Replace: nn.Linear(d_model, d_model)
        self.W_k = None  # Replace: nn.Linear(d_model, d_model)
        self.W_v = None  # Replace: nn.Linear(d_model, d_model)
        self.W_o = None  # Replace: nn.Linear(d_model, d_model)
        
        self.dropout = None  # Replace: nn.Dropout(dropout)
    
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
        batch_size = decoder_hidden.shape[0]
        tgt_seq = decoder_hidden.shape[1]
        src_seq = encoder_output.shape[1]
        
        # TODO: Queries from decoder, Keys and Values from encoder
        Q = None  # Replace: self.W_q(decoder_hidden)
        K = None  # Replace: self.W_k(encoder_output)
        V = None  # Replace: self.W_v(encoder_output)
        
        # TODO: Reshape for multi-head
        # Q: [batch, tgt_seq, heads, d_k] -> [batch, heads, tgt_seq, d_k]
        # K, V: [batch, src_seq, heads, d_k] -> [batch, heads, src_seq, d_k]
        Q = None  # Replace
        K = None  # Replace
        V = None  # Replace
        
        # TODO: Compute attention scores
        scores = None  # Replace: torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # TODO: Apply encoder padding mask if provided
        if encoder_mask is not None:
            pass  # Replace: scores = scores.masked_fill(encoder_mask == 0, float('-inf'))
        
        # TODO: Softmax and dropout
        attention_weights = None  # Replace
        
        # TODO: Apply attention to V
        attn_output = None  # Replace
        
        # TODO: Reshape and project
        attn_output = None  # Replace: reshape back to [batch, tgt_seq, d_model]
        output = None  # Replace: self.W_o(attn_output)
        
        return output, attention_weights


# ============================================================================
# Exercise 4: Feed-Forward Network (reuse from Day 17)
# ============================================================================

class FeedForward(nn.Module):
    """Position-wise FFN (same as encoder)."""
    
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        
        d_ff = d_ff or d_model * 4
        
        # TODO: Two linear layers with GELU activation
        self.linear1 = None  # Replace: nn.Linear(d_model, d_ff)
        self.linear2 = None  # Replace: nn.Linear(d_ff, d_model)
        self.dropout = None  # Replace: nn.Dropout(dropout)
    
    def forward(self, x):
        # TODO: linear1 -> GELU -> dropout -> linear2
        x = None  # Replace: self.dropout(F.gelu(self.linear1(x)))
        x = None  # Replace: self.linear2(x)
        return x


# ============================================================================
# Exercise 5: Layer Normalization (reuse from Day 17)
# ============================================================================

class LayerNorm(nn.Module):
    """Layer Normalization."""
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        
        # TODO: Learnable scale and shift
        self.gamma = None  # Replace: nn.Parameter(torch.ones(d_model))
        self.beta = None   # Replace: nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        # TODO: Normalize over last dimension
        mean = None  # Replace: x.mean(dim=-1, keepdim=True)
        var = None   # Replace: ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = None  # Replace: (x - mean) / torch.sqrt(var + self.eps)
        return None  # Replace: self.gamma * x_norm + self.beta


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
        
        # TODO: Masked self-attention sublayer
        self.self_attn = None  # Replace: MaskedMultiHeadAttention(d_model, num_heads, dropout)
        
        # TODO: Cross-attention sublayer
        self.cross_attn = None  # Replace: CrossAttention(d_model, num_heads, dropout)
        
        # TODO: Feed-forward sublayer
        self.ffn = None  # Replace: FeedForward(d_model, d_ff, dropout)
        
        # TODO: Layer norms (one for each sublayer)
        self.norm1 = None  # Replace: LayerNorm(d_model)
        self.norm2 = None  # Replace: LayerNorm(d_model)
        self.norm3 = None  # Replace: LayerNorm(d_model)
        
        self.dropout = None  # Replace: nn.Dropout(dropout)
    
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
        # TODO: Sublayer 1 - Masked Self-Attention (Pre-Norm)
        # Step 1: Normalize
        normed = None  # Replace: self.norm1(x)
        
        # Step 2: Masked self-attention (use_causal_mask=True by default)
        attn_output, _ = self.self_attn(normed, mask=tgt_mask) if self.self_attn else (None, None)
        
        # Step 3: Residual connection with dropout
        x = None  # Replace: x + self.dropout(attn_output)
        
        # TODO: Sublayer 2 - Cross-Attention (Pre-Norm)
        # Step 1: Normalize
        normed = None  # Replace: self.norm2(x)
        
        # Step 2: Cross-attention (decoder attends to encoder)
        cross_output, _ = self.cross_attn(normed, encoder_output, src_mask) if self.cross_attn else (None, None)
        
        # Step 3: Residual connection
        x = None  # Replace: x + self.dropout(cross_output)
        
        # TODO: Sublayer 3 - Feed-Forward (Pre-Norm)
        # Step 1: Normalize
        normed = None  # Replace: self.norm3(x)
        
        # Step 2: FFN
        ffn_output = None  # Replace: self.ffn(normed)
        
        # Step 3: Residual connection
        x = None  # Replace: x + self.dropout(ffn_output)
        
        return x


# ============================================================================
# Exercise 7: Transformer Decoder Stack
# ============================================================================

class TransformerDecoder(nn.Module):
    """
    Stack of N decoder blocks with final layer norm.
    """
    
    def __init__(self, d_model, num_heads, num_layers, d_ff=None, dropout=0.0):
        super().__init__()
        
        # TODO: Create stack of decoder blocks
        self.layers = None  # Replace: nn.ModuleList([PreNormDecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        # TODO: Final layer norm (for pre-norm architecture)
        self.final_norm = None  # Replace: LayerNorm(d_model)
    
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
        # TODO: Pass through all decoder layers
        if self.layers is not None:
            for layer in self.layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # TODO: Apply final layer norm
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        return x


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
