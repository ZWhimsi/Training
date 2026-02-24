"""
Day 19: Full Transformer Architecture
=====================================
Estimated time: 1-2 hours
Prerequisites: Day 17 (encoder), Day 18 (decoder)

Learning objectives:
- Combine encoder and decoder into full Transformer
- Implement proper embeddings with positional encoding
- Build sequence-to-sequence model (e.g., for translation)
- Understand the complete forward pass
- Compare with PyTorch's nn.Transformer

Key Concepts:
-------------
The full Transformer architecture for sequence-to-sequence tasks:

1. Encoder:
   - Processes source sequence (e.g., English sentence)
   - Produces contextual representations for each token
   - Source tokens can attend to all other source tokens

2. Decoder:
   - Generates target sequence (e.g., French translation)
   - Uses masked self-attention (causal, left-to-right)
   - Uses cross-attention to attend to encoder output
   - Outputs probability distribution over vocabulary

Full Architecture:
    src_tokens -> Embedding + PosEnc -> Encoder Stack -> encoder_output
    tgt_tokens -> Embedding + PosEnc -> Decoder Stack(with encoder_output) -> Linear -> vocab_logits

Training vs Inference:
- Training: Teacher forcing - use ground truth target as decoder input
- Inference: Autoregressive - feed previous predictions back
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Exercise 1: Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need".
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # TODO: Create positional encoding matrix and register as buffer
        # API hints:
        # - torch.zeros(max_len, d_model) -> empty encoding matrix
        # - torch.arange(0, max_len).unsqueeze(1).float() -> position indices
        # - torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) -> div_term
        # - torch.sin(position * div_term) -> sin for even indices (pe[:, 0::2])
        # - torch.cos(position * div_term) -> cos for odd indices (pe[:, 1::2])
        # - self.register_buffer('pe', pe.unsqueeze(0)) -> register as buffer [1, max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            [batch, seq, d_model] with positional encoding added
        """
        # TODO: Add positional encoding and apply dropout
        # API hints:
        # - x + self.pe[:, :x.size(1)] -> add positional encoding
        # - self.dropout(x) -> apply dropout
        return None


# ============================================================================
# Exercise 2: Token Embedding with Scaling
# ============================================================================

class TransformerEmbedding(nn.Module):
    """
    Combines token embedding with positional encoding.
    
    The original Transformer scales embeddings by sqrt(d_model)
    before adding positional encoding.
    """
    
    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1, padding_idx=None):
        super().__init__()
        
        self.d_model = d_model
        
        # TODO: Create token embedding and positional encoding
        # API hints:
        # - nn.Embedding(vocab_size, d_model, padding_idx=padding_idx) -> token embedding
        # - PositionalEncoding(d_model, max_len, dropout) -> positional encoding
        self.token_embedding = None
        self.pos_encoding = None
    
    def forward(self, tokens):
        """
        Args:
            tokens: [batch, seq] token indices
        Returns:
            [batch, seq, d_model] embedded tokens with position info
        """
        # TODO: Get embeddings, scale by sqrt(d_model), add positional encoding
        # API hints:
        # - self.token_embedding(tokens) -> token embeddings
        # - math.sqrt(self.d_model) -> scaling factor
        # - self.pos_encoding(x) -> add positional encoding
        return None


# ============================================================================
# Exercise 3: Encoder Stack (from Day 17)
# ============================================================================

class LayerNorm(nn.Module):
    """Layer Normalization."""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_q, seq_k = query.shape[1], key.shape[1]
        
        Q = self.W_q(query).view(batch_size, seq_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_k, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)
        
        return self.W_o(attn_output), attn_weights


class FeedForward(nn.Module):
    """Position-wise FFN."""
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class EncoderBlock(nn.Module):
    """Pre-Norm Encoder Block."""
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of encoder blocks."""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff=None, dropout=0.0):
        super().__init__()
        
        # TODO: Create encoder layers and final norm
        # API hints:
        # - nn.ModuleList([...]) -> list of modules
        # - EncoderBlock(d_model, num_heads, d_ff, dropout) -> encoder block
        # - LayerNorm(d_model) -> final layer norm
        self.layers = None
        self.final_norm = None
    
    def forward(self, x, mask=None):
        # TODO: Pass through all layers, apply final norm
        # API hints:
        # - for layer in self.layers: x = layer(x, mask)
        # - self.final_norm(x) -> final normalization
        return None


# ============================================================================
# Exercise 4: Decoder Stack (from Day 18)
# ============================================================================

def create_causal_mask(seq_len, device=None):
    """Create causal (look-ahead) mask."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


class DecoderBlock(nn.Module):
    """Pre-Norm Decoder Block with masked self-attention and cross-attention."""
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0):
        super().__init__()
        
        # TODO: Create self-attention, cross-attention, FFN, layer norms, dropout
        # API hints:
        # - MultiHeadAttention(d_model, num_heads, dropout) -> for self and cross attention
        # - FeedForward(d_model, d_ff, dropout) -> feed-forward network
        # - LayerNorm(d_model) -> layer norm (need 3)
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
            x: [batch, tgt_seq, d_model]
            encoder_output: [batch, src_seq, d_model]
            src_mask: Padding mask for source
            tgt_mask: Combined causal + padding mask for target
        """
        # TODO: Implement pre-norm decoder block with 3 sublayers
        # API hints:
        # - Sublayer 1: x = x + dropout(self_attn(norm1(x), norm1(x), norm1(x), tgt_mask))
        # - Sublayer 2: x = x + dropout(cross_attn(norm2(x), encoder_output, encoder_output, src_mask))
        # - Sublayer 3: x = x + dropout(ffn(norm3(x)))
        # - self.self_attn(q, k, v, mask) returns (output, attn_weights)
        return None


class TransformerDecoder(nn.Module):
    """Stack of decoder blocks."""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff=None, dropout=0.0):
        super().__init__()
        
        # TODO: Create decoder layers and final norm
        # API hints:
        # - nn.ModuleList([...]) -> list of modules
        # - DecoderBlock(d_model, num_heads, d_ff, dropout) -> decoder block
        # - LayerNorm(d_model) -> final layer norm
        self.layers = None
        self.final_norm = None
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # TODO: Pass through all layers, apply final norm
        # API hints:
        # - for layer in self.layers: x = layer(x, encoder_output, src_mask, tgt_mask)
        # - self.final_norm(x) -> final normalization
        return None


# ============================================================================
# Exercise 5: Full Transformer Model
# ============================================================================

class Transformer(nn.Module):
    """
    Full Transformer for sequence-to-sequence tasks.
    
    Architecture:
        Source -> Encoder -> encoder_output
        Target -> Decoder(encoder_output) -> Linear -> logits
    """
    
    def __init__(self, 
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 max_len=5000,
                 padding_idx=0):
        super().__init__()
        
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # TODO: Create source/target embeddings, encoder, decoder, output projection
        # API hints:
        # - TransformerEmbedding(vocab_size, d_model, max_len, dropout, padding_idx) -> embeddings
        # - TransformerEncoder(d_model, num_heads, num_encoder_layers, d_ff, dropout) -> encoder
        # - TransformerDecoder(d_model, num_heads, num_decoder_layers, d_ff, dropout) -> decoder
        # - nn.Linear(d_model, tgt_vocab_size) -> output projection
        self.src_embedding = None
        self.tgt_embedding = None
        self.encoder = None
        self.decoder = None
        self.output_projection = None
    
    def create_src_mask(self, src):
        """Create padding mask for source sequence."""
        # TODO: Create mask where padding tokens are 0
        # API hints:
        # - (src != self.padding_idx) -> boolean mask
        # - tensor.unsqueeze(1).unsqueeze(2) -> shape [batch, 1, 1, src_seq]
        return None
    
    def create_tgt_mask(self, tgt):
        """Create combined causal and padding mask for target."""
        # TODO: Create combined causal and padding mask
        # API hints:
        # - create_causal_mask(tgt_len, device) -> causal mask [tgt_len, tgt_len]
        # - tensor.unsqueeze(0).unsqueeze(0) -> add batch/head dims
        # - (tgt != self.padding_idx).unsqueeze(1).unsqueeze(2) -> padding mask
        # - causal_mask * padding_mask -> combine masks (element-wise AND)
        return None
    
    def encode(self, src, src_mask=None):
        """Encode source sequence."""
        # TODO: Embed source and pass through encoder
        # API hints:
        # - self.src_embedding(src) -> source embeddings
        # - self.encoder(src_emb, src_mask) -> encoder output
        return None
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence given encoder output."""
        # TODO: Embed target and pass through decoder
        # API hints:
        # - self.tgt_embedding(tgt) -> target embeddings
        # - self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask) -> decoder output
        return None
    
    def forward(self, src, tgt):
        """
        Full forward pass.
        
        Args:
            src: [batch, src_seq] source token indices
            tgt: [batch, tgt_seq] target token indices (shifted right)
        
        Returns:
            logits: [batch, tgt_seq, tgt_vocab_size]
        """
        # TODO: Create masks, encode, decode, project to vocabulary
        # API hints:
        # - self.create_src_mask(src), self.create_tgt_mask(tgt) -> masks
        # - self.encode(src, src_mask) -> encoder output
        # - self.decode(tgt, encoder_output, src_mask, tgt_mask) -> decoder output
        # - self.output_projection(decoder_output) -> logits
        return None


# ============================================================================
# Exercise 6: Autoregressive Generation
# ============================================================================

def greedy_decode(model, src, max_len, start_token, end_token):
    """
    Greedy decoding for generation.
    
    Args:
        model: Trained Transformer model
        src: [batch, src_seq] source sequence
        max_len: Maximum generation length
        start_token: Start-of-sequence token id
        end_token: End-of-sequence token id
    
    Returns:
        generated: [batch, generated_len] generated token indices
    """
    model.eval()
    batch_size = src.shape[0]
    device = src.device
    
    # Encode source once
    src_mask = model.create_src_mask(src)
    encoder_output = model.encode(src, src_mask)
    
    # Start with SOS token
    generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
    
    # TODO: Implement autoregressive generation loop
    for _ in range(max_len - 1):
        # Create target mask for current sequence
        tgt_mask = model.create_tgt_mask(generated)
        
        # Decode
        decoder_output = model.decode(generated, encoder_output, src_mask, tgt_mask)
        
        if decoder_output is None:
            break
        
        # Get logits for last position
        if model.output_projection is not None:
            logits = model.output_projection(decoder_output[:, -1:])
            # Greedy: take argmax
            next_token = logits.argmax(dim=-1)
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS (simplified - doesn't handle batch properly)
            if (next_token == end_token).all():
                break
    
    return generated


# ============================================================================
# Exercise 7: Compare with PyTorch's nn.Transformer
# ============================================================================

def compare_with_pytorch_transformer():
    """
    Compare our implementation with PyTorch's nn.Transformer.
    
    Note: PyTorch's Transformer has some differences:
    - Different mask format (additive vs multiplicative)
    - Different default architecture choices
    """
    d_model = 64
    num_heads = 4
    num_layers = 2
    batch_size = 2
    src_len, tgt_len = 10, 8
    vocab_size = 100
    
    # Our model
    our_model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        d_ff=d_model * 4,
        dropout=0.0
    )
    
    # Test input
    src = torch.randint(1, vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, vocab_size, (batch_size, tgt_len))
    
    # Forward pass
    output = our_model(src, tgt)
    
    if output is not None:
        return True, f"Output shape: {output.shape}"
    return False, "Model not implemented"


# ============================================================================
# Demonstrating Training vs Inference
# ============================================================================

def demonstrate_training_mode():
    """Show how training uses teacher forcing."""
    print("Training Mode (Teacher Forcing):")
    print("================================")
    print("Input:  <SOS> I love coding")
    print("Target: I love coding <EOS>")
    print("\nAt each step, decoder sees ground truth previous tokens:")
    print("Step 1: <SOS> -> predict 'I'")
    print("Step 2: <SOS> I -> predict 'love'")
    print("Step 3: <SOS> I love -> predict 'coding'")
    print("Step 4: <SOS> I love coding -> predict '<EOS>'")


def demonstrate_inference_mode():
    """Show how inference is autoregressive."""
    print("\nInference Mode (Autoregressive):")
    print("================================")
    print("Start with: <SOS>")
    print("Step 1: <SOS> -> predict 'The' -> sequence: <SOS> The")
    print("Step 2: <SOS> The -> predict 'cat' -> sequence: <SOS> The cat")
    print("Step 3: <SOS> The cat -> predict 'sat' -> sequence: <SOS> The cat sat")
    print("Continue until <EOS> or max_length...")


if __name__ == "__main__":
    print("Day 19: Full Transformer Architecture")
    print("=" * 50)
    
    # Demo: Full model
    print("\nDemo: Full Transformer Model")
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    batch, src_len, tgt_len = 2, 10, 8
    src = torch.randint(1, 1000, (batch, src_len))
    tgt = torch.randint(1, 1000, (batch, tgt_len))
    
    output = model(src, tgt)
    if output is not None:
        print(f"Source shape: {src.shape}")
        print(f"Target shape: {tgt.shape}")
        print(f"Output shape: {output.shape}")
        print(f"  (batch, tgt_seq, vocab_size)")
    
    # Training vs Inference
    print("\n")
    demonstrate_training_mode()
    demonstrate_inference_mode()
    
    # Parameter count
    if model.encoder is not None:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal parameters: {total_params:,}")
    
    print("\nRun test_day19.py to verify your implementations!")
