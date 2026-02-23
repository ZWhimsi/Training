"""
Day 20: GPT-style Decoder-Only Model
====================================
Estimated time: 1-2 hours
Prerequisites: Day 18 (decoder block), Day 19 (full transformer)

Learning objectives:
- Understand decoder-only architecture (GPT, LLaMA, etc.)
- Implement causal language modeling objective
- Build a complete GPT-style model
- Understand the difference from encoder-decoder architecture
- Implement key-value caching for efficient inference

Key Concepts:
-------------
Decoder-only models (like GPT) differ from encoder-decoder:

1. No Encoder:
   - Only uses decoder blocks
   - No cross-attention sublayer (since no encoder)
   
2. Causal Language Modeling:
   - Predicts next token given all previous tokens
   - Training objective: P(x_t | x_1, ..., x_{t-1})
   - Uses causal mask to prevent looking ahead

3. Architecture (per block):
   - Masked Self-Attention (causal)
   - Feed-Forward Network
   - Residual connections + LayerNorm

GPT Block (Pre-Norm):
    x = x + MaskedSelfAttention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

Key-Value Caching:
- During autoregressive generation, we can cache K and V
- Only compute Q for the new token
- Dramatically speeds up generation

Modern Variants:
- GPT-2/3: Standard pre-norm
- LLaMA: RMSNorm, RoPE positional encoding, SwiGLU FFN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# Exercise 1: RMS Layer Normalization (used in LLaMA)
# ============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (used in LLaMA, Gemma, etc.)
    
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
    
    Simpler than LayerNorm: no mean subtraction, no beta
    """
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        
        # TODO: Create learnable scale parameter (no shift in RMSNorm)
        self.weight = None  # Replace: nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq, d_model]
        Returns:
            [batch, seq, d_model]
        """
        # TODO: Compute RMS: sqrt(mean(x^2))
        # HINT: rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        rms = None  # Replace
        
        # TODO: Normalize and scale
        # HINT: return x / rms * self.weight
        return None  # Replace


# ============================================================================
# Exercise 2: Rotary Position Embedding (RoPE) - used in LLaMA, GPT-NeoX
# ============================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Instead of adding positional information, RoPE rotates the 
    query and key vectors based on position.
    
    For position m, dimension pairs (i, i+1), rotation angle = m * theta_i
    where theta_i = 10000^(-2i/d)
    """
    
    def __init__(self, d_model, max_len=5000, base=10000):
        super().__init__()
        
        self.d_model = d_model
        
        # TODO: Compute inverse frequencies
        # inv_freq = 1 / (base ** (arange(0, d_model, 2) / d_model))
        inv_freq = None  # Replace
        
        if inv_freq is not None:
            self.register_buffer('inv_freq', inv_freq)
        
        # TODO: Precompute cos and sin for positions up to max_len
        # position = arange(max_len)
        # freqs = outer(position, inv_freq) -> [max_len, d_model/2]
        # cos_cached = cos(freqs)
        # sin_cached = sin(freqs)
        self.register_buffer('cos_cached', torch.zeros(max_len, d_model // 2))
        self.register_buffer('sin_cached', torch.zeros(max_len, d_model // 2))
    
    def forward(self, q, k, positions=None):
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            q: [batch, heads, seq, d_k]
            k: [batch, heads, seq, d_k]
            positions: Optional [seq] position indices
        
        Returns:
            q_rotated, k_rotated with same shapes
        """
        seq_len = q.shape[2]
        
        if positions is None:
            positions = torch.arange(seq_len, device=q.device)
        
        # Get cos and sin for these positions
        cos = self.cos_cached[positions]  # [seq, d_k/2]
        sin = self.sin_cached[positions]  # [seq, d_k/2]
        
        # TODO: Apply rotation
        # Split q and k into pairs, rotate each pair
        # This is the simplified version; full implementation is more complex
        
        return q, k  # Return unmodified for now (placeholder)


# ============================================================================
# Exercise 3: GPT Block (Decoder-Only Block)
# ============================================================================

class GPTBlock(nn.Module):
    """
    GPT-style decoder block (no cross-attention).
    
    Structure (Pre-Norm):
        x = x + MaskedSelfAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.0, use_rms_norm=False):
        super().__init__()
        
        d_ff = d_ff or d_model * 4
        
        # TODO: Causal self-attention
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = None  # Replace: nn.Linear(d_model, d_model)
        self.W_k = None  # Replace: nn.Linear(d_model, d_model)
        self.W_v = None  # Replace: nn.Linear(d_model, d_model)
        self.W_o = None  # Replace: nn.Linear(d_model, d_model)
        
        # TODO: FFN
        self.ffn_linear1 = None  # Replace: nn.Linear(d_model, d_ff)
        self.ffn_linear2 = None  # Replace: nn.Linear(d_ff, d_model)
        
        # TODO: Layer norms (RMSNorm or standard LayerNorm)
        if use_rms_norm:
            self.norm1 = None  # Replace: RMSNorm(d_model)
            self.norm2 = None  # Replace: RMSNorm(d_model)
        else:
            self.norm1 = None  # Replace: nn.LayerNorm(d_model)
            self.norm2 = None  # Replace: nn.LayerNorm(d_model)
        
        self.dropout = None  # Replace: nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, x, causal_mask=None, kv_cache=None):
        """
        Args:
            x: [batch, seq, d_model]
            causal_mask: Optional [seq, seq] causal mask
            kv_cache: Optional tuple (cached_k, cached_v) for generation
        
        Returns:
            output: [batch, seq, d_model]
            new_kv_cache: Updated (k, v) cache
        """
        batch_size, seq_len, _ = x.shape
        
        # ===== Sublayer 1: Causal Self-Attention =====
        residual = x
        x = self.norm1(x) if self.norm1 else x
        
        # TODO: Compute Q, K, V
        Q = self.W_q(x) if self.W_q else None
        K = self.W_k(x) if self.W_k else None
        V = self.W_v(x) if self.W_v else None
        
        if Q is None:
            return x, None
        
        # TODO: Handle KV cache for efficient generation
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            K = torch.cat([cached_k, K], dim=1)
            V = torch.cat([cached_v, V], dim=1)
        
        new_kv_cache = (K, V)
        
        # TODO: Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # TODO: Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # TODO: Apply causal mask
        if causal_mask is not None:
            # Expand mask for batch and heads if needed
            if causal_mask.dim() == 2:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            # Handle case where K is longer due to cache
            k_len = K.shape[2]
            q_len = Q.shape[2]
            if causal_mask.shape[-1] != k_len or causal_mask.shape[-2] != q_len:
                # Create appropriate mask for cached scenario
                causal_mask = torch.tril(torch.ones(q_len, k_len, device=x.device))
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # TODO: Softmax and apply attention
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights) if self.dropout else attn_weights
        attn_output = torch.matmul(attn_weights, V)
        
        # TODO: Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.W_o(attn_output)
        attn_output = self.dropout(attn_output) if self.dropout else attn_output
        
        x = residual + attn_output
        
        # ===== Sublayer 2: FFN =====
        residual = x
        x = self.norm2(x) if self.norm2 else x
        
        # TODO: FFN forward
        if self.ffn_linear1 is not None and self.ffn_linear2 is not None:
            ffn_out = self.ffn_linear1(x)
            ffn_out = F.gelu(ffn_out)
            ffn_out = self.dropout(ffn_out) if self.dropout else ffn_out
            ffn_out = self.ffn_linear2(ffn_out)
            ffn_out = self.dropout(ffn_out) if self.dropout else ffn_out
            x = residual + ffn_out
        
        return x, new_kv_cache


# ============================================================================
# Exercise 4: Complete GPT Model
# ============================================================================

class GPT(nn.Module):
    """
    GPT-style decoder-only language model.
    
    Architecture:
        tokens -> Embedding + PosEmb -> N x GPTBlock -> LayerNorm -> Linear -> logits
    """
    
    def __init__(self, 
                 vocab_size,
                 d_model=768,
                 num_heads=12,
                 num_layers=12,
                 d_ff=None,
                 max_len=1024,
                 dropout=0.1,
                 use_rms_norm=False):
        super().__init__()
        
        self.d_model = d_model
        
        # TODO: Token embedding
        self.token_embedding = None  # Replace: nn.Embedding(vocab_size, d_model)
        
        # TODO: Position embedding (learned, not sinusoidal)
        self.position_embedding = None  # Replace: nn.Embedding(max_len, d_model)
        
        self.dropout = None  # Replace: nn.Dropout(dropout)
        
        # TODO: GPT blocks
        self.blocks = None  # Replace: nn.ModuleList([GPTBlock(d_model, num_heads, d_ff, dropout, use_rms_norm) for _ in range(num_layers)])
        
        # TODO: Final layer norm
        if use_rms_norm:
            self.final_norm = None  # Replace: RMSNorm(d_model)
        else:
            self.final_norm = None  # Replace: nn.LayerNorm(d_model)
        
        # TODO: Output projection to vocabulary (often tied with token_embedding)
        self.lm_head = None  # Replace: nn.Linear(d_model, vocab_size, bias=False)
        
        self.max_len = max_len
    
    def forward(self, input_ids, use_cache=False, past_kv_cache=None):
        """
        Args:
            input_ids: [batch, seq] token indices
            use_cache: Whether to use/return KV cache
            past_kv_cache: List of (k, v) tuples, one per layer
        
        Returns:
            logits: [batch, seq, vocab_size]
            present_kv_cache: Updated cache if use_cache=True
        """
        batch_size, seq_len = input_ids.shape
        
        # Determine position indices
        if past_kv_cache is not None:
            # We're continuing generation, positions start after cached
            past_len = past_kv_cache[0][0].shape[1] if past_kv_cache[0] is not None else 0
            positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        else:
            positions = torch.arange(seq_len, device=input_ids.device)
        
        # TODO: Get embeddings
        if self.token_embedding is None:
            return None, None
        
        x = self.token_embedding(input_ids)
        
        if self.position_embedding is not None:
            x = x + self.position_embedding(positions).unsqueeze(0)
        
        x = self.dropout(x) if self.dropout else x
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        
        # TODO: Pass through blocks
        present_kv_cache = []
        if self.blocks is not None:
            for i, block in enumerate(self.blocks):
                layer_cache = past_kv_cache[i] if past_kv_cache else None
                x, new_cache = block(x, causal_mask, layer_cache)
                if use_cache:
                    present_kv_cache.append(new_cache)
        
        # TODO: Final norm and projection
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = None
        
        if use_cache:
            return logits, present_kv_cache
        return logits, None
    
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive text generation.
        
        Args:
            input_ids: [batch, seq] initial token indices
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (1.0 = neutral)
            top_k: If set, only sample from top-k tokens
        
        Returns:
            [batch, seq + max_new_tokens] generated sequence
        """
        self.eval()
        
        generated = input_ids.clone()
        past_kv_cache = None
        
        for _ in range(max_new_tokens):
            # Get logits for last token only (if using cache)
            if past_kv_cache is not None:
                current_input = generated[:, -1:]
            else:
                current_input = generated
            
            logits, past_kv_cache = self(current_input, use_cache=True, past_kv_cache=past_kv_cache)
            
            if logits is None:
                break
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # TODO: Apply top-k filtering if specified
            if top_k is not None:
                # HINT: Set logits below top-k threshold to -inf
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Truncate if exceeding max length
            if generated.shape[1] >= self.max_len:
                break
        
        return generated


# ============================================================================
# Exercise 5: Causal Language Modeling Loss
# ============================================================================

def compute_lm_loss(logits, targets, ignore_index=-100):
    """
    Compute causal language modeling loss.
    
    For input [x1, x2, x3, x4], targets are [x2, x3, x4, <end>]
    We predict the next token at each position.
    
    Args:
        logits: [batch, seq, vocab_size] model predictions
        targets: [batch, seq] target token indices
        ignore_index: Index to ignore in loss (e.g., padding)
    
    Returns:
        loss: Scalar cross-entropy loss
    """
    # TODO: Flatten logits and targets
    # logits: [batch * seq, vocab_size]
    # targets: [batch * seq]
    batch_size, seq_len, vocab_size = logits.shape
    
    logits_flat = None  # Replace: logits.view(-1, vocab_size)
    targets_flat = None  # Replace: targets.view(-1)
    
    # TODO: Compute cross-entropy loss
    # HINT: F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index)
    loss = None  # Replace
    
    return loss


def prepare_lm_batch(token_ids):
    """
    Prepare input and target for language modeling.
    
    Input: all tokens except last
    Target: all tokens except first (shifted by 1)
    
    Args:
        token_ids: [batch, seq] full sequence
    
    Returns:
        input_ids: [batch, seq-1]
        target_ids: [batch, seq-1]
    """
    # TODO: Create input (everything except last token)
    input_ids = None  # Replace: token_ids[:, :-1]
    
    # TODO: Create target (everything except first token)
    target_ids = None  # Replace: token_ids[:, 1:]
    
    return input_ids, target_ids


# ============================================================================
# Exercise 6: Model Configurations (GPT-2 sizes)
# ============================================================================

def get_gpt2_config(size='small'):
    """
    Get GPT-2 model configurations.
    
    GPT-2 sizes:
    - Small: 117M params (d=768, h=12, l=12)
    - Medium: 345M params (d=1024, h=16, l=24)
    - Large: 762M params (d=1280, h=20, l=36)
    - XL: 1.5B params (d=1600, h=25, l=48)
    """
    configs = {
        'small': {'d_model': 768, 'num_heads': 12, 'num_layers': 12},
        'medium': {'d_model': 1024, 'num_heads': 16, 'num_layers': 24},
        'large': {'d_model': 1280, 'num_heads': 20, 'num_layers': 36},
        'xl': {'d_model': 1600, 'num_heads': 25, 'num_layers': 48},
    }
    return configs.get(size, configs['small'])


# ============================================================================
# Comparing GPT vs Encoder-Decoder Transformer
# ============================================================================

def compare_architectures():
    """Compare decoder-only vs encoder-decoder architectures."""
    print("="*60)
    print("GPT (Decoder-Only) vs Encoder-Decoder Transformer")
    print("="*60)
    
    print("\nDecoder-Only (GPT, LLaMA, etc.):")
    print("  - No encoder, only decoder blocks")
    print("  - No cross-attention")
    print("  - Causal attention (can only see past)")
    print("  - Best for: text generation, completion")
    print("  - Single sequence input/output")
    
    print("\nEncoder-Decoder (T5, BART, etc.):")
    print("  - Encoder processes source sequence")
    print("  - Decoder generates target sequence")
    print("  - Cross-attention connects them")
    print("  - Best for: translation, summarization")
    print("  - Two sequence inputs (source, target)")
    
    print("\nBlock Comparison:")
    print("  GPT Block:      SelfAttn -> FFN")
    print("  Decoder Block:  SelfAttn -> CrossAttn -> FFN")


if __name__ == "__main__":
    print("Day 20: GPT-style Decoder-Only Model")
    print("=" * 50)
    
    # Demo: GPT model
    print("\nDemo: GPT Model")
    model = GPT(
        vocab_size=50257,  # GPT-2 vocab size
        d_model=256,       # Smaller for demo
        num_heads=4,
        num_layers=4,
        max_len=512
    )
    
    batch, seq = 2, 16
    input_ids = torch.randint(0, 50257, (batch, seq))
    
    logits, _ = model(input_ids)
    if logits is not None:
        print(f"Input shape: {input_ids.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"  (batch, seq, vocab_size)")
    
    # Demo: Generation
    print("\nDemo: Text Generation")
    if model.token_embedding is not None:
        prompt = torch.randint(0, 50257, (1, 5))  # Short prompt
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
        print(f"Prompt length: {prompt.shape[1]}")
        print(f"Generated length: {generated.shape[1]}")
    
    # Architecture comparison
    print("\n")
    compare_architectures()
    
    # Model sizes
    print("\n\nGPT-2 Model Sizes:")
    for size in ['small', 'medium', 'large', 'xl']:
        config = get_gpt2_config(size)
        print(f"  {size}: d_model={config['d_model']}, heads={config['num_heads']}, layers={config['num_layers']}")
    
    print("\nRun test_day20.py to verify your implementations!")
