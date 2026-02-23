"""
Day 28: MLA Full Implementation
===============================
Estimated time: 1-2 hours
Prerequisites: Day 26 (MLA basics), Day 27 (KV compression)

Learning objectives:
- Combine all MLA components into a complete attention module
- Implement proper KV caching with compressed latents
- Handle incremental decoding efficiently
- Build a complete MLA transformer block
- Compare inference efficiency with standard attention

Key Concepts:
-------------
Complete MLA Architecture:
    DeepSeek-V2 MLA combines all components we've built:
    
    1. Query Processing:
       c_q = RMSNorm(x @ W_down_q)        # Compress query
       q_content = c_q @ W_up_q           # Query content
       q_rope = RoPE(x @ W_q_rope)        # Query position
    
    2. KV Processing:
       c_kv = RMSNorm(x @ W_down_kv)      # Compress KV (CACHED)
       k_content = c_kv @ W_up_k          # Key content
       k_rope = RoPE(x @ W_k_rope)        # Key position (recomputed)
       v = c_kv @ W_up_v                  # Value
    
    3. Attention:
       scores = (q_content @ k_content.T + q_rope @ k_rope.T) / sqrt(d)
       output = softmax(scores) @ v

Caching Strategy:
    During inference, we cache:
    - c_kv: Compressed KV latent (small!)
    - k_rope: RoPE key component (for each position)
    
    NOT cached (recomputed from c_kv):
    - k_content, v (these are derived from cached c_kv)
    
    Trade-off: More compute at inference, but much less memory

Memory Savings Example:
    7B model, 32K context, 32 layers:
    - Standard: ~17 GB KV cache
    - MLA (512 latent): ~1 GB cache (+ ~0.25 GB for k_rope)
    
    This enables much longer contexts on limited hardware!

Generation Flow:
    Prefill (process prompt):
        1. Compute c_kv for all positions
        2. Compute k_rope for all positions
        3. Cache both
    
    Decode (generate tokens):
        1. For new token: compute q, c_kv_new, k_rope_new
        2. Reconstruct full K, V from all cached c_kv
        3. Compute attention
        4. Append new c_kv, k_rope to cache
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


# Import components from Day 27
try:
    from day27 import RMSNorm, precompute_freqs_cis, apply_rotary_emb
except ImportError:
    # Fallback implementations for testing
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        
        def forward(self, x):
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight
    
    def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)
    
    def apply_rotary_emb(x, freqs_cis):
        x_shape = x.shape
        x = x.view(*x_shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x.float())
        seq_len = x_complex.shape[-2] if x_complex.dim() > 2 else x_complex.shape[0]
        freqs = freqs_cis[:seq_len]
        while freqs.dim() < x_complex.dim():
            freqs = freqs.unsqueeze(0)
        x_rotated = x_complex * freqs
        x_out = torch.view_as_real(x_rotated)
        return x_out.view(*x_shape).type_as(x)


# ============================================================================
# MLA Configuration
# ============================================================================

@dataclass
class MLAConfig:
    """Configuration for Multi-head Latent Attention."""
    d_model: int = 2048
    num_heads: int = 16
    head_dim: int = 128
    
    # Latent dimensions (compression)
    d_kv_latent: int = 512    # KV compression dimension
    d_q_latent: int = 384     # Query compression dimension (optional)
    
    # RoPE dimensions
    rope_dim: int = 64        # RoPE dimension per head
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    
    # Other
    dropout: float = 0.0
    use_q_compression: bool = True  # Whether to compress queries too


# ============================================================================
# Exercise 1: MLA KV Cache
# ============================================================================

class MLAKVCache:
    """
    Cache for MLA that stores compressed latents and RoPE keys.
    """
    
    def __init__(self, batch_size: int, max_seq_len: int, 
                 d_kv_latent: int, num_heads: int, rope_dim: int,
                 device: torch.device = None, dtype: torch.dtype = None):
        """
        Args:
            batch_size: Batch size
            max_seq_len: Maximum sequence length
            d_kv_latent: KV latent dimension
            num_heads: Number of attention heads  
            rope_dim: RoPE dimension per head
            device: Device to allocate on
            dtype: Data type
        """
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.d_kv_latent = d_kv_latent
        self.num_heads = num_heads
        self.rope_dim = rope_dim
        self.device = device
        self.dtype = dtype
        
        # Current length of cached sequence
        self.seq_len = 0
        
        # TODO: Initialize cache tensors
        # HINT:
        #   # Compressed KV latent cache
        #   self.c_kv = torch.zeros(
        #       batch_size, max_seq_len, d_kv_latent,
        #       device=device, dtype=dtype
        #   )
        #   
        #   # RoPE key cache
        #   self.k_rope = torch.zeros(
        #       batch_size, max_seq_len, num_heads, rope_dim,
        #       device=device, dtype=dtype
        #   )
        self.c_kv = None   # Replace
        self.k_rope = None # Replace
    
    def update(self, c_kv_new: torch.Tensor, k_rope_new: torch.Tensor) -> int:
        """
        Append new cached values.
        
        Args:
            c_kv_new: New compressed latents (batch, new_len, d_kv_latent)
            k_rope_new: New RoPE keys (batch, new_len, num_heads, rope_dim)
        
        Returns:
            New sequence length
        
        TODO: Append to cache
        HINT:
            new_len = c_kv_new.shape[1]
            
            self.c_kv[:, self.seq_len:self.seq_len + new_len] = c_kv_new
            self.k_rope[:, self.seq_len:self.seq_len + new_len] = k_rope_new
            
            self.seq_len += new_len
            return self.seq_len
        """
        return 0  # Replace
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached values up to current sequence length.
        
        Returns:
            c_kv: Cached latents (batch, seq_len, d_kv_latent)
            k_rope: Cached RoPE keys (batch, seq_len, num_heads, rope_dim)
        
        TODO: Return cached values
        HINT:
            return self.c_kv[:, :self.seq_len], self.k_rope[:, :self.seq_len]
        """
        return None, None  # Replace
    
    def reset(self):
        """Reset cache to empty state."""
        self.seq_len = 0


# ============================================================================
# Exercise 2: Complete MLA Module
# ============================================================================

class MultiheadLatentAttention(nn.Module):
    """
    Complete Multi-head Latent Attention implementation.
    
    Combines query compression, KV compression, and decoupled RoPE.
    """
    
    def __init__(self, config: MLAConfig):
        super().__init__()
        
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_kv = config.num_heads * config.head_dim
        self.rope_dim = config.rope_dim
        
        # Scale factor for attention
        total_dim = config.head_dim + config.rope_dim
        self.scale = total_dim ** -0.5
        
        # TODO: Initialize all projection layers
        # HINT:
        #   # Query projections
        #   if config.use_q_compression:
        #       self.q_down = nn.Linear(config.d_model, config.d_q_latent, bias=False)
        #       self.q_norm = RMSNorm(config.d_q_latent)
        #       self.q_up = nn.Linear(config.d_q_latent, self.d_kv, bias=False)
        #   else:
        #       self.q_proj = nn.Linear(config.d_model, self.d_kv, bias=False)
        #   
        #   self.q_rope_proj = nn.Linear(config.d_model, config.num_heads * config.rope_dim, bias=False)
        #   
        #   # KV projections
        #   self.kv_down = nn.Linear(config.d_model, config.d_kv_latent, bias=False)
        #   self.kv_norm = RMSNorm(config.d_kv_latent)
        #   self.k_up = nn.Linear(config.d_kv_latent, self.d_kv, bias=False)
        #   self.v_up = nn.Linear(config.d_kv_latent, self.d_kv, bias=False)
        #   self.k_rope_proj = nn.Linear(config.d_model, config.num_heads * config.rope_dim, bias=False)
        #   
        #   # Output projection
        #   self.out_proj = nn.Linear(self.d_kv, config.d_model, bias=False)
        #   
        #   # Dropout
        #   self.dropout = nn.Dropout(config.dropout)
        #   
        #   # RoPE frequencies
        #   self.register_buffer('freqs_cis', 
        #       precompute_freqs_cis(config.rope_dim, config.max_seq_len, config.rope_theta))
        
        # Query projections
        self.q_down = None       # Replace (if using compression)
        self.q_norm = None       # Replace (if using compression)
        self.q_up = None         # Replace (if using compression)
        self.q_proj = None       # Replace (if not using compression)
        self.q_rope_proj = None  # Replace
        
        # KV projections
        self.kv_down = None      # Replace
        self.kv_norm = None      # Replace
        self.k_up = None         # Replace
        self.v_up = None         # Replace
        self.k_rope_proj = None  # Replace
        
        # Output
        self.out_proj = None     # Replace
        self.dropout = None      # Replace
    
    def compute_q(self, x: torch.Tensor, start_pos: int = 0
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute query content and RoPE components.
        
        Args:
            x: Input (batch, seq_len, d_model)
            start_pos: Starting position for RoPE
        
        Returns:
            q_content: (batch, num_heads, seq_len, head_dim)
            q_rope: (batch, num_heads, seq_len, rope_dim)
        
        TODO: Compute query components
        HINT:
            batch, seq_len, _ = x.shape
            
            # Content query
            if self.config.use_q_compression:
                c_q = self.q_down(x)
                c_q = self.q_norm(c_q)
                q = self.q_up(c_q)
            else:
                q = self.q_proj(x)
            
            q_content = q.view(batch, seq_len, self.num_heads, self.head_dim)
            q_content = q_content.transpose(1, 2)  # (batch, heads, seq, head_dim)
            
            # RoPE query
            q_rope = self.q_rope_proj(x)
            q_rope = q_rope.view(batch, seq_len, self.num_heads, self.rope_dim)
            q_rope = apply_rotary_emb(q_rope, self.freqs_cis[start_pos:start_pos + seq_len])
            q_rope = q_rope.transpose(1, 2)  # (batch, heads, seq, rope_dim)
            
            return q_content, q_rope
        """
        batch, seq_len, _ = x.shape
        return (
            torch.zeros(batch, self.num_heads, seq_len, self.head_dim),
            torch.zeros(batch, self.num_heads, seq_len, self.rope_dim)
        )  # Replace
    
    def compute_kv(self, x: torch.Tensor, start_pos: int = 0
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute KV components (for caching and attention).
        
        Args:
            x: Input (batch, seq_len, d_model)
            start_pos: Starting position for RoPE
        
        Returns:
            c_kv: Compressed latent for caching (batch, seq_len, d_kv_latent)
            k_content: (batch, num_heads, seq_len, head_dim)
            k_rope: For caching (batch, seq_len, num_heads, rope_dim)
            v: (batch, num_heads, seq_len, head_dim)
        
        TODO: Compute KV components
        HINT:
            batch, seq_len, _ = x.shape
            
            # Compress to latent
            c_kv = self.kv_down(x)
            c_kv = self.kv_norm(c_kv)
            
            # Reconstruct K content and V
            k = self.k_up(c_kv)
            v = self.v_up(c_kv)
            
            k_content = k.view(batch, seq_len, self.num_heads, self.head_dim)
            k_content = k_content.transpose(1, 2)
            
            v = v.view(batch, seq_len, self.num_heads, self.head_dim)
            v = v.transpose(1, 2)
            
            # RoPE K (keep in seq-first format for caching)
            k_rope = self.k_rope_proj(x)
            k_rope = k_rope.view(batch, seq_len, self.num_heads, self.rope_dim)
            k_rope = apply_rotary_emb(k_rope, self.freqs_cis[start_pos:start_pos + seq_len])
            
            return c_kv, k_content, k_rope, v
        """
        batch, seq_len, _ = x.shape
        return (
            torch.zeros(batch, seq_len, self.config.d_kv_latent),
            torch.zeros(batch, self.num_heads, seq_len, self.head_dim),
            torch.zeros(batch, seq_len, self.num_heads, self.rope_dim),
            torch.zeros(batch, self.num_heads, seq_len, self.head_dim)
        )  # Replace
    
    def reconstruct_kv_from_cache(self, c_kv: torch.Tensor, k_rope: torch.Tensor
                                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstruct full K, V from cached latent.
        
        Args:
            c_kv: Cached compressed latent (batch, cache_len, d_kv_latent)
            k_rope: Cached RoPE keys (batch, cache_len, num_heads, rope_dim)
        
        Returns:
            k_content: (batch, num_heads, cache_len, head_dim)
            k_rope: (batch, num_heads, cache_len, rope_dim)
            v: (batch, num_heads, cache_len, head_dim)
        
        TODO: Reconstruct from cache
        HINT:
            batch, cache_len, _ = c_kv.shape
            
            k = self.k_up(c_kv)
            v = self.v_up(c_kv)
            
            k_content = k.view(batch, cache_len, self.num_heads, self.head_dim)
            k_content = k_content.transpose(1, 2)
            
            v = v.view(batch, cache_len, self.num_heads, self.head_dim)
            v = v.transpose(1, 2)
            
            k_rope_out = k_rope.transpose(1, 2)
            
            return k_content, k_rope_out, v
        """
        batch, cache_len, _ = c_kv.shape
        return (
            torch.zeros(batch, self.num_heads, cache_len, self.head_dim),
            torch.zeros(batch, self.num_heads, cache_len, self.rope_dim),
            torch.zeros(batch, self.num_heads, cache_len, self.head_dim)
        )  # Replace
    
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[MLAKVCache] = None,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full MLA forward pass with optional caching.
        
        Args:
            x: Input (batch, seq_len, d_model)
            cache: Optional KV cache
            start_pos: Position offset (for generation)
            mask: Optional attention mask (batch, 1, seq_q, seq_k)
        
        Returns:
            output: (batch, seq_len, d_model)
            attn_weights: (batch, num_heads, seq_len, total_seq_len)
        
        TODO: Implement full forward
        HINT:
            batch, seq_len, _ = x.shape
            
            # Compute Q
            q_content, q_rope = self.compute_q(x, start_pos)
            
            # Compute or retrieve KV
            if cache is None:
                # No cache - compute everything
                _, k_content, k_rope_raw, v = self.compute_kv(x, start_pos)
                k_rope = k_rope_raw.transpose(1, 2)
            else:
                # With cache
                c_kv_new, _, k_rope_new, _ = self.compute_kv(x, start_pos)
                cache.update(c_kv_new, k_rope_new)
                
                # Get all cached KV
                c_kv_all, k_rope_all = cache.get()
                k_content, k_rope, v = self.reconstruct_kv_from_cache(c_kv_all, k_rope_all)
            
            # Attention scores: content + position
            scores_content = torch.matmul(q_content, k_content.transpose(-2, -1))
            scores_rope = torch.matmul(q_rope, k_rope.transpose(-2, -1))
            scores = (scores_content + scores_rope) * self.scale
            
            # Apply mask
            if mask is not None:
                scores = scores + mask
            
            # Softmax and apply to V
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)
            output = output.transpose(1, 2).reshape(batch, seq_len, self.d_kv)
            output = self.out_proj(output)
            
            return output, attn_weights
        """
        batch, seq_len, _ = x.shape
        total_len = seq_len if cache is None else seq_len + (cache.seq_len if cache.c_kv is not None else 0)
        return (
            torch.zeros_like(x),
            torch.zeros(batch, self.num_heads, seq_len, total_len)
        )  # Replace


# ============================================================================
# Exercise 3: MLA Transformer Block
# ============================================================================

class MLATransformerBlock(nn.Module):
    """
    Transformer block using MLA instead of standard attention.
    """
    
    def __init__(self, config: MLAConfig, ffn_hidden_mult: float = 4.0):
        super().__init__()
        
        self.config = config
        
        # TODO: Initialize block components
        # HINT:
        #   # Attention with pre-norm
        #   self.attn_norm = RMSNorm(config.d_model)
        #   self.attn = MultiheadLatentAttention(config)
        #   
        #   # FFN with pre-norm
        #   self.ffn_norm = RMSNorm(config.d_model)
        #   ffn_hidden = int(config.d_model * ffn_hidden_mult)
        #   self.ffn = nn.Sequential(
        #       nn.Linear(config.d_model, ffn_hidden, bias=False),
        #       nn.GELU(),
        #       nn.Linear(ffn_hidden, config.d_model, bias=False),
        #   )
        self.attn_norm = None  # Replace
        self.attn = None       # Replace
        self.ffn_norm = None   # Replace
        self.ffn = None        # Replace
    
    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[MLAKVCache] = None,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer block.
        
        TODO: Implement transformer block
        HINT:
            # Attention with residual
            normed = self.attn_norm(x)
            attn_out, attn_weights = self.attn(normed, cache, start_pos, mask)
            x = x + attn_out
            
            # FFN with residual
            normed = self.ffn_norm(x)
            x = x + self.ffn(normed)
            
            return x, attn_weights
        """
        batch, seq_len, _ = x.shape
        return x, torch.zeros(batch, self.config.num_heads, seq_len, seq_len)  # Replace


# ============================================================================
# Exercise 4: MLA Model (Stack of Blocks)
# ============================================================================

class MLAModel(nn.Module):
    """
    Complete MLA model with multiple layers.
    """
    
    def __init__(self, config: MLAConfig, num_layers: int = 4):
        super().__init__()
        
        self.config = config
        self.num_layers = num_layers
        
        # TODO: Initialize layers
        # HINT:
        #   self.layers = nn.ModuleList([
        #       MLATransformerBlock(config) for _ in range(num_layers)
        #   ])
        #   self.final_norm = RMSNorm(config.d_model)
        self.layers = None      # Replace
        self.final_norm = None  # Replace
    
    def forward(
        self,
        x: torch.Tensor,
        caches: Optional[Dict[int, MLAKVCache]] = None,
        start_pos: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass through all layers.
        
        Args:
            x: Input (batch, seq_len, d_model)
            caches: Dict mapping layer index to cache
            start_pos: Position for RoPE
            mask: Attention mask
        
        Returns:
            output: Final hidden states
            all_attn_weights: Dict of attention weights per layer
        
        TODO: Implement forward
        HINT:
            all_attn_weights = {}
            
            for i, layer in enumerate(self.layers):
                cache = caches.get(i) if caches else None
                x, attn = layer(x, cache, start_pos, mask)
                all_attn_weights[i] = attn
            
            x = self.final_norm(x)
            return x, all_attn_weights
        """
        return x, {}  # Replace


# ============================================================================
# Exercise 5: Generation with MLA
# ============================================================================

def generate_with_mla(
    model: MLAModel,
    prompt_embeds: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, Dict[int, MLAKVCache]]:
    """
    Generate tokens using MLA with caching.
    
    Args:
        model: MLA model
        prompt_embeds: Embedded prompt (batch, prompt_len, d_model)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        all_hidden_states: Hidden states for all positions
        caches: Final cache state
    
    TODO: Implement generation loop
    HINT:
        batch, prompt_len, d_model = prompt_embeds.shape
        config = model.config
        
        # Initialize caches
        caches = {
            i: MLAKVCache(batch, prompt_len + max_new_tokens,
                          config.d_kv_latent, config.num_heads, config.rope_dim,
                          device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            for i in range(model.num_layers)
        }
        
        # Create causal mask for prefill
        prefill_mask = torch.triu(
            torch.ones(prompt_len, prompt_len, device=prompt_embeds.device) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # Prefill: process prompt
        hidden, _ = model(prompt_embeds, caches, start_pos=0, mask=prefill_mask)
        all_hidden = [hidden]
        
        # Decode: generate token by token
        current_pos = prompt_len
        current_hidden = hidden[:, -1:, :]  # Last token's hidden state
        
        for _ in range(max_new_tokens):
            # Forward single token (no mask needed - attends to all previous)
            next_hidden, _ = model(current_hidden, caches, start_pos=current_pos)
            
            all_hidden.append(next_hidden)
            current_hidden = next_hidden
            current_pos += 1
        
        return torch.cat(all_hidden, dim=1), caches
    """
    return prompt_embeds, {}  # Replace


# ============================================================================
# Exercise 6: Memory Comparison
# ============================================================================

def compare_memory_usage(config: MLAConfig, seq_len: int, num_layers: int) -> dict:
    """
    Compare memory usage between standard attention and MLA.
    
    Args:
        config: MLA configuration
        seq_len: Sequence length
        num_layers: Number of layers
    
    Returns:
        Dictionary with memory statistics
    
    TODO: Compute memory comparison
    HINT:
        d_kv = config.num_heads * config.head_dim
        
        # Standard attention: cache K and V
        std_cache_per_layer = 2 * seq_len * d_kv  # K and V
        std_total = num_layers * std_cache_per_layer
        
        # MLA: cache compressed latent + RoPE keys
        mla_latent_per_layer = seq_len * config.d_kv_latent
        mla_rope_per_layer = seq_len * config.num_heads * config.rope_dim
        mla_cache_per_layer = mla_latent_per_layer + mla_rope_per_layer
        mla_total = num_layers * mla_cache_per_layer
        
        # Compute at inference (reconstruction)
        # MLA needs to reconstruct K, V from latent - adds compute
        mla_compute_per_layer = seq_len * (config.d_kv_latent * d_kv * 2)  # K and V up projection
        
        return {
            'standard_cache_elements': std_total,
            'mla_cache_elements': mla_total,
            'memory_reduction': std_total / mla_total,
            'mla_recompute_flops_per_layer': mla_compute_per_layer
        }
    """
    return {
        'standard_cache_elements': 0,
        'mla_cache_elements': 0,
        'memory_reduction': 0.0,
        'mla_recompute_flops_per_layer': 0
    }  # Replace


if __name__ == "__main__":
    print("Day 28: MLA Full Implementation")
    print("=" * 60)
    
    # Configuration
    config = MLAConfig(
        d_model=512,
        num_heads=8,
        head_dim=64,
        d_kv_latent=128,
        d_q_latent=96,
        rope_dim=32,
        max_seq_len=256
    )
    
    print(f"\nConfiguration:")
    for field in ['d_model', 'num_heads', 'head_dim', 'd_kv_latent', 'd_q_latent', 'rope_dim']:
        print(f"  {field}: {getattr(config, field)}")
    
    batch_size = 2
    seq_len = 16
    
    # Test MLA KV Cache
    print("\n" + "=" * 60)
    print("Testing MLAKVCache")
    print("=" * 60)
    
    cache = MLAKVCache(batch_size, 64, config.d_kv_latent, config.num_heads, 
                       config.rope_dim, device='cpu')
    if cache.c_kv is not None:
        c_kv = torch.randn(batch_size, seq_len, config.d_kv_latent)
        k_rope = torch.randn(batch_size, seq_len, config.num_heads, config.rope_dim)
        new_len = cache.update(c_kv, k_rope)
        print(f"After update: cache length = {new_len}")
        cached_c, cached_k = cache.get()
        print(f"Cached c_kv: {cached_c.shape}")
        print(f"Cached k_rope: {cached_k.shape}")
    
    # Test complete MLA
    print("\n" + "=" * 60)
    print("Testing MultiheadLatentAttention")
    print("=" * 60)
    
    mla = MultiheadLatentAttention(config)
    if mla.kv_down is not None:
        x = torch.randn(batch_size, seq_len, config.d_model)
        output, attn = mla(x)
        print(f"Input: {x.shape}")
        print(f"Output: {output.shape}")
        print(f"Attention: {attn.shape}")
    
    # Test MLA Transformer Block
    print("\n" + "=" * 60)
    print("Testing MLATransformerBlock")
    print("=" * 60)
    
    block = MLATransformerBlock(config)
    if block.attn is not None:
        x = torch.randn(batch_size, seq_len, config.d_model)
        output, attn = block(x)
        print(f"Block output: {output.shape}")
    
    # Test generation
    print("\n" + "=" * 60)
    print("Testing Generation with MLA")
    print("=" * 60)
    
    model = MLAModel(config, num_layers=2)
    if model.layers is not None:
        prompt = torch.randn(batch_size, 8, config.d_model)
        all_hidden, caches = generate_with_mla(model, prompt, max_new_tokens=4)
        print(f"Prompt length: 8")
        print(f"Generated length: {all_hidden.shape[1]}")
    
    # Memory comparison
    print("\n" + "=" * 60)
    print("Memory Comparison (seq_len=8192, 32 layers)")
    print("=" * 60)
    
    mem = compare_memory_usage(config, 8192, 32)
    if mem['memory_reduction'] > 0:
        print(f"Memory reduction: {mem['memory_reduction']:.1f}x")
    
    print("\nRun test_day28.py to verify your implementations!")
