"""
Day 23: KV Cache for Inference
==============================
Estimated time: 1-2 hours
Prerequisites: Day 22 (grouped query attention), Day 16 (multi-head attention)

Learning objectives:
- Understand why KV caching is essential for efficient autoregressive generation
- Implement KV cache storage and retrieval
- Build cached attention for single-token generation
- Understand memory-compute tradeoffs in inference
- Implement incremental generation with cache

Key Concepts:
-------------
Problem: Autoregressive Generation
    For each new token, standard attention recomputes attention over ALL previous tokens.
    Generating sequence of length N requires O(N²) attention computations.

Solution: KV Cache
    - Cache K and V projections from previous tokens
    - For new token, only compute its Q, K, V
    - Concatenate new K, V to cache
    - Compute attention using full cache
    - Result: O(N) attention per token, O(N²) total (unavoidable)

Cache Structure:
    K cache: (batch, num_kv_heads, cache_len, head_dim)
    V cache: (batch, num_kv_heads, cache_len, head_dim)

Memory Consideration:
    Cache size = 2 * num_layers * batch * num_kv_heads * max_seq_len * head_dim * 4 bytes
    For a 7B model with 32 layers, 32 heads, 4096 context: ~4GB per batch!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


# ============================================================================
# Exercise 1: KV Cache Data Structure
# ============================================================================

@dataclass
class KVCache:
    """
    Key-Value cache for a single attention layer.
    
    Stores cached keys and values, tracks current sequence length.
    """
    k_cache: torch.Tensor  # (batch, num_kv_heads, max_seq_len, head_dim)
    v_cache: torch.Tensor  # (batch, num_kv_heads, max_seq_len, head_dim)
    seq_len: int  # Current cached sequence length


def create_kv_cache(batch_size: int, num_kv_heads: int, max_seq_len: int, 
                    head_dim: int, device: torch.device = None) -> KVCache:
    """
    Create an empty KV cache.
    
    Args:
        batch_size: Batch size
        num_kv_heads: Number of KV heads (may be fewer than query heads in GQA)
        max_seq_len: Maximum sequence length to cache
        head_dim: Dimension per head
        device: Device to create tensors on
    
    Returns:
        KVCache with pre-allocated zero tensors
    
    TODO: Create the cache tensors
    HINT:
        k_cache = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device)
        v_cache = torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device)
        return KVCache(k_cache=k_cache, v_cache=v_cache, seq_len=0)
    """
    return KVCache(
        k_cache=torch.zeros(1),
        v_cache=torch.zeros(1),
        seq_len=0
    )  # Replace


def update_kv_cache(cache: KVCache, new_k: torch.Tensor, 
                    new_v: torch.Tensor) -> KVCache:
    """
    Update cache with new key-value pairs.
    
    Args:
        cache: Existing KV cache
        new_k: New keys (batch, num_kv_heads, new_seq_len, head_dim)
        new_v: New values (batch, num_kv_heads, new_seq_len, head_dim)
    
    Returns:
        Updated cache with new entries appended
    
    TODO: Append new K, V to cache
    HINT:
        new_seq_len = new_k.size(2)
        start_pos = cache.seq_len
        end_pos = start_pos + new_seq_len
        
        # Write new values to cache at correct positions
        cache.k_cache[:, :, start_pos:end_pos, :] = new_k
        cache.v_cache[:, :, start_pos:end_pos, :] = new_v
        cache.seq_len = end_pos
        
        return cache
    """
    return cache  # Replace


def get_cached_kv(cache: KVCache) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve valid (non-padding) entries from cache.
    
    Args:
        cache: KV cache
    
    Returns:
        (k, v) tensors of shape (batch, num_kv_heads, seq_len, head_dim)
    
    TODO: Slice cache to valid length
    HINT:
        k = cache.k_cache[:, :, :cache.seq_len, :]
        v = cache.v_cache[:, :, :cache.seq_len, :]
        return k, v
    """
    return cache.k_cache, cache.v_cache  # Replace


# ============================================================================
# Exercise 2: Cached Attention
# ============================================================================

class CachedAttention(nn.Module):
    """
    Attention layer with KV caching support.
    
    Supports two modes:
    1. Prefill: Process full prompt, populate cache
    2. Decode: Process one token at a time, use cache
    """
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int = None):
        super().__init__()
        
        num_kv_heads = num_kv_heads or num_heads
        
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5
        
        # TODO: Create projection layers
        # HINT:
        #   self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        #   self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        #   self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        #   self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.W_q = None  # Replace
        self.W_k = None  # Replace
        self.W_v = None  # Replace
        self.W_o = None  # Replace
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads."""
        if self.num_kv_groups == 1:
            return x
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x = x.unsqueeze(2).expand(batch, num_kv_heads, self.num_kv_groups, seq_len, head_dim)
        return x.reshape(batch, self.num_heads, seq_len, head_dim)
    
    def forward(self, x: torch.Tensor, 
                cache: Optional[KVCache] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, KVCache]:
        """
        Forward pass with optional KV cache.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
               During decode, seq_len=1 (single new token)
            cache: Optional existing cache. If None, creates new cache.
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
            cache: Updated KV cache
        
        TODO: Implement cached attention
        HINT:
            batch, seq_len, _ = x.shape
            
            # Project current input to Q, K, V
            q = self.W_q(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.W_k(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.W_v(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            # Update cache
            if cache is not None:
                cache = update_kv_cache(cache, k, v)
                k_full, v_full = get_cached_kv(cache)
            else:
                # Create new cache
                cache = KVCache(k_cache=k, v_cache=v, seq_len=seq_len)
                k_full, v_full = k, v
            
            # Repeat KV for GQA
            k_full = self._repeat_kv(k_full)
            v_full = self._repeat_kv(v_full)
            
            # Compute attention
            scores = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            attn = F.softmax(scores, dim=-1)
            output = torch.matmul(attn, v_full)
            
            # Reshape and project
            output = output.transpose(1, 2).reshape(batch, seq_len, self.d_model)
            output = self.W_o(output)
            
            return output, cache
        """
        batch, seq_len, _ = x.shape
        dummy_cache = KVCache(
            k_cache=torch.zeros(batch, self.num_kv_heads, 1, self.head_dim),
            v_cache=torch.zeros(batch, self.num_kv_heads, 1, self.head_dim),
            seq_len=0
        )
        return torch.zeros_like(x), dummy_cache  # Replace


# ============================================================================
# Exercise 3: Cached Transformer Block
# ============================================================================

class CachedTransformerBlock(nn.Module):
    """
    Transformer block with KV caching support.
    """
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int = None,
                 d_ff: int = None):
        super().__init__()
        
        d_ff = d_ff or d_model * 4
        
        # TODO: Initialize components
        # HINT:
        #   self.attention = CachedAttention(d_model, num_heads, num_kv_heads)
        #   self.ffn = nn.Sequential(
        #       nn.Linear(d_model, d_ff),
        #       nn.GELU(),
        #       nn.Linear(d_ff, d_model)
        #   )
        #   self.norm1 = nn.LayerNorm(d_model)
        #   self.norm2 = nn.LayerNorm(d_model)
        self.attention = None  # Replace
        self.ffn = None        # Replace
        self.norm1 = None      # Replace
        self.norm2 = None      # Replace
    
    def forward(self, x: torch.Tensor, 
                cache: Optional[KVCache] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, KVCache]:
        """
        Forward pass with cache.
        
        TODO: Implement pre-norm transformer block with cache
        HINT:
            # Attention with cache
            normed = self.norm1(x)
            attn_out, cache = self.attention(normed, cache, mask)
            x = x + attn_out
            
            # FFN (no caching needed)
            normed = self.norm2(x)
            x = x + self.ffn(normed)
            
            return x, cache
        """
        dummy_cache = KVCache(
            k_cache=torch.zeros(1),
            v_cache=torch.zeros(1),
            seq_len=0
        )
        return x, dummy_cache  # Replace


# ============================================================================
# Exercise 4: Multi-Layer Cache Manager
# ============================================================================

class LayerCaches:
    """
    Manages KV caches for all layers in a transformer.
    """
    
    def __init__(self, num_layers: int):
        self.caches: Dict[int, KVCache] = {}
        self.num_layers = num_layers
    
    def get(self, layer_idx: int) -> Optional[KVCache]:
        """Get cache for a specific layer."""
        return self.caches.get(layer_idx)
    
    def set(self, layer_idx: int, cache: KVCache):
        """Set cache for a specific layer."""
        self.caches[layer_idx] = cache
    
    def clear(self):
        """Clear all caches."""
        self.caches.clear()


def create_layer_caches(num_layers: int, batch_size: int, num_kv_heads: int,
                        max_seq_len: int, head_dim: int, 
                        device: torch.device = None) -> LayerCaches:
    """
    Create caches for all transformer layers.
    
    Args:
        num_layers: Number of transformer layers
        batch_size: Batch size
        num_kv_heads: Number of KV heads per layer
        max_seq_len: Maximum sequence length
        head_dim: Dimension per head
        device: Device for tensors
    
    Returns:
        LayerCaches with pre-allocated caches for all layers
    
    TODO: Create caches for all layers
    HINT:
        caches = LayerCaches(num_layers)
        for i in range(num_layers):
            cache = create_kv_cache(batch_size, num_kv_heads, max_seq_len, head_dim, device)
            caches.set(i, cache)
        return caches
    """
    return LayerCaches(num_layers)  # Replace


# ============================================================================
# Exercise 5: Cached Transformer Model
# ============================================================================

class CachedTransformer(nn.Module):
    """
    Complete transformer with KV caching for efficient inference.
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int,
                 num_layers: int, num_kv_heads: int = None, max_seq_len: int = 2048):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads or num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // num_heads
        
        # TODO: Initialize model components
        # HINT:
        #   self.token_emb = nn.Embedding(vocab_size, d_model)
        #   self.layers = nn.ModuleList([
        #       CachedTransformerBlock(d_model, num_heads, self.num_kv_heads)
        #       for _ in range(num_layers)
        #   ])
        #   self.final_norm = nn.LayerNorm(d_model)
        #   self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.token_emb = None   # Replace
        self.layers = None      # Replace
        self.final_norm = None  # Replace
        self.output_proj = None # Replace
    
    def forward(self, token_ids: torch.Tensor,
                layer_caches: Optional[LayerCaches] = None,
                start_pos: int = 0) -> Tuple[torch.Tensor, LayerCaches]:
        """
        Forward pass with caching.
        
        Args:
            token_ids: Token indices (batch, seq_len)
            layer_caches: Existing caches (None for prefill)
            start_pos: Starting position in sequence (for positional encoding)
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            layer_caches: Updated caches
        
        TODO: Implement forward pass with per-layer caching
        HINT:
            batch, seq_len = token_ids.shape
            
            # Get embeddings
            x = self.token_emb(token_ids)
            
            # Create caches if needed
            if layer_caches is None:
                layer_caches = LayerCaches(self.num_layers)
            
            # Create causal mask for current tokens
            # During decode (seq_len=1), we attend to all cached positions
            total_len = start_pos + seq_len
            mask = torch.tril(torch.ones(total_len, total_len, device=x.device))
            mask = mask[start_pos:total_len, :total_len]  # Only rows for new tokens
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, total_len)
            
            # Forward through layers
            for i, layer in enumerate(self.layers):
                cache = layer_caches.get(i)
                x, new_cache = layer(x, cache, mask)
                layer_caches.set(i, new_cache)
            
            x = self.final_norm(x)
            logits = self.output_proj(x)
            
            return logits, layer_caches
        """
        batch, seq_len = token_ids.shape
        if layer_caches is None:
            layer_caches = LayerCaches(self.num_layers)
        return torch.zeros(batch, seq_len, 1000), layer_caches  # Replace


# ============================================================================
# Exercise 6: Autoregressive Generation with Cache
# ============================================================================

def generate_with_cache(model: CachedTransformer, 
                        prompt_ids: torch.Tensor,
                        max_new_tokens: int,
                        temperature: float = 1.0) -> torch.Tensor:
    """
    Generate tokens autoregressively using KV cache.
    
    Args:
        model: CachedTransformer model
        prompt_ids: Initial token IDs (batch, prompt_len)
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Full sequence including prompt and generated tokens
    
    TODO: Implement generation loop with cache
    HINT:
        model.eval()
        device = prompt_ids.device
        
        # Prefill: process entire prompt
        with torch.no_grad():
            logits, caches = model(prompt_ids)
            
        generated = prompt_ids.clone()
        
        # Generate new tokens one at a time
        for i in range(max_new_tokens):
            # Get logits for last position
            next_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Decode: process only new token with cache
            with torch.no_grad():
                logits, caches = model(next_token, caches, start_pos=generated.size(1)-1)
        
        return generated
    """
    return prompt_ids  # Replace


# ============================================================================
# Exercise 7: Cache Memory Analysis
# ============================================================================

def compute_cache_memory(num_layers: int, num_kv_heads: int, head_dim: int,
                         max_seq_len: int, batch_size: int,
                         dtype_bytes: int = 2) -> dict:
    """
    Compute memory required for KV cache.
    
    Args:
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV heads per layer
        head_dim: Dimension per head
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)
    
    Returns:
        Dictionary with memory analysis
    
    TODO: Compute cache memory requirements
    HINT:
        # K and V cache per layer
        per_layer_elements = 2 * batch_size * num_kv_heads * max_seq_len * head_dim
        per_layer_bytes = per_layer_elements * dtype_bytes
        
        total_bytes = per_layer_bytes * num_layers
        
        return {
            'per_layer_bytes': per_layer_bytes,
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'total_gb': total_bytes / (1024 * 1024 * 1024),
            'per_token_bytes': total_bytes / max_seq_len
        }
    """
    return {
        'per_layer_bytes': 0,
        'total_bytes': 0,
        'total_mb': 0.0,
        'total_gb': 0.0,
        'per_token_bytes': 0
    }  # Replace


if __name__ == "__main__":
    print("Day 23: KV Cache for Inference")
    print("=" * 50)
    
    # Configuration
    d_model = 256
    num_heads = 8
    num_kv_heads = 2
    num_layers = 4
    vocab_size = 1000
    batch_size = 2
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  num_layers: {num_layers}")
    
    # Test cache creation
    print("\nTesting KV Cache:")
    cache = create_kv_cache(batch_size, num_kv_heads, 100, d_model // num_heads)
    print(f"  K cache shape: {cache.k_cache.shape}")
    print(f"  Initial seq_len: {cache.seq_len}")
    
    # Test cache update
    new_k = torch.randn(batch_size, num_kv_heads, 10, d_model // num_heads)
    new_v = torch.randn(batch_size, num_kv_heads, 10, d_model // num_heads)
    cache = update_kv_cache(cache, new_k, new_v)
    print(f"  After update seq_len: {cache.seq_len}")
    
    # Test cached attention
    print("\nTesting CachedAttention:")
    attn = CachedAttention(d_model, num_heads, num_kv_heads)
    if attn.W_q is not None:
        x = torch.randn(batch_size, 16, d_model)
        output, cache = attn(x)
        print(f"  Prefill output: {output.shape}")
        
        # Single token decode
        x_single = torch.randn(batch_size, 1, d_model)
        output, cache = attn(x_single, cache)
        print(f"  Decode output: {output.shape}")
        print(f"  Cache seq_len: {cache.seq_len}")
    
    # Memory analysis
    print("\nCache Memory Analysis (7B model config):")
    mem = compute_cache_memory(
        num_layers=32,
        num_kv_heads=8,  # GQA
        head_dim=128,
        max_seq_len=4096,
        batch_size=1,
        dtype_bytes=2  # fp16
    )
    for k, v in mem.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    print("\nRun test_day23.py to verify your implementations!")
