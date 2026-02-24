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
    """
    # TODO: Create the cache tensors
    # API hints:
    # - torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device) -> K cache
    # - torch.zeros(batch_size, num_kv_heads, max_seq_len, head_dim, device=device) -> V cache
    # - return KVCache(k_cache=k_cache, v_cache=v_cache, seq_len=0)
    return KVCache(
        k_cache=torch.zeros(1),
        v_cache=torch.zeros(1),
        seq_len=0
    )


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
    """
    # TODO: Append new K, V to cache at correct positions
    # API hints:
    # - new_k.size(2) -> get new sequence length
    # - start_pos = cache.seq_len, end_pos = start_pos + new_seq_len
    # - cache.k_cache[:, :, start_pos:end_pos, :] = new_k -> write to cache
    # - cache.v_cache[:, :, start_pos:end_pos, :] = new_v -> write to cache
    # - cache.seq_len = end_pos -> update length
    return cache


def get_cached_kv(cache: KVCache) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve valid (non-padding) entries from cache.
    
    Args:
        cache: KV cache
    
    Returns:
        (k, v) tensors of shape (batch, num_kv_heads, seq_len, head_dim)
    """
    # TODO: Slice cache to valid length
    # API hints:
    # - cache.k_cache[:, :, :cache.seq_len, :] -> get valid K entries
    # - cache.v_cache[:, :, :cache.seq_len, :] -> get valid V entries
    return cache.k_cache, cache.v_cache


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
        # API hints:
        # - nn.Linear(d_model, num_heads * head_dim, bias=False) -> Q projection
        # - nn.Linear(d_model, num_kv_heads * head_dim, bias=False) -> K, V projections
        # - nn.Linear(d_model, d_model, bias=False) -> output projection
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None
    
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
        """
        # TODO: Implement cached attention
        # API hints:
        # - self.W_q(x), self.W_k(x), self.W_v(x) -> project to Q, K, V
        # - tensor.view(batch, seq, num_heads, head_dim).transpose(1, 2) -> reshape
        # - update_kv_cache(cache, k, v) -> update cache with new K, V
        # - get_cached_kv(cache) -> get full K, V from cache
        # - self._repeat_kv(k_full) -> repeat KV for GQA
        # - torch.matmul(q, k.transpose(-2, -1)) * self.scale -> attention scores
        # - F.softmax(scores, dim=-1) -> attention weights
        # - torch.matmul(attn, v) -> weighted sum
        # - output.transpose(1, 2).reshape(batch, seq, d_model) -> reshape
        # - self.W_o(output) -> output projection
        batch, seq_len, _ = x.shape
        dummy_cache = KVCache(
            k_cache=torch.zeros(batch, self.num_kv_heads, 1, self.head_dim),
            v_cache=torch.zeros(batch, self.num_kv_heads, 1, self.head_dim),
            seq_len=0
        )
        return torch.zeros_like(x), dummy_cache


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
        # API hints:
        # - CachedAttention(d_model, num_heads, num_kv_heads) -> cached attention
        # - nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)) -> FFN
        # - nn.LayerNorm(d_model) -> layer norm
        self.attention = None
        self.ffn = None
        self.norm1 = None
        self.norm2 = None
    
    def forward(self, x: torch.Tensor, 
                cache: Optional[KVCache] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, KVCache]:
        """
        Forward pass with cache.
        """
        # TODO: Implement pre-norm transformer block with cache
        # API hints:
        # - self.norm1(x) -> pre-norm before attention
        # - self.attention(normed, cache, mask) -> returns (output, updated_cache)
        # - x + attn_out -> residual connection
        # - self.norm2(x) -> pre-norm before FFN
        # - self.ffn(normed) -> FFN output
        dummy_cache = KVCache(
            k_cache=torch.zeros(1),
            v_cache=torch.zeros(1),
            seq_len=0
        )
        return x, dummy_cache


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
    """
    # TODO: Create caches for all layers
    # API hints:
    # - LayerCaches(num_layers) -> create cache manager
    # - create_kv_cache(batch_size, num_kv_heads, max_seq_len, head_dim, device) -> per-layer cache
    # - caches.set(i, cache) -> store cache for layer i
    return LayerCaches(num_layers)


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
        # API hints:
        # - nn.Embedding(vocab_size, d_model) -> token embedding
        # - nn.ModuleList([CachedTransformerBlock(...) for _ in range(num_layers)]) -> layers
        # - nn.LayerNorm(d_model) -> final layer norm
        # - nn.Linear(d_model, vocab_size, bias=False) -> output projection
        self.token_emb = None
        self.layers = None
        self.final_norm = None
        self.output_proj = None
    
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
        """
        # TODO: Implement forward pass with per-layer caching
        # API hints:
        # - self.token_emb(token_ids) -> get embeddings
        # - LayerCaches(self.num_layers) -> create new caches if needed
        # - torch.tril(torch.ones(total_len, total_len)) -> causal mask
        # - mask[start_pos:total_len, :total_len] -> slice for decode
        # - for i, layer in enumerate(self.layers): -> iterate layers
        # - layer_caches.get(i), layer_caches.set(i, cache) -> get/set cache
        # - layer(x, cache, mask) -> returns (output, updated_cache)
        # - self.final_norm(x) -> final normalization
        # - self.output_proj(x) -> project to vocab
        batch, seq_len = token_ids.shape
        if layer_caches is None:
            layer_caches = LayerCaches(self.num_layers)
        return torch.zeros(batch, seq_len, 1000), layer_caches


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
    """
    # TODO: Implement generation loop with cache
    # API hints:
    # - model.eval() -> set to eval mode
    # - model(prompt_ids) -> prefill, returns (logits, caches)
    # - logits[:, -1, :] / temperature -> get last position logits with temp
    # - F.softmax(next_logits, dim=-1) -> softmax for sampling
    # - torch.multinomial(probs, num_samples=1) -> sample next token
    # - torch.cat([generated, next_token], dim=1) -> append to sequence
    # - model(next_token, caches, start_pos=...) -> decode with cache
    return prompt_ids


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
    """
    # TODO: Compute cache memory requirements
    # API hints:
    # - per_layer_elements = 2 * batch_size * num_kv_heads * max_seq_len * head_dim
    # - per_layer_bytes = per_layer_elements * dtype_bytes
    # - total_bytes = per_layer_bytes * num_layers
    # - total_mb = total_bytes / (1024 * 1024)
    # - total_gb = total_bytes / (1024 * 1024 * 1024)
    return {
        'per_layer_bytes': 0,
        'total_bytes': 0,
        'total_mb': 0.0,
        'total_gb': 0.0,
        'per_token_bytes': 0
    }


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
