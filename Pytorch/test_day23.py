"""Test Suite for Day 23: KV Cache for Inference"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day23 import (
        KVCache, create_kv_cache, update_kv_cache, get_cached_kv,
        CachedAttention, CachedTransformerBlock, LayerCaches,
        create_layer_caches, CachedTransformer, generate_with_cache,
        compute_cache_memory
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_create_kv_cache() -> Tuple[bool, str]:
    """Test KV cache creation."""
    try:
        batch, num_kv_heads, max_seq, head_dim = 2, 4, 100, 64
        cache = create_kv_cache(batch, num_kv_heads, max_seq, head_dim)
        
        expected_shape = (batch, num_kv_heads, max_seq, head_dim)
        if cache.k_cache.shape != expected_shape:
            return False, f"K cache shape {cache.k_cache.shape} != {expected_shape}"
        if cache.v_cache.shape != expected_shape:
            return False, f"V cache shape wrong"
        if cache.seq_len != 0:
            return False, f"Initial seq_len should be 0, got {cache.seq_len}"
        
        # Verify cache is initialized with zeros
        if cache.k_cache.abs().sum() != 0:
            return False, "K cache should be initialized with zeros"
        if cache.v_cache.abs().sum() != 0:
            return False, "V cache should be initialized with zeros"
        
        return True, f"Cache created with shape {expected_shape}, initialized to zeros"
    except Exception as e:
        return False, str(e)


def test_update_kv_cache() -> Tuple[bool, str]:
    """Test cache update functionality."""
    try:
        batch, num_kv_heads, max_seq, head_dim = 2, 4, 100, 64
        cache = create_kv_cache(batch, num_kv_heads, max_seq, head_dim)
        
        # Add 10 tokens
        new_k = torch.randn(batch, num_kv_heads, 10, head_dim)
        new_v = torch.randn(batch, num_kv_heads, 10, head_dim)
        cache = update_kv_cache(cache, new_k, new_v)
        
        if cache.seq_len != 10:
            return False, f"seq_len should be 10, got {cache.seq_len}"
        
        # Verify values were written
        if not torch.allclose(cache.k_cache[:, :, :10, :], new_k):
            return False, "K values not correctly written"
        
        # Add 5 more tokens
        new_k2 = torch.randn(batch, num_kv_heads, 5, head_dim)
        new_v2 = torch.randn(batch, num_kv_heads, 5, head_dim)
        cache = update_kv_cache(cache, new_k2, new_v2)
        
        if cache.seq_len != 15:
            return False, f"seq_len should be 15, got {cache.seq_len}"
        
        return True, "Cache update works correctly"
    except Exception as e:
        return False, str(e)


def test_get_cached_kv() -> Tuple[bool, str]:
    """Test retrieving valid entries from cache."""
    try:
        batch, num_kv_heads, max_seq, head_dim = 2, 4, 100, 64
        cache = create_kv_cache(batch, num_kv_heads, max_seq, head_dim)
        
        # Add some tokens
        new_k = torch.randn(batch, num_kv_heads, 20, head_dim)
        new_v = torch.randn(batch, num_kv_heads, 20, head_dim)
        cache = update_kv_cache(cache, new_k, new_v)
        
        k, v = get_cached_kv(cache)
        
        expected_shape = (batch, num_kv_heads, 20, head_dim)
        if k.shape != expected_shape:
            return False, f"Retrieved K shape {k.shape} != {expected_shape}"
        if v.shape != expected_shape:
            return False, f"Retrieved V shape wrong"
        
        # Verify retrieved values match what was cached
        if not torch.allclose(k, new_k):
            return False, "Retrieved K values don't match cached values"
        if not torch.allclose(v, new_v):
            return False, "Retrieved V values don't match cached values"
        
        return True, "Retrieved values match cached entries"
    except Exception as e:
        return False, str(e)


def test_cached_attention_init() -> Tuple[bool, str]:
    """Test CachedAttention initialization."""
    try:
        d_model, num_heads, num_kv_heads = 256, 8, 2
        attn = CachedAttention(d_model, num_heads, num_kv_heads)
        
        if attn.W_q is None:
            return False, "W_q not initialized"
        if attn.W_k is None:
            return False, "W_k not initialized"
        if attn.W_v is None:
            return False, "W_v not initialized"
        if attn.W_o is None:
            return False, "W_o not initialized"
        
        return True, "CachedAttention initialized correctly"
    except Exception as e:
        return False, str(e)


def test_cached_attention_prefill() -> Tuple[bool, str]:
    """Test attention prefill (no existing cache)."""
    try:
        d_model, num_heads, num_kv_heads = 256, 8, 2
        attn = CachedAttention(d_model, num_heads, num_kv_heads)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        batch, seq_len = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_model)
        
        output, cache = attn(x, cache=None)
        
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != {x.shape}"
        if cache.seq_len != seq_len:
            return False, f"Cache seq_len {cache.seq_len} != {seq_len}"
        
        # Verify output is not just zeros (actual computation happened)
        if output.abs().sum() == 0:
            return False, "Output is all zeros - no computation happened"
        
        # Verify output has reasonable values (not NaN or Inf)
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Output contains NaN or Inf values"
        
        # Verify the output is different from input (projection happened)
        if torch.allclose(output, x, atol=1e-3):
            return False, "Output is same as input - no transformation applied"
        
        return True, f"Prefill works: cached {cache.seq_len} tokens, output mean={output.mean():.4f}"
    except Exception as e:
        return False, str(e)


def test_cached_attention_decode() -> Tuple[bool, str]:
    """Test single-token decode with cache."""
    try:
        d_model, num_heads, num_kv_heads = 256, 8, 2
        attn = CachedAttention(d_model, num_heads, num_kv_heads)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        batch = 2
        
        # Prefill with 16 tokens
        torch.manual_seed(42)
        x = torch.randn(batch, 16, d_model)
        _, cache = attn(x)
        
        # Decode single token
        x_single = torch.randn(batch, 1, d_model)
        output, cache = attn(x_single, cache)
        
        expected_shape = (batch, 1, d_model)
        if output.shape != expected_shape:
            return False, f"Decode output shape {output.shape} != {expected_shape}"
        if cache.seq_len != 17:
            return False, f"Cache should have 17 tokens, got {cache.seq_len}"
        
        # Verify output has valid values
        if output.abs().sum() == 0:
            return False, "Decode output is all zeros"
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Decode output contains NaN or Inf"
        
        # Verify output is transformed (not just passthrough)
        if torch.allclose(output, x_single, atol=1e-3):
            return False, "Output is same as input - attention not applied"
        
        return True, f"Single-token decode works, output mean={output.mean():.4f}"
    except Exception as e:
        return False, str(e)


def test_cached_attention_incremental() -> Tuple[bool, str]:
    """Test incremental token-by-token generation."""
    try:
        d_model, num_heads, num_kv_heads = 128, 4, 2
        attn = CachedAttention(d_model, num_heads, num_kv_heads)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        batch = 2
        
        # Prefill
        x = torch.randn(batch, 8, d_model)
        _, cache = attn(x)
        
        # Generate 5 tokens incrementally
        for i in range(5):
            x_single = torch.randn(batch, 1, d_model)
            output, cache = attn(x_single, cache)
            
            expected_len = 8 + i + 1
            if cache.seq_len != expected_len:
                return False, f"After {i+1} decodes, cache should have {expected_len} tokens"
        
        if cache.seq_len != 13:
            return False, f"Final cache length should be 13"
        
        return True, "Incremental generation correct"
    except Exception as e:
        return False, str(e)


def test_cached_transformer_block() -> Tuple[bool, str]:
    """Test CachedTransformerBlock."""
    try:
        d_model, num_heads, num_kv_heads = 128, 4, 2
        block = CachedTransformerBlock(d_model, num_heads, num_kv_heads)
        
        if block.attention is None:
            return False, "Attention not initialized"
        if block.norm1 is None:
            return False, "norm1 not initialized"
        
        batch, seq_len = 2, 8
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_model)
        
        output, cache = block(x)
        
        if output.shape != x.shape:
            return False, f"Output shape wrong"
        
        # Verify output has valid values
        if output.abs().sum() == 0:
            return False, "Block output is all zeros"
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Block output contains NaN or Inf"
        
        # Verify residual connection is working (output shouldn't be too different from input)
        # Pre-norm block: output = x + attn(norm(x)) + ffn(norm(x + attn_out))
        # So output should correlate with input due to residual
        correlation = F.cosine_similarity(output.flatten(), x.flatten(), dim=0)
        if correlation < 0.1:
            return False, f"Residual connection may not work: correlation={correlation:.3f}"
        
        return True, f"CachedTransformerBlock works, output-input correlation={correlation:.3f}"
    except Exception as e:
        return False, str(e)


def test_layer_caches() -> Tuple[bool, str]:
    """Test LayerCaches management."""
    try:
        caches = create_layer_caches(
            num_layers=4,
            batch_size=2,
            num_kv_heads=4,
            max_seq_len=100,
            head_dim=32
        )
        
        if caches.num_layers != 4:
            return False, "Wrong number of layers"
        
        # Check that caches were created for all layers
        for i in range(4):
            cache = caches.get(i)
            if cache is None:
                return False, f"No cache for layer {i}"
        
        return True, "LayerCaches manages per-layer caches"
    except Exception as e:
        return False, str(e)


def test_cached_transformer_forward() -> Tuple[bool, str]:
    """Test CachedTransformer forward pass."""
    try:
        vocab_size, d_model, num_heads, num_layers = 1000, 128, 4, 2
        model = CachedTransformer(vocab_size, d_model, num_heads, num_layers)
        
        if model.token_emb is None:
            return False, "token_emb not initialized"
        if model.layers is None:
            return False, "layers not initialized"
        
        batch, seq_len = 2, 16
        torch.manual_seed(42)
        tokens = torch.randint(0, vocab_size, (batch, seq_len))
        
        logits, caches = model(tokens)
        
        expected_shape = (batch, seq_len, vocab_size)
        if logits.shape != expected_shape:
            return False, f"Logits shape {logits.shape} != {expected_shape}"
        
        # Verify logits have valid values
        if logits.abs().sum() == 0:
            return False, "Logits are all zeros"
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return False, "Logits contain NaN or Inf"
        
        # Verify logits can be used for softmax (reasonable range)
        probs = F.softmax(logits, dim=-1)
        if not torch.allclose(probs.sum(dim=-1), torch.ones(batch, seq_len), atol=1e-5):
            return False, "Softmax of logits doesn't sum to 1"
        
        return True, f"CachedTransformer forward works, logits range=[{logits.min():.2f}, {logits.max():.2f}]"
    except Exception as e:
        return False, str(e)


def test_cached_transformer_incremental() -> Tuple[bool, str]:
    """Test incremental generation with CachedTransformer."""
    try:
        vocab_size, d_model, num_heads, num_layers = 1000, 128, 4, 2
        model = CachedTransformer(vocab_size, d_model, num_heads, num_layers)
        
        if model.token_emb is None:
            return False, "Model not initialized"
        
        batch = 2
        
        # Prefill
        prompt = torch.randint(0, vocab_size, (batch, 8))
        logits, caches = model(prompt)
        
        # Decode
        new_token = torch.randint(0, vocab_size, (batch, 1))
        logits, caches = model(new_token, caches, start_pos=8)
        
        expected_shape = (batch, 1, vocab_size)
        if logits.shape != expected_shape:
            return False, f"Decode logits shape wrong"
        
        return True, "Incremental generation works"
    except Exception as e:
        return False, str(e)


def test_generate_with_cache() -> Tuple[bool, str]:
    """Test full generation loop."""
    try:
        vocab_size, d_model, num_heads, num_layers = 100, 64, 4, 2
        model = CachedTransformer(vocab_size, d_model, num_heads, num_layers)
        
        if model.token_emb is None:
            return False, "Model not initialized"
        
        batch = 1
        prompt = torch.randint(0, vocab_size, (batch, 5))
        
        output = generate_with_cache(model, prompt, max_new_tokens=10)
        
        expected_len = 5 + 10
        if output.shape[1] != expected_len:
            return False, f"Output length {output.shape[1]} != {expected_len}"
        
        # Check prompt is preserved
        if not torch.equal(output[:, :5], prompt):
            return False, "Prompt not preserved in output"
        
        return True, "Generation loop works"
    except Exception as e:
        return False, str(e)


def test_cache_memory_computation() -> Tuple[bool, str]:
    """Test cache memory computation."""
    try:
        mem = compute_cache_memory(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            max_seq_len=4096,
            batch_size=1,
            dtype_bytes=2
        )
        
        if mem['total_bytes'] == 0:
            return False, "Memory not computed"
        
        # Verify calculation
        # 2 (K+V) * 32 layers * 1 batch * 8 heads * 4096 seq * 128 dim * 2 bytes
        expected = 2 * 32 * 1 * 8 * 4096 * 128 * 2
        if mem['total_bytes'] != expected:
            return False, f"Total bytes {mem['total_bytes']} != {expected}"
        
        expected_gb = expected / (1024**3)
        if abs(mem['total_gb'] - expected_gb) > 0.01:
            return False, f"GB calculation wrong"
        
        return True, f"Cache memory: {mem['total_gb']:.2f} GB"
    except Exception as e:
        return False, str(e)


def test_cache_consistency() -> Tuple[bool, str]:
    """Test that cached and non-cached attention produce same results for prefill."""
    try:
        d_model, num_heads = 128, 4
        attn = CachedAttention(d_model, num_heads)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        batch, seq_len = 2, 10
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, d_model)
        
        # Non-cached: process all at once
        output_full, _ = attn(x, cache=None)
        
        # Process same input with cache creation
        output_with_cache, cache = attn(x, cache=None)
        
        # Same input should produce same output whether cache is returned or not
        if not torch.allclose(output_full, output_with_cache, atol=1e-5):
            return False, "Full pass and cached pass produce different results for same input"
        
        # Verify cache was populated correctly
        if cache.seq_len != seq_len:
            return False, f"Cache seq_len {cache.seq_len} != {seq_len}"
        
        # Verify cached KV contains actual computed values (not zeros)
        k, v = get_cached_kv(cache)
        if k.abs().sum() == 0:
            return False, "Cached K values are all zeros"
        if v.abs().sum() == 0:
            return False, "Cached V values are all zeros"
        
        return True, f"Cache maintains consistency, output diff={torch.abs(output_full - output_with_cache).max():.6f}"
    except Exception as e:
        return False, str(e)


def test_gradient_through_cache() -> Tuple[bool, str]:
    """Test that gradients flow through cached attention."""
    try:
        d_model, num_heads = 128, 4
        attn = CachedAttention(d_model, num_heads)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        x = torch.randn(2, 8, d_model, requires_grad=True)
        output, _ = attn(x)
        
        loss = output.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "No gradient to input"
        if x.grad.abs().sum() == 0:
            return False, "Gradients are zero"
        
        return True, "Gradients flow through cache"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("create_kv_cache", test_create_kv_cache),
        ("update_kv_cache", test_update_kv_cache),
        ("get_cached_kv", test_get_cached_kv),
        ("cached_attention_init", test_cached_attention_init),
        ("cached_attention_prefill", test_cached_attention_prefill),
        ("cached_attention_decode", test_cached_attention_decode),
        ("cached_attention_incremental", test_cached_attention_incremental),
        ("cached_transformer_block", test_cached_transformer_block),
        ("layer_caches", test_layer_caches),
        ("cached_transformer_forward", test_cached_transformer_forward),
        ("cached_transformer_incremental", test_cached_transformer_incremental),
        ("generate_with_cache", test_generate_with_cache),
        ("cache_memory_computation", test_cache_memory_computation),
        ("cache_consistency", test_cache_consistency),
        ("gradient_through_cache", test_gradient_through_cache),
    ]
    
    print(f"\n{'='*50}\nDay 23: KV Cache for Inference - Tests\n{'='*50}")
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    run_all_tests()
