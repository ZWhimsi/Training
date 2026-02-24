"""Test Suite for Day 23: KV Cache for Inference"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
try:
    from day23 import (
        KVCache, create_kv_cache, update_kv_cache, get_cached_kv,
        CachedAttention, CachedTransformerBlock, LayerCaches,
        create_layer_caches, CachedTransformer, generate_with_cache,
        compute_cache_memory
    )
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_create_kv_cache():
    """Test KV cache creation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, num_kv_heads, max_seq, head_dim = 2, 4, 100, 64
    cache = create_kv_cache(batch, num_kv_heads, max_seq, head_dim)
    
    expected_shape = (batch, num_kv_heads, max_seq, head_dim)
    assert cache.k_cache.shape == expected_shape, f"K cache shape {cache.k_cache.shape} != {expected_shape}"
    assert cache.v_cache.shape == expected_shape, "V cache shape wrong"
    assert cache.seq_len == 0, f"Initial seq_len should be 0, got {cache.seq_len}"
    
    assert cache.k_cache.abs().sum() == 0, "K cache should be initialized with zeros"
    assert cache.v_cache.abs().sum() == 0, "V cache should be initialized with zeros"

def test_update_kv_cache():
    """Test cache update functionality."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, num_kv_heads, max_seq, head_dim = 2, 4, 100, 64
    cache = create_kv_cache(batch, num_kv_heads, max_seq, head_dim)
    
    new_k = torch.randn(batch, num_kv_heads, 10, head_dim)
    new_v = torch.randn(batch, num_kv_heads, 10, head_dim)
    cache = update_kv_cache(cache, new_k, new_v)
    
    assert cache.seq_len == 10, f"seq_len should be 10, got {cache.seq_len}"
    
    assert torch.allclose(cache.k_cache[:, :, :10, :], new_k), "K values not correctly written"
    
    new_k2 = torch.randn(batch, num_kv_heads, 5, head_dim)
    new_v2 = torch.randn(batch, num_kv_heads, 5, head_dim)
    cache = update_kv_cache(cache, new_k2, new_v2)
    
    assert cache.seq_len == 15, f"seq_len should be 15, got {cache.seq_len}"

def test_get_cached_kv():
    """Test retrieving valid entries from cache."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, num_kv_heads, max_seq, head_dim = 2, 4, 100, 64
    cache = create_kv_cache(batch, num_kv_heads, max_seq, head_dim)
    
    new_k = torch.randn(batch, num_kv_heads, 20, head_dim)
    new_v = torch.randn(batch, num_kv_heads, 20, head_dim)
    cache = update_kv_cache(cache, new_k, new_v)
    
    k, v = get_cached_kv(cache)
    
    expected_shape = (batch, num_kv_heads, 20, head_dim)
    assert k.shape == expected_shape, f"Retrieved K shape {k.shape} != {expected_shape}"
    assert v.shape == expected_shape, "Retrieved V shape wrong"
    
    assert torch.allclose(k, new_k), "Retrieved K values don't match cached values"
    assert torch.allclose(v, new_v), "Retrieved V values don't match cached values"

def test_cached_attention_init():
    """Test CachedAttention initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 256, 8, 2
    attn = CachedAttention(d_model, num_heads, num_kv_heads)
    
    assert attn.W_q is not None, "W_q not initialized"
    assert attn.W_k is not None, "W_k not initialized"
    assert attn.W_v is not None, "W_v not initialized"
    assert attn.W_o is not None, "W_o not initialized"

def test_cached_attention_prefill():
    """Test attention prefill (no existing cache)."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 256, 8, 2
    attn = CachedAttention(d_model, num_heads, num_kv_heads)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    batch, seq_len = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    
    output, cache = attn(x, cache=None)
    
    assert output.shape == x.shape, f"Output shape {output.shape} != {x.shape}"
    assert cache.seq_len == seq_len, f"Cache seq_len {cache.seq_len} != {seq_len}"
    
    assert output.abs().sum() != 0, "Output is all zeros - no computation happened"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Output contains NaN or Inf values"
    
    assert not torch.allclose(output, x, atol=1e-3), "Output is same as input - no transformation applied"

def test_cached_attention_decode():
    """Test single-token decode with cache."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 256, 8, 2
    attn = CachedAttention(d_model, num_heads, num_kv_heads)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    batch = 2
    
    torch.manual_seed(42)
    x = torch.randn(batch, 16, d_model)
    _, cache = attn(x)
    
    x_single = torch.randn(batch, 1, d_model)
    output, cache = attn(x_single, cache)
    
    expected_shape = (batch, 1, d_model)
    assert output.shape == expected_shape, f"Decode output shape {output.shape} != {expected_shape}"
    assert cache.seq_len == 17, f"Cache should have 17 tokens, got {cache.seq_len}"
    
    assert output.abs().sum() != 0, "Decode output is all zeros"
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Decode output contains NaN or Inf"
    
    assert not torch.allclose(output, x_single, atol=1e-3), "Output is same as input - attention not applied"

def test_cached_attention_incremental():
    """Test incremental token-by-token generation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 128, 4, 2
    attn = CachedAttention(d_model, num_heads, num_kv_heads)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    batch = 2
    
    x = torch.randn(batch, 8, d_model)
    _, cache = attn(x)
    
    for i in range(5):
        x_single = torch.randn(batch, 1, d_model)
        output, cache = attn(x_single, cache)
        
        expected_len = 8 + i + 1
        assert cache.seq_len == expected_len, f"After {i+1} decodes, cache should have {expected_len} tokens"
    
    assert cache.seq_len == 13, "Final cache length should be 13"

def test_cached_transformer_block():
    """Test CachedTransformerBlock."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 128, 4, 2
    block = CachedTransformerBlock(d_model, num_heads, num_kv_heads)
    
    assert block.attention is not None, "Attention not initialized"
    assert block.norm1 is not None, "norm1 not initialized"
    
    batch, seq_len = 2, 8
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    
    output, cache = block(x)
    
    assert output.shape == x.shape, "Output shape wrong"
    
    assert output.abs().sum() != 0, "Block output is all zeros"
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Block output contains NaN or Inf"
    
    correlation = F.cosine_similarity(output.flatten(), x.flatten(), dim=0)
    assert correlation >= 0.1, f"Residual connection may not work: correlation={correlation:.3f}"

def test_layer_caches():
    """Test LayerCaches management."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    caches = create_layer_caches(
        num_layers=4,
        batch_size=2,
        num_kv_heads=4,
        max_seq_len=100,
        head_dim=32
    )
    
    assert caches.num_layers == 4, "Wrong number of layers"
    
    for i in range(4):
        cache = caches.get(i)
        assert cache is not None, f"No cache for layer {i}"

def test_cached_transformer_forward():
    """Test CachedTransformer forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    vocab_size, d_model, num_heads, num_layers = 1000, 128, 4, 2
    model = CachedTransformer(vocab_size, d_model, num_heads, num_layers)
    
    assert model.token_emb is not None, "token_emb not initialized"
    assert model.layers is not None, "layers not initialized"
    
    batch, seq_len = 2, 16
    torch.manual_seed(42)
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    
    logits, caches = model(tokens)
    
    expected_shape = (batch, seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Logits shape {logits.shape} != {expected_shape}"
    
    assert logits.abs().sum() != 0, "Logits are all zeros"
    assert not torch.isnan(logits).any() and not torch.isinf(logits).any(), "Logits contain NaN or Inf"
    
    probs = F.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch, seq_len), atol=1e-5), "Softmax of logits doesn't sum to 1"

def test_cached_transformer_incremental():
    """Test incremental generation with CachedTransformer."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    vocab_size, d_model, num_heads, num_layers = 1000, 128, 4, 2
    model = CachedTransformer(vocab_size, d_model, num_heads, num_layers)
    
    assert model.token_emb is not None, "Model not initialized"
    
    batch = 2
    
    prompt = torch.randint(0, vocab_size, (batch, 8))
    logits, caches = model(prompt)
    
    new_token = torch.randint(0, vocab_size, (batch, 1))
    logits, caches = model(new_token, caches, start_pos=8)
    
    expected_shape = (batch, 1, vocab_size)
    assert logits.shape == expected_shape, "Decode logits shape wrong"

def test_generate_with_cache():
    """Test full generation loop."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    vocab_size, d_model, num_heads, num_layers = 100, 64, 4, 2
    model = CachedTransformer(vocab_size, d_model, num_heads, num_layers)
    
    assert model.token_emb is not None, "Model not initialized"
    
    batch = 1
    prompt = torch.randint(0, vocab_size, (batch, 5))
    
    output = generate_with_cache(model, prompt, max_new_tokens=10)
    
    expected_len = 5 + 10
    assert output.shape[1] == expected_len, f"Output length {output.shape[1]} != {expected_len}"
    
    assert torch.equal(output[:, :5], prompt), "Prompt not preserved in output"

def test_cache_memory_computation():
    """Test cache memory computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    mem = compute_cache_memory(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=4096,
        batch_size=1,
        dtype_bytes=2
    )
    
    assert mem['total_bytes'] != 0, "Memory not computed"
    
    expected = 2 * 32 * 1 * 8 * 4096 * 128 * 2
    assert mem['total_bytes'] == expected, f"Total bytes {mem['total_bytes']} != {expected}"
    
    expected_gb = expected / (1024**3)
    assert abs(mem['total_gb'] - expected_gb) <= 0.01, "GB calculation wrong"

def test_cache_consistency():
    """Test that cached and non-cached attention produce same results for prefill."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 128, 4
    attn = CachedAttention(d_model, num_heads)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    batch, seq_len = 2, 10
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    
    output_full, _ = attn(x, cache=None)
    
    output_with_cache, cache = attn(x, cache=None)
    
    assert torch.allclose(output_full, output_with_cache, atol=1e-5), "Full pass and cached pass produce different results for same input"
    
    assert cache.seq_len == seq_len, f"Cache seq_len {cache.seq_len} != {seq_len}"
    
    k, v = get_cached_kv(cache)
    assert k.abs().sum() != 0, "Cached K values are all zeros"
    assert v.abs().sum() != 0, "Cached V values are all zeros"

def test_gradient_through_cache():
    """Test that gradients flow through cached attention."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 128, 4
    attn = CachedAttention(d_model, num_heads)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    x = torch.randn(2, 8, d_model, requires_grad=True)
    output, _ = attn(x)
    
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient to input"
    assert x.grad.abs().sum() != 0, "Gradients are zero"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
