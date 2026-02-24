"""Test Suite for Day 28: MLA Full Implementation"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
try:
    from day28 import (
        MLAConfig, MLAKVCache, MultiheadLatentAttention,
        MLATransformerBlock, MLAModel, generate_with_mla,
        compare_memory_usage
    )
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def get_test_config():
    """Get a test configuration with small dimensions."""
    return MLAConfig(
        d_model=256,
        num_heads=4,
        head_dim=32,
        d_kv_latent=64,
        d_q_latent=48,
        rope_dim=16,
        max_seq_len=128,
        use_q_compression=True
    )

def test_mla_config():
    """Test MLAConfig creation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    
    assert config.d_model == 256, "d_model not set"
    assert config.num_heads == 4, "num_heads not set"

def test_kv_cache_init():
    """Test MLAKVCache initialization with zeros."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    batch = 2
    max_len = 64
    
    cache = MLAKVCache(batch, max_len, config.d_kv_latent, 
                      config.num_heads, config.rope_dim)
    
    assert cache.c_kv is not None, "c_kv not initialized"
    assert cache.k_rope is not None, "k_rope not initialized"
    
    assert cache.c_kv.shape == (batch, max_len, config.d_kv_latent), "c_kv shape wrong"
    assert cache.k_rope.shape == (batch, max_len, config.num_heads, config.rope_dim), "k_rope shape wrong"
    
    assert cache.c_kv.abs().sum() == 0, "c_kv should be initialized with zeros"
    assert cache.k_rope.abs().sum() == 0, "k_rope should be initialized with zeros"
    
    assert cache.seq_len == 0, f"seq_len should be 0, got {cache.seq_len}"

def test_kv_cache_update():
    """Test MLAKVCache update stores values correctly."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    batch, max_len = 2, 64
    
    cache = MLAKVCache(batch, max_len, config.d_kv_latent,
                      config.num_heads, config.rope_dim)
    
    assert cache.c_kv is not None, "Cache not initialized"
    
    seq_len = 8
    torch.manual_seed(42)
    c_kv = torch.randn(batch, seq_len, config.d_kv_latent)
    k_rope = torch.randn(batch, seq_len, config.num_heads, config.rope_dim)
    
    new_len = cache.update(c_kv, k_rope)
    
    assert new_len == seq_len, f"New length {new_len} != {seq_len}"
    
    assert torch.allclose(cache.c_kv[:, :seq_len], c_kv), "c_kv values not stored correctly"
    assert torch.allclose(cache.k_rope[:, :seq_len], k_rope), "k_rope values not stored correctly"
    
    c_kv2 = torch.randn(batch, seq_len, config.d_kv_latent)
    k_rope2 = torch.randn(batch, seq_len, config.num_heads, config.rope_dim)
    new_len = cache.update(c_kv2, k_rope2)
    
    assert new_len == 2 * seq_len, f"New length {new_len} != {2 * seq_len}"
    
    assert torch.allclose(cache.c_kv[:, seq_len:2*seq_len], c_kv2), "Second c_kv values not appended correctly"

def test_kv_cache_get():
    """Test MLAKVCache retrieval returns correct values."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    batch, max_len = 2, 64
    
    cache = MLAKVCache(batch, max_len, config.d_kv_latent,
                      config.num_heads, config.rope_dim)
    
    assert cache.c_kv is not None, "Cache not initialized"
    
    seq_len = 8
    torch.manual_seed(42)
    c_kv = torch.randn(batch, seq_len, config.d_kv_latent)
    k_rope = torch.randn(batch, seq_len, config.num_heads, config.rope_dim)
    cache.update(c_kv, k_rope)
    
    cached_c, cached_k = cache.get()
    
    assert cached_c is not None and cached_k is not None, "Get returned None"
    
    assert cached_c.shape == (batch, seq_len, config.d_kv_latent), f"Retrieved c_kv shape wrong: {cached_c.shape}"
    
    assert torch.allclose(cached_c, c_kv), "Retrieved c_kv values don't match stored values"
    assert torch.allclose(cached_k, k_rope), "Retrieved k_rope values don't match stored values"

def test_mla_attention_init():
    """Test MultiheadLatentAttention initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiheadLatentAttention(config)
    
    assert mla.kv_down is not None, "kv_down not initialized"
    assert mla.k_up is not None, "k_up not initialized"
    assert mla.v_up is not None, "v_up not initialized"
    assert mla.out_proj is not None, "out_proj not initialized"

def test_mla_attention_compute_q():
    """Test query computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiheadLatentAttention(config)
    
    assert mla.kv_down is not None, "MLA not initialized"
    
    batch, seq = 2, 16
    x = torch.randn(batch, seq, config.d_model)
    
    q_content, q_rope = mla.compute_q(x)
    
    assert q_content.shape == (batch, config.num_heads, seq, config.head_dim), f"q_content shape wrong: {q_content.shape}"
    assert q_rope.shape == (batch, config.num_heads, seq, config.rope_dim), f"q_rope shape wrong: {q_rope.shape}"

def test_mla_attention_compute_kv():
    """Test KV computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiheadLatentAttention(config)
    
    assert mla.kv_down is not None, "MLA not initialized"
    
    batch, seq = 2, 16
    x = torch.randn(batch, seq, config.d_model)
    
    c_kv, k_content, k_rope, v = mla.compute_kv(x)
    
    assert c_kv.shape == (batch, seq, config.d_kv_latent), "c_kv shape wrong"
    assert k_content.shape == (batch, config.num_heads, seq, config.head_dim), "k_content shape wrong"
    assert k_rope.shape == (batch, seq, config.num_heads, config.rope_dim), "k_rope shape wrong"
    assert v.shape == (batch, config.num_heads, seq, config.head_dim), "v shape wrong"

def test_mla_attention_forward():
    """Test MLA forward pass produces valid attention."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiheadLatentAttention(config)
    
    assert mla.kv_down is not None, "MLA not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, config.d_model)
    
    output, attn = mla(x)
    
    assert output.shape == x.shape, f"Output shape {output.shape} != {x.shape}"
    
    expected_attn_shape = (batch, config.num_heads, seq, seq)
    assert attn.shape == expected_attn_shape, "Attention shape wrong"
    
    assert output.abs().sum() != 0, "Output is all zeros"
    assert (attn >= 0).all(), "Attention weights have negative values"
    
    attn_sums = attn.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4), "Attention weights don't sum to 1"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Output contains NaN or Inf"

def test_mla_attention_with_cache():
    """Test MLA with caching."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiheadLatentAttention(config)
    
    assert mla.kv_down is not None, "MLA not initialized"
    
    batch = 2
    
    cache = MLAKVCache(batch, 64, config.d_kv_latent,
                      config.num_heads, config.rope_dim)
    
    assert cache.c_kv is not None, "Cache not initialized"
    
    x1 = torch.randn(batch, 8, config.d_model)
    output1, _ = mla(x1, cache=cache, start_pos=0)
    
    assert cache.seq_len == 8, f"Cache length after prefill: {cache.seq_len}"
    
    x2 = torch.randn(batch, 1, config.d_model)
    output2, attn2 = mla(x2, cache=cache, start_pos=8)
    
    assert cache.seq_len == 9, f"Cache length after decode: {cache.seq_len}"
    
    assert attn2.shape == (batch, config.num_heads, 1, 9), f"Decode attention shape wrong: {attn2.shape}"

def test_mla_transformer_block_init():
    """Test MLATransformerBlock initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    block = MLATransformerBlock(config)
    
    assert block.attn is not None, "attn not initialized"
    assert block.attn_norm is not None, "attn_norm not initialized"
    assert block.ffn is not None, "ffn not initialized"
    assert block.ffn_norm is not None, "ffn_norm not initialized"

def test_mla_transformer_block_forward():
    """Test MLATransformerBlock forward with residual verification."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    block = MLATransformerBlock(config)
    
    assert block.attn is not None, "Block not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, config.d_model)
    
    output, attn = block(x)
    
    assert output.shape == x.shape, f"Output shape {output.shape} != {x.shape}"
    
    assert output.abs().sum() != 0, "Output is all zeros"
    
    correlation = F.cosine_similarity(output.flatten(), x.flatten(), dim=0)
    assert correlation >= 0.1, f"Residual connection may not work: correlation={correlation:.3f}"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Output contains NaN or Inf"

def test_mla_model_init():
    """Test MLAModel initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    model = MLAModel(config, num_layers=2)
    
    assert model.layers is not None, "layers not initialized"
    assert len(model.layers) == 2, f"Expected 2 layers, got {len(model.layers)}"
    assert model.final_norm is not None, "final_norm not initialized"

def test_mla_model_forward():
    """Test MLAModel forward through multiple layers."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    model = MLAModel(config, num_layers=2)
    
    assert model.layers is not None, "Model not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, config.d_model)
    
    output, attn_weights = model(x)
    
    assert output.shape == x.shape, "Output shape wrong"
    
    assert output.abs().sum() != 0, "Output is all zeros"
    
    assert len(attn_weights) == 2, f"Expected attention weights from 2 layers, got {len(attn_weights)}"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Output contains NaN or Inf"
    
    output_rms = output.pow(2).mean(dim=-1).sqrt().mean()
    assert output_rms <= 10, f"Output may not be normalized, RMS={output_rms:.2f}"

def test_generation():
    """Test generation with MLA produces valid sequence."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    model = MLAModel(config, num_layers=2)
    
    assert model.layers is not None, "Model not initialized"
    
    batch = 2
    prompt_len = 8
    max_new = 4
    
    torch.manual_seed(42)
    prompt = torch.randn(batch, prompt_len, config.d_model)
    
    all_hidden, caches = generate_with_mla(model, prompt, max_new_tokens=max_new)
    
    expected_len = prompt_len + max_new
    assert all_hidden.shape[1] == expected_len, f"Generated length {all_hidden.shape[1]} != {expected_len}"
    
    assert all_hidden.abs().sum() != 0, "Generated hidden states are all zeros"
    
    assert len(caches) == 2, f"Expected caches for 2 layers, got {len(caches)}"
    
    for layer_idx, cache in caches.items():
        assert cache.seq_len == expected_len, f"Layer {layer_idx} cache length {cache.seq_len} != {expected_len}"
    
    assert not torch.isnan(all_hidden).any() and not torch.isinf(all_hidden).any(), "Generated hidden states contain NaN or Inf"

def test_memory_comparison():
    """Test memory comparison calculation with expected formula."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    seq_len = 1024
    num_layers = 8
    
    mem = compare_memory_usage(config, seq_len=seq_len, num_layers=num_layers)
    
    assert mem['memory_reduction'] != 0, "Memory comparison not computed"
    
    d_kv = config.num_heads * config.head_dim
    std_per_token = 2 * d_kv
    mla_per_token = config.d_kv_latent + config.num_heads * config.rope_dim
    
    expected_reduction = std_per_token / mla_per_token
    
    assert mem['memory_reduction'] >= 1.5, f"Expected >1.5x reduction, got {mem['memory_reduction']:.2f}x"
    
    expected_std_elements = num_layers * seq_len * std_per_token
    assert mem['standard_cache_elements'] == expected_std_elements, f"Standard elements {mem['standard_cache_elements']} != {expected_std_elements}"

def test_gradient_flow():
    """Test gradient flow through MLA model."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    model = MLAModel(config, num_layers=2)
    
    assert model.layers is not None, "Model not initialized"
    
    x = torch.randn(2, 8, config.d_model, requires_grad=True)
    output, _ = model(x)
    
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient to input"
    assert x.grad.abs().sum() != 0, "Gradients are zero"

def test_causal_mask():
    """Test that causal masking works correctly."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiheadLatentAttention(config)
    
    assert mla.kv_down is not None, "MLA not initialized"
    
    batch, seq = 1, 8
    x = torch.randn(batch, seq, config.d_model)
    
    mask = torch.triu(
        torch.ones(seq, seq) * float('-inf'),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)
    
    output, attn = mla(x, mask=mask)
    
    upper_attn = torch.triu(attn[0, 0], diagonal=1)
    assert upper_attn.abs().max() <= 1e-6, "Attention is not causal"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
