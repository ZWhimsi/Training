"""Test Suite for Day 26: Multi-head Latent Attention (MLA) Basics"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
try:
    from day26 import (
        calculate_kv_cache_size, calculate_mla_cache_size,
        DownProjection, UpProjection, LowRankKVProjection,
        analyze_compression, BasicMLAAttention, measure_reconstruction_error
    )
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_kv_cache_size_calculation():
    """Test standard KV cache size calculation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = calculate_kv_cache_size(
        num_layers=32, seq_len=32768, num_heads=32, head_dim=128
    )
    
    assert result['total_bytes'] != 0, "Cache size not computed"
    
    expected_gb = 2 * 32 * 32768 * 32 * 128 * 2 / (1024**3)
    assert abs(result['total_gb'] - expected_gb) <= 0.1, f"Expected ~{expected_gb:.1f}GB, got {result['total_gb']:.1f}GB"

def test_mla_cache_size_calculation():
    """Test MLA cache size calculation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = calculate_mla_cache_size(
        num_layers=32, seq_len=32768, d_latent=512
    )
    
    assert result['total_bytes'] != 0, "MLA cache size not computed"
    
    std_result = calculate_kv_cache_size(32, 32768, 32, 128)
    if std_result['total_bytes'] > 0:
        ratio = std_result['total_bytes'] / result['total_bytes']
        assert ratio >= 5, f"MLA should be >5x smaller, got {ratio:.1f}x"

def test_down_projection_init():
    """Test DownProjection initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent = 256, 64
    proj = DownProjection(d_model, d_latent)
    
    assert proj.down_proj is not None, "down_proj not initialized"
    assert proj.down_proj.weight.shape == (d_latent, d_model), f"Weight shape wrong: {proj.down_proj.weight.shape}"

def test_down_projection_forward():
    """Test DownProjection forward pass computes linear projection."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent = 256, 64
    proj = DownProjection(d_model, d_latent)
    
    assert proj.down_proj is not None, "down_proj not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, d_model)
    out = proj(x)
    
    expected_shape = (batch, seq, d_latent)
    assert out.shape == expected_shape, f"Output shape {out.shape} != {expected_shape}"
    
    with torch.no_grad():
        expected = F.linear(x, proj.down_proj.weight, 
                           proj.down_proj.bias if proj.down_proj.bias is not None else None)
    
    assert torch.allclose(out, expected, atol=1e-5), "Output doesn't match expected linear projection"
    assert out.abs().sum() != 0, "Output is all zeros"

def test_up_projection_init():
    """Test UpProjection initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_latent, d_output = 64, 256
    proj = UpProjection(d_latent, d_output)
    
    assert proj.up_proj is not None, "up_proj not initialized"

def test_up_projection_forward():
    """Test UpProjection forward pass computes linear projection."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_latent, d_output = 64, 256
    proj = UpProjection(d_latent, d_output)
    
    assert proj.up_proj is not None, "up_proj not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    c = torch.randn(batch, seq, d_latent)
    out = proj(c)
    
    expected_shape = (batch, seq, d_output)
    assert out.shape == expected_shape, f"Output shape {out.shape} != {expected_shape}"
    
    with torch.no_grad():
        expected = F.linear(c, proj.up_proj.weight,
                           proj.up_proj.bias if proj.up_proj.bias is not None else None)
    
    assert torch.allclose(out, expected, atol=1e-5), "Output doesn't match expected linear projection"
    assert out.abs().sum() != 0, "Output is all zeros"

def test_low_rank_kv_projection_init():
    """Test LowRankKVProjection initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
    proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
    
    assert proj.down_proj is not None, "down_proj not initialized"
    assert proj.up_proj_k is not None, "up_proj_k not initialized"
    assert proj.up_proj_v is not None, "up_proj_v not initialized"

def test_low_rank_kv_compress():
    """Test KV compression computes correct linear transformation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
    proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
    
    assert proj.down_proj is not None, "Projection not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, d_model)
    c = proj.compress(x)
    
    expected_shape = (batch, seq, d_latent)
    assert c.shape == expected_shape, f"Compressed shape {c.shape} != {expected_shape}"
    
    with torch.no_grad():
        expected = F.linear(x, proj.down_proj.weight,
                           proj.down_proj.bias if proj.down_proj.bias is not None else None)
    
    assert torch.allclose(c, expected, atol=1e-5), "Compressed output doesn't match linear projection"
    
    input_size = x.numel()
    compressed_size = c.numel()
    compression_ratio = input_size / compressed_size
    assert compression_ratio >= 1.5, f"Compression ratio {compression_ratio:.2f} too low"

def test_low_rank_kv_reconstruct():
    """Test K, V reconstruction from latent."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
    proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
    
    assert proj.down_proj is not None, "Projection not initialized"
    
    batch, seq = 2, 16
    c = torch.randn(batch, seq, d_latent)
    k, v = proj.reconstruct_kv(c)
    
    expected_shape = (batch, seq, num_heads, head_dim)
    assert k.shape == expected_shape, f"K shape {k.shape} != {expected_shape}"
    assert v.shape == expected_shape, f"V shape {v.shape} != {expected_shape}"

def test_low_rank_kv_full_forward():
    """Test full forward pass of LowRankKVProjection."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
    proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
    
    assert proj.down_proj is not None, "Projection not initialized"
    
    batch, seq = 2, 16
    x = torch.randn(batch, seq, d_model)
    c, k, v = proj(x)
    
    assert c.shape == (batch, seq, d_latent), "Latent shape wrong"
    assert k.shape == (batch, seq, num_heads, head_dim), "K shape wrong"
    assert v.shape == (batch, seq, num_heads, head_dim), "V shape wrong"

def test_compression_analysis():
    """Test compression ratio analysis."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim = 4096, 512, 32, 128
    result = analyze_compression(d_model, d_latent, num_heads, head_dim)
    
    assert result['compression_ratio'] != 0, "Compression ratio not computed"
    
    expected_ratio = 2 * num_heads * head_dim / d_latent
    assert abs(result['compression_ratio'] - expected_ratio) <= 0.1, f"Expected ratio ~{expected_ratio}, got {result['compression_ratio']}"

def test_basic_mla_attention_init():
    """Test BasicMLAAttention initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
    attn = BasicMLAAttention(d_model, d_latent, num_heads, head_dim)
    
    assert attn.W_q is not None, "W_q not initialized"
    assert attn.kv_proj is not None, "kv_proj not initialized"
    assert attn.W_o is not None, "W_o not initialized"

def test_basic_mla_attention_forward():
    """Test BasicMLAAttention forward pass produces valid attention."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
    attn = BasicMLAAttention(d_model, d_latent, num_heads, head_dim)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, d_model)
    output, weights, _ = attn(x)
    
    assert output.shape == x.shape, f"Output shape {output.shape} != {x.shape}"
    
    expected_weights_shape = (batch, num_heads, seq, seq)
    assert weights.shape == expected_weights_shape, "Weights shape wrong"
    
    assert output.abs().sum() != 0, "Output is all zeros"
    assert (weights >= 0).all(), "Attention weights have negative values"
    
    attn_sums = weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4), "Attention weights don't sum to 1"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Output contains NaN or Inf"

def test_basic_mla_attention_cache():
    """Test BasicMLAAttention with caching."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
    attn = BasicMLAAttention(d_model, d_latent, num_heads, head_dim)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    batch, seq = 2, 8
    x = torch.randn(batch, seq, d_model)
    
    output1, _, cache = attn(x, use_cache=True)
    
    assert cache is not None, "Cache not returned"
    assert cache.shape == (batch, seq, d_latent), f"Cache shape {cache.shape} != {(batch, seq, d_latent)}"

def test_basic_mla_attention_incremental():
    """Test incremental generation with cache."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
    attn = BasicMLAAttention(d_model, d_latent, num_heads, head_dim)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    batch = 2
    
    x = torch.randn(batch, 4, d_model)
    _, _, cache = attn(x, use_cache=True)
    
    assert cache is not None, "Initial cache not created"
    
    new_token = torch.randn(batch, 1, d_model)
    output, _, new_cache = attn(new_token, kv_cache=cache, use_cache=True)
    
    assert output.shape == (batch, 1, d_model), "Incremental output shape wrong"
    assert new_cache is not None, "Updated cache not returned"
    
    expected_cache_len = 4 + 1
    assert new_cache.shape[1] == expected_cache_len, f"Cache length {new_cache.shape[1]} != {expected_cache_len}"

def test_gradient_flow():
    """Test gradient flow through MLA."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
    attn = BasicMLAAttention(d_model, d_latent, num_heads, head_dim)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    x = torch.randn(2, 8, d_model, requires_grad=True)
    output, _, _ = attn(x)
    
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient to input"
    assert x.grad.abs().sum() != 0, "Gradients are zero"

def test_compression_vs_standard_equivalence():
    """Test that low-rank can approximate full-rank."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim = 64, 32, 4, 16
    d_kv = num_heads * head_dim
    
    proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
    
    assert proj.down_proj is not None, "Projection not initialized"
    
    with torch.no_grad():
        full_W_k = proj.down_proj.weight.T @ proj.up_proj_k.weight.T
        
    batch, seq = 2, 8
    x = torch.randn(batch, seq, d_model)
    
    c, k, v = proj(x)
    k_flat = k.view(batch, seq, -1)
    
    k_full = F.linear(x, full_W_k.T)
    
    error = (k_flat - k_full).abs().max()
    assert error <= 1e-4, f"Factorization error too large: {error}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
