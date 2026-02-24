"""Test Suite for Day 27: MLA Key-Value Compression"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
try:
    from day27 import (
        RMSNorm, KVDownProjection, precompute_freqs_cis, apply_rotary_emb,
        DecoupledRoPEKey, MLAKVCompression, MLAQueryCompression,
        compute_mla_attention_scores
    )
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_rmsnorm_init():
    """Test RMSNorm initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim = 256
    norm = RMSNorm(dim)
    
    assert norm.weight is not None, "weight not initialized"
    assert norm.weight.shape == (dim,), f"weight shape {norm.weight.shape} != ({dim},)"

def test_rmsnorm_forward():
    """Test RMSNorm forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim = 256
    norm = RMSNorm(dim)
    
    assert norm.weight is not None, "norm not initialized"
    
    batch, seq = 2, 16
    x = torch.randn(batch, seq, dim) * 10
    out = norm(x)
    
    assert out.shape == x.shape, f"Output shape {out.shape} != {x.shape}"
    
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1), f"RMS not ~1 after norm: {rms.mean():.4f}"

def test_rmsnorm_scale():
    """Test that RMSNorm weight (scale) affects output."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim = 256
    norm = RMSNorm(dim)
    
    assert norm.weight is not None, "norm not initialized"
    
    x = torch.randn(2, 8, dim)
    
    out1 = norm(x)
    
    with torch.no_grad():
        norm.weight.fill_(2.0)
    out2 = norm(x)
    
    ratio = out2.abs().mean() / out1.abs().mean()
    assert abs(ratio - 2.0) <= 0.1, f"Weight scaling wrong: {ratio:.2f} != 2.0"

def test_kv_down_projection_init():
    """Test KVDownProjection initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent = 512, 128
    proj = KVDownProjection(d_model, d_latent)
    
    assert proj.down_proj is not None, "down_proj not initialized"
    assert proj.norm is not None, "norm not initialized"

def test_kv_down_projection_forward():
    """Test KVDownProjection forward pass applies linear + norm."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent = 512, 128
    proj = KVDownProjection(d_model, d_latent)
    
    assert proj.down_proj is not None, "proj not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, d_model)
    c = proj(x)
    
    expected_shape = (batch, seq, d_latent)
    assert c.shape == expected_shape, f"Output shape {c.shape} != {expected_shape}"
    
    assert c.abs().sum() != 0, "Output is all zeros"
    
    rms = c.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.2), f"RMSNorm may not be applied correctly, mean RMS={rms.mean():.4f}"
    
    compression_ratio = d_model / d_latent
    assert compression_ratio >= 2, f"Expected compression ratio >= 2, got {compression_ratio}"

def test_precompute_freqs():
    """Test RoPE frequency precomputation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim, max_len = 64, 128
    freqs = precompute_freqs_cis(dim, max_len)
    
    expected_shape = (max_len, dim // 2)
    assert freqs.shape == expected_shape, f"Shape {freqs.shape} != {expected_shape}"
    
    assert freqs.abs().sum() != 0, "Frequencies are all zero"
    
    magnitudes = freqs.abs()
    assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5), "Frequencies should have unit magnitude"

def test_apply_rotary_emb():
    """Test rotary embedding application."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim, seq_len = 64, 16
    freqs = precompute_freqs_cis(dim, seq_len * 2)
    
    assert freqs.abs().sum() != 0, "Frequencies not computed"
    
    batch, num_heads = 2, 8
    x = torch.randn(batch, seq_len, num_heads, dim)
    x_rot = apply_rotary_emb(x, freqs[:seq_len])
    
    assert x_rot.shape == x.shape, f"Output shape {x_rot.shape} != {x.shape}"
    
    orig_norm = x.norm()
    rot_norm = x_rot.norm()
    assert abs(orig_norm - rot_norm) / orig_norm <= 0.01, "Rotation changed magnitude significantly"

def test_rotary_position_difference():
    """Test that different positions get different rotations."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim, seq_len = 64, 16
    freqs = precompute_freqs_cis(dim, seq_len * 2)
    
    assert freqs.abs().sum() != 0, "Frequencies not computed"
    
    x = torch.ones(1, seq_len, 1, dim)
    x_rot = apply_rotary_emb(x, freqs[:seq_len])
    
    pos0 = x_rot[0, 0, 0]
    pos1 = x_rot[0, 1, 0]
    
    assert not torch.allclose(pos0, pos1, atol=1e-5), "Different positions should have different rotations"

def test_decoupled_rope_key_init():
    """Test DecoupledRoPEKey initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim, rope_dim = 512, 128, 8, 64, 32
    key_proj = DecoupledRoPEKey(d_model, d_latent, num_heads, head_dim, rope_dim)
    
    assert key_proj.up_proj_k is not None, "up_proj_k not initialized"
    assert key_proj.rope_proj is not None, "rope_proj not initialized"

def test_decoupled_rope_key_forward():
    """Test DecoupledRoPEKey forward pass with content and RoPE verification."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim, rope_dim = 512, 128, 8, 64, 32
    key_proj = DecoupledRoPEKey(d_model, d_latent, num_heads, head_dim, rope_dim)
    
    assert key_proj.up_proj_k is not None, "key_proj not initialized"
    
    batch, seq = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq, d_model)
    c_kv = torch.randn(batch, seq, d_latent)
    
    k_content, k_rope = key_proj(x, c_kv)
    
    assert k_content.shape == (batch, seq, num_heads, head_dim), "k_content shape wrong"
    assert k_rope.shape == (batch, seq, num_heads, rope_dim), "k_rope shape wrong"
    
    with torch.no_grad():
        expected_content = key_proj.up_proj_k(c_kv)
        expected_content = expected_content.view(batch, seq, num_heads, head_dim)
    
    assert torch.allclose(k_content, expected_content, atol=1e-5), "k_content doesn't match expected up projection"
    
    pos0 = k_rope[0, 0, 0]
    pos1 = k_rope[0, 1, 0]
    assert not torch.allclose(pos0, pos1, atol=1e-5), "k_rope should have different values at different positions (RoPE)"
    
    assert k_content.abs().sum() != 0, "k_content is all zeros"
    assert k_rope.abs().sum() != 0, "k_rope is all zeros"

def test_mla_kv_compression_init():
    """Test MLAKVCompression initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim, rope_dim = 512, 128, 8, 64, 32
    mla = MLAKVCompression(d_model, d_latent, num_heads, head_dim, rope_dim)
    
    assert mla.kv_down is not None, "kv_down not initialized"
    assert mla.up_proj_k is not None, "up_proj_k not initialized"
    assert mla.up_proj_v is not None, "up_proj_v not initialized"
    assert mla.rope_proj is not None, "rope_proj not initialized"

def test_mla_kv_compression_compress():
    """Test MLAKVCompression compress method."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim, rope_dim = 512, 128, 8, 64, 32
    mla = MLAKVCompression(d_model, d_latent, num_heads, head_dim, rope_dim)
    
    assert mla.kv_down is not None, "mla not initialized"
    
    batch, seq = 2, 16
    x = torch.randn(batch, seq, d_model)
    c = mla.compress(x)
    
    expected_shape = (batch, seq, d_latent)
    assert c.shape == expected_shape, f"Compressed shape {c.shape} != {expected_shape}"

def test_mla_kv_compression_full_forward():
    """Test MLAKVCompression full forward."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim, rope_dim = 512, 128, 8, 64, 32
    mla = MLAKVCompression(d_model, d_latent, num_heads, head_dim, rope_dim)
    
    assert mla.kv_down is not None, "mla not initialized"
    
    batch, seq = 2, 16
    x = torch.randn(batch, seq, d_model)
    c_kv, k_content, k_rope, v = mla(x)
    
    assert c_kv.shape == (batch, seq, d_latent), "c_kv shape wrong"
    assert k_content.shape == (batch, seq, num_heads, head_dim), "k_content shape wrong"
    assert k_rope.shape == (batch, seq, num_heads, rope_dim), "k_rope shape wrong"
    assert v.shape == (batch, seq, num_heads, head_dim), "v shape wrong"

def test_mla_query_compression_init():
    """Test MLAQueryCompression initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_q_latent, num_heads, head_dim, rope_dim = 512, 96, 8, 64, 32
    q_comp = MLAQueryCompression(d_model, d_q_latent, num_heads, head_dim, rope_dim)
    
    assert q_comp.q_down is not None, "q_down not initialized"
    assert q_comp.q_up is not None, "q_up not initialized"
    assert q_comp.q_rope_proj is not None, "q_rope_proj not initialized"

def test_mla_query_compression_forward():
    """Test MLAQueryCompression forward."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_q_latent, num_heads, head_dim, rope_dim = 512, 96, 8, 64, 32
    q_comp = MLAQueryCompression(d_model, d_q_latent, num_heads, head_dim, rope_dim)
    
    assert q_comp.q_down is not None, "q_comp not initialized"
    
    batch, seq = 2, 16
    x = torch.randn(batch, seq, d_model)
    q_content, q_rope = q_comp(x)
    
    assert q_content.shape == (batch, seq, num_heads, head_dim), "q_content shape wrong"
    assert q_rope.shape == (batch, seq, num_heads, rope_dim), "q_rope shape wrong"

def test_mla_attention_scores():
    """Test combined MLA attention score computation matches formula."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, num_heads, seq_q, seq_k = 2, 8, 16, 24
    head_dim, rope_dim = 64, 32
    
    torch.manual_seed(42)
    q_content = torch.randn(batch, num_heads, seq_q, head_dim)
    q_rope = torch.randn(batch, num_heads, seq_q, rope_dim)
    k_content = torch.randn(batch, num_heads, seq_k, head_dim)
    k_rope = torch.randn(batch, num_heads, seq_k, rope_dim)
    
    scores = compute_mla_attention_scores(q_content, q_rope, k_content, k_rope)
    
    expected_shape = (batch, num_heads, seq_q, seq_k)
    assert scores.shape == expected_shape, f"Scores shape {scores.shape} != {expected_shape}"
    
    assert scores.abs().sum() != 0, "Scores are all zero"
    
    total_dim = head_dim + rope_dim
    scale = total_dim ** -0.5
    
    content_scores = torch.matmul(q_content, k_content.transpose(-2, -1))
    rope_scores = torch.matmul(q_rope, k_rope.transpose(-2, -1))
    expected_scores = (content_scores + rope_scores) * scale
    
    assert torch.allclose(scores, expected_scores, atol=1e-4), "Scores don't match expected formula"
    
    assert not torch.isnan(scores).any() and not torch.isinf(scores).any(), "Scores contain NaN or Inf"

def test_gradient_flow():
    """Test gradient flow through MLA compression."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_latent, num_heads, head_dim, rope_dim = 256, 64, 4, 32, 16
    mla = MLAKVCompression(d_model, d_latent, num_heads, head_dim, rope_dim)
    
    assert mla.kv_down is not None, "mla not initialized"
    
    x = torch.randn(2, 8, d_model, requires_grad=True)
    c_kv, k_content, k_rope, v = mla(x)
    
    loss = c_kv.sum() + k_content.sum() + k_rope.sum() + v.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient to input"
    assert x.grad.abs().sum() != 0, "Gradients are zero"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
