"""Test Suite for Day 30: DeepSeek Block"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
try:
    from day30 import (
        DeepSeekBlockConfig, RMSNorm, SwiGLUFFN,
        MultiHeadLatentAttention, DeepSeekBlock,
        RotaryEmbedding, apply_rotary_pos_emb,
        DeepSeekBlockWithRoPE,
        create_causal_mask, create_causal_mask_with_cache,
        count_parameters
    )
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def get_test_config():
    """Get test configuration."""
    return DeepSeekBlockConfig(
        d_model=128,
        num_heads=4,
        num_kv_heads=2,
        latent_dim=32,
        d_ff=344,
        dropout=0.0
    )

def test_rms_norm_init():
    """Test RMSNorm initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim = 64
    norm = RMSNorm(dim)
    
    assert norm.weight is not None, "Weight parameter not initialized"
    assert norm.weight.shape == (dim,), f"Weight shape {norm.weight.shape} != ({dim},)"

def test_rms_norm_forward():
    """Test RMSNorm forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim = 64
    eps = 1e-6
    norm = RMSNorm(dim, eps)
    
    assert norm.weight is not None, "RMSNorm not initialized"
    
    batch, seq_len = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, dim) * 10
    
    out = norm(x)
    
    assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"
    
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    expected = (x / rms) * norm.weight
    
    assert torch.allclose(out, expected, atol=1e-5), "RMSNorm output doesn't match formula: x / sqrt(mean(x²) + eps) * weight"
    
    rms_out = torch.sqrt(torch.mean(out ** 2, dim=-1))
    assert torch.allclose(rms_out, torch.ones_like(rms_out), atol=0.1), f"Output RMS should be ~1, got {rms_out.mean():.2f}"

def test_rms_norm_vs_layer_norm():
    """Test that RMSNorm differs from LayerNorm."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim = 64
    rms_norm = RMSNorm(dim)
    layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
    
    assert rms_norm.weight is not None, "RMSNorm not initialized"
    
    torch.manual_seed(42)
    x = torch.randn(2, 16, dim) + 5
    
    with torch.no_grad():
        rms_norm.weight.fill_(1.0)
    
    rms_out = rms_norm(x)
    ln_out = layer_norm(x)
    
    rms_mean = rms_out.mean(dim=-1)
    ln_mean = ln_out.mean(dim=-1)
    
    assert not torch.allclose(rms_mean, torch.zeros_like(rms_mean), atol=0.1), "RMSNorm should NOT center the data (mean != 0)"
    
    assert torch.allclose(ln_mean, torch.zeros_like(ln_mean), atol=0.01), "LayerNorm should center the data (mean ~0)"
    
    diff = (rms_out - ln_out).abs().mean()
    assert diff >= 0.01, "RMSNorm output too similar to LayerNorm"

def test_swiglu_ffn_init():
    """Test SwiGLU FFN initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_ff = 64, 172
    ffn = SwiGLUFFN(d_model, d_ff)
    
    assert ffn.w_gate is not None, "w_gate not initialized"
    assert ffn.w_up is not None, "w_up not initialized"
    assert ffn.w_down is not None, "w_down not initialized"

def test_swiglu_ffn_forward():
    """Test SwiGLU FFN forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_ff = 64, 172
    ffn = SwiGLUFFN(d_model, d_ff)
    
    assert ffn.w_gate is not None, "SwiGLU FFN not initialized"
    
    batch, seq_len = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    
    out = ffn(x)
    
    assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"
    
    gate = F.silu(ffn.w_gate(x))
    up = ffn.w_up(x)
    hidden = gate * up
    expected = ffn.w_down(hidden)
    
    assert torch.allclose(out, expected, atol=1e-5), "SwiGLU output doesn't match formula: silu(W_gate(x)) * W_up(x) @ W_down"

def test_swiglu_gradient_flow():
    """Test gradient flow through SwiGLU."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, d_ff = 64, 172
    ffn = SwiGLUFFN(d_model, d_ff)
    
    assert ffn.w_gate is not None, "SwiGLU FFN not initialized"
    
    x = torch.randn(2, 16, d_model, requires_grad=True)
    out = ffn(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient to input"
    assert x.grad.abs().sum() != 0, "Zero gradient"

def test_mla_init():
    """Test MLA initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiHeadLatentAttention(config)
    
    assert mla.W_q is not None, "W_q not initialized"
    assert mla.W_kv_compress is not None, "W_kv_compress not initialized"
    assert mla.W_k_expand is not None, "W_k_expand not initialized"
    assert mla.W_v_expand is not None, "W_v_expand not initialized"
    assert mla.W_o is not None, "W_o not initialized"

def test_mla_forward():
    """Test MLA forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiHeadLatentAttention(config)
    
    assert mla.W_q is not None, "MLA not initialized"
    
    batch, seq_len = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, config.d_model)
    
    out, attn_weights, kv_cache = mla(x)
    
    assert out.shape == x.shape, f"Output shape {out.shape} != {x.shape}"
    
    expected_attn = (batch, config.num_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_attn, f"Attention shape {attn_weights.shape} != {expected_attn}"
    
    attn_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5), "Attention weights don't sum to 1"
    
    assert (attn_weights >= -1e-6).all(), "Attention weights should be non-negative"
    
    assert mla.latent_dim < config.num_kv_heads * mla.head_dim, "MLA should use KV compression"

def test_mla_with_mask():
    """Test MLA with causal mask."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiHeadLatentAttention(config)
    
    assert mla.W_q is not None, "MLA not initialized"
    
    batch, seq_len = 2, 16
    x = torch.randn(batch, seq_len, config.d_model)
    
    mask = create_causal_mask(seq_len)
    out, attn_weights, _ = mla(x, mask)
    
    upper_tri = torch.triu(attn_weights[0, 0], diagonal=1)
    assert upper_tri.abs().sum() <= 1e-5, "Attention is not causal"

def test_mla_kv_cache():
    """Test MLA KV caching."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiHeadLatentAttention(config)
    
    assert mla.W_q is not None, "MLA not initialized"
    
    batch = 2
    torch.manual_seed(42)
    
    prompt_len = 8
    x_prompt = torch.randn(batch, prompt_len, config.d_model)
    out1, attn1, kv_cache = mla(x_prompt)
    
    assert kv_cache is not None, "KV cache not returned"
    
    k_cache, v_cache = kv_cache
    
    expected_kv_shape = (batch, config.num_kv_heads, prompt_len, config.d_model // config.num_heads)
    assert k_cache.shape == expected_kv_shape, f"K cache shape {k_cache.shape} != {expected_kv_shape}"
    
    x_new = torch.randn(batch, 1, config.d_model)
    out2, attn2, new_cache = mla(x_new, kv_cache=kv_cache)
    
    assert new_cache is not None, "Cache not updated"
    
    new_k, new_v = new_cache
    assert new_k.shape[2] == prompt_len + 1, f"Cache not extended: {new_k.shape[2]} != {prompt_len + 1}"
    
    assert torch.allclose(new_k[:, :, :prompt_len, :], k_cache, atol=1e-5), "K cache portion not preserved during extension"
    assert torch.allclose(new_v[:, :, :prompt_len, :], v_cache, atol=1e-5), "V cache portion not preserved during extension"

def test_mla_latent_compression():
    """Test that MLA actually compresses KV."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiHeadLatentAttention(config)
    
    assert mla.W_kv_compress is not None, "MLA not initialized"
    
    kv_compress_out = config.latent_dim
    full_kv = config.num_kv_heads * (config.d_model // config.num_heads)
    
    compression_ratio = full_kv / kv_compress_out
    
    assert compression_ratio >= 1, f"No compression: ratio = {compression_ratio}"

def test_deepseek_block_init():
    """Test DeepSeek block initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    block = DeepSeekBlock(config)
    
    assert block.attn_norm is not None, "attn_norm not initialized"
    assert block.attention is not None, "attention not initialized"
    assert block.ffn_norm is not None, "ffn_norm not initialized"
    assert block.ffn is not None, "ffn not initialized"

def test_deepseek_block_forward():
    """Test DeepSeek block forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    block = DeepSeekBlock(config)
    
    assert block.attention is not None, "Block not initialized"
    
    batch, seq_len = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, config.d_model)
    mask = create_causal_mask(seq_len)
    
    out, attn_weights, kv_cache = block(x, mask)
    
    assert out.shape == x.shape, f"Output shape {out.shape} != {x.shape}"
    
    assert not torch.allclose(out, x, atol=1e-3), "Block output too similar to input"
    
    upper_tri = torch.triu(attn_weights[0, 0], diagonal=1)
    assert upper_tri.abs().sum() <= 1e-5, "Attention should be causal"
    
    attn_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-5), "Attention weights should sum to 1"

def test_deepseek_block_residual():
    """Test that block has residual connections."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    block = DeepSeekBlock(config)
    
    assert block.attention is not None, "Block not initialized"
    
    batch, seq_len = 2, 16
    x = torch.randn(batch, seq_len, config.d_model)
    
    with torch.no_grad():
        for name, param in block.named_parameters():
            if 'norm' not in name:
                param.zero_()
    
    out, _, _ = block(x)
    
    diff = (out - x).abs().mean()
    
    assert diff <= 0.1, f"Residual not working: diff = {diff:.4f}"

def test_deepseek_block_gradient():
    """Test gradient flow through DeepSeek block."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    block = DeepSeekBlock(config)
    
    assert block.attention is not None, "Block not initialized"
    
    x = torch.randn(2, 16, config.d_model, requires_grad=True)
    mask = create_causal_mask(16)
    
    out, _, _ = block(x, mask)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient to input"
    assert x.grad.abs().sum() != 0, "Zero gradient"
    
    grad_count = sum(1 for p in block.parameters() if p.grad is not None)
    param_count = sum(1 for p in block.parameters())
    
    assert grad_count == param_count, f"Only {grad_count}/{param_count} params have gradients"

def test_rope_init():
    """Test RoPE initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim = 64
    base = 10000.0
    rope = RotaryEmbedding(dim, base=base)
    
    cos, sin = rope(torch.tensor([0]), seq_len=16)
    
    assert cos.shape == (16, dim // 2), f"cos shape {cos.shape} != (16, {dim // 2})"
    assert sin.shape == (16, dim // 2), f"sin shape {sin.shape} != (16, {dim // 2})"
    
    sum_sq = cos ** 2 + sin ** 2
    assert torch.allclose(sum_sq, torch.ones_like(sum_sq), atol=1e-5), "cos² + sin² should equal 1"
    
    assert abs(cos[0, 0] - 1.0) <= 1e-5, "cos(0) should be 1"
    assert abs(sin[0, 0]) <= 1e-5, "sin(0) should be 0"

def test_apply_rope():
    """Test applying RoPE to Q and K."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, num_heads, seq_len, head_dim = 2, 4, 16, 64
    
    torch.manual_seed(42)
    q = torch.randn(batch, num_heads, seq_len, head_dim)
    k = torch.randn(batch, num_heads // 2, seq_len, head_dim)
    
    rope = RotaryEmbedding(head_dim)
    cos, sin = rope(q, seq_len)
    
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    assert q_rot.shape == q.shape, f"Q shape changed: {q_rot.shape}"
    assert k_rot.shape == k.shape, f"K shape changed: {k_rot.shape}"
    
    assert not torch.allclose(q, q_rot), "RoPE didn't modify Q"
    assert not torch.allclose(k, k_rot), "RoPE didn't modify K"
    
    q_norm = q.norm(dim=-1)
    q_rot_norm = q_rot.norm(dim=-1)
    assert torch.allclose(q_norm, q_rot_norm, atol=1e-4), "RoPE should preserve vector norms"
    
    k_norm = k.norm(dim=-1)
    k_rot_norm = k_rot.norm(dim=-1)
    assert torch.allclose(k_norm, k_rot_norm, atol=1e-4), "RoPE should preserve K vector norms"

def test_causal_mask():
    """Test causal mask creation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len = 8
    mask = create_causal_mask(seq_len)
    
    assert mask.shape == (1, 1, seq_len, seq_len), f"Mask shape {mask.shape} != (1, 1, {seq_len}, {seq_len})"
    
    mask_2d = mask.squeeze()
    assert torch.allclose(mask_2d, torch.tril(torch.ones(seq_len, seq_len))), "Mask is not lower triangular"

def test_causal_mask_with_cache():
    """Test causal mask for cached inference."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    q_len, kv_len = 1, 16
    mask = create_causal_mask_with_cache(q_len, kv_len)
    
    assert mask.shape == (1, 1, q_len, kv_len), f"Mask shape {mask.shape} != (1, 1, {q_len}, {kv_len})"
    
    assert torch.all(mask == 1), "Query should attend to all cached KV"

def test_parameter_count():
    """Test parameter counting."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    block = DeepSeekBlock(config)
    
    assert block.attention is not None, "Block not initialized"
    
    param_count = count_parameters(block)
    
    assert param_count != 0, "No parameters counted"
    
    min_expected = config.d_model * config.d_model
    assert param_count >= min_expected, f"Too few parameters: {param_count}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
