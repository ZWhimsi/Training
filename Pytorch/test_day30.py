"""Test Suite for Day 30: DeepSeek Block"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

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


def test_rms_norm_init() -> Tuple[bool, str]:
    """Test RMSNorm initialization."""
    try:
        dim = 64
        norm = RMSNorm(dim)
        
        if norm.weight is None:
            return False, "Weight parameter not initialized"
        if norm.weight.shape != (dim,):
            return False, f"Weight shape {norm.weight.shape} != ({dim},)"
        
        return True, "RMSNorm initialized correctly"
    except Exception as e:
        return False, str(e)


def test_rms_norm_forward() -> Tuple[bool, str]:
    """Test RMSNorm forward pass."""
    try:
        dim = 64
        norm = RMSNorm(dim)
        
        if norm.weight is None:
            return False, "RMSNorm not initialized"
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, dim) * 10  # Large values
        
        out = norm(x)
        
        if out.shape != x.shape:
            return False, f"Output shape {out.shape} != input shape {x.shape}"
        
        # Check normalization effect
        rms_in = torch.sqrt(torch.mean(x ** 2, dim=-1))
        rms_out = torch.sqrt(torch.mean(out ** 2, dim=-1))
        
        # Output should be more normalized (closer to 1)
        if rms_out.mean() > rms_in.mean():
            return False, "RMSNorm didn't reduce variance"
        
        return True, f"RMSNorm works, RMS: {rms_in.mean():.2f} -> {rms_out.mean():.2f}"
    except Exception as e:
        return False, str(e)


def test_rms_norm_vs_layer_norm() -> Tuple[bool, str]:
    """Test that RMSNorm differs from LayerNorm."""
    try:
        dim = 64
        rms_norm = RMSNorm(dim)
        layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        
        if rms_norm.weight is None:
            return False, "RMSNorm not initialized"
        
        x = torch.randn(2, 16, dim) + 5  # Non-zero mean
        
        rms_out = rms_norm(x)
        ln_out = layer_norm(x)
        
        # They should produce different outputs (RMS doesn't center)
        diff = (rms_out - ln_out).abs().mean()
        
        if diff < 0.01:
            return False, "RMSNorm output too similar to LayerNorm"
        
        return True, f"RMSNorm differs from LayerNorm by {diff:.3f}"
    except Exception as e:
        return False, str(e)


def test_swiglu_ffn_init() -> Tuple[bool, str]:
    """Test SwiGLU FFN initialization."""
    try:
        d_model, d_ff = 64, 172
        ffn = SwiGLUFFN(d_model, d_ff)
        
        if ffn.w_gate is None:
            return False, "w_gate not initialized"
        if ffn.w_up is None:
            return False, "w_up not initialized"
        if ffn.w_down is None:
            return False, "w_down not initialized"
        
        return True, "SwiGLU FFN initialized"
    except Exception as e:
        return False, str(e)


def test_swiglu_ffn_forward() -> Tuple[bool, str]:
    """Test SwiGLU FFN forward pass."""
    try:
        d_model, d_ff = 64, 172
        ffn = SwiGLUFFN(d_model, d_ff)
        
        if ffn.w_gate is None:
            return False, "SwiGLU FFN not initialized"
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, d_model)
        
        out = ffn(x)
        
        if out.shape != x.shape:
            return False, f"Output shape {out.shape} != input shape {x.shape}"
        
        return True, f"SwiGLU FFN output shape: {out.shape}"
    except Exception as e:
        return False, str(e)


def test_swiglu_gradient_flow() -> Tuple[bool, str]:
    """Test gradient flow through SwiGLU."""
    try:
        d_model, d_ff = 64, 172
        ffn = SwiGLUFFN(d_model, d_ff)
        
        if ffn.w_gate is None:
            return False, "SwiGLU FFN not initialized"
        
        x = torch.randn(2, 16, d_model, requires_grad=True)
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "No gradient to input"
        if x.grad.abs().sum() == 0:
            return False, "Zero gradient"
        
        return True, "Gradient flows through SwiGLU"
    except Exception as e:
        return False, str(e)


def test_mla_init() -> Tuple[bool, str]:
    """Test MLA initialization."""
    try:
        config = get_test_config()
        mla = MultiHeadLatentAttention(config)
        
        if mla.W_q is None:
            return False, "W_q not initialized"
        if mla.W_kv_compress is None:
            return False, "W_kv_compress not initialized"
        if mla.W_k_expand is None:
            return False, "W_k_expand not initialized"
        if mla.W_v_expand is None:
            return False, "W_v_expand not initialized"
        if mla.W_o is None:
            return False, "W_o not initialized"
        
        return True, "MLA initialized correctly"
    except Exception as e:
        return False, str(e)


def test_mla_forward() -> Tuple[bool, str]:
    """Test MLA forward pass."""
    try:
        config = get_test_config()
        mla = MultiHeadLatentAttention(config)
        
        if mla.W_q is None:
            return False, "MLA not initialized"
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        out, attn_weights, kv_cache = mla(x)
        
        if out.shape != x.shape:
            return False, f"Output shape {out.shape} != {x.shape}"
        
        expected_attn = (batch, config.num_heads, seq_len, seq_len)
        if attn_weights.shape != expected_attn:
            return False, f"Attention shape {attn_weights.shape} != {expected_attn}"
        
        return True, f"MLA forward: {x.shape} -> {out.shape}"
    except Exception as e:
        return False, str(e)


def test_mla_with_mask() -> Tuple[bool, str]:
    """Test MLA with causal mask."""
    try:
        config = get_test_config()
        mla = MultiHeadLatentAttention(config)
        
        if mla.W_q is None:
            return False, "MLA not initialized"
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        mask = create_causal_mask(seq_len)
        out, attn_weights, _ = mla(x, mask)
        
        # Check causal pattern in attention
        upper_tri = torch.triu(attn_weights[0, 0], diagonal=1)
        if upper_tri.abs().sum() > 1e-5:
            return False, "Attention is not causal"
        
        return True, "MLA respects causal mask"
    except Exception as e:
        return False, str(e)


def test_mla_kv_cache() -> Tuple[bool, str]:
    """Test MLA KV caching."""
    try:
        config = get_test_config()
        mla = MultiHeadLatentAttention(config)
        
        if mla.W_q is None:
            return False, "MLA not initialized"
        
        batch = 2
        
        # Prefill with prompt
        prompt_len = 8
        x_prompt = torch.randn(batch, prompt_len, config.d_model)
        out1, _, kv_cache = mla(x_prompt)
        
        if kv_cache is None:
            return False, "KV cache not returned"
        
        k_cache, v_cache = kv_cache
        
        # Check cache shapes
        expected_kv_shape = (batch, config.num_kv_heads, prompt_len, config.d_model // config.num_heads)
        if k_cache.shape != expected_kv_shape:
            return False, f"K cache shape {k_cache.shape} != {expected_kv_shape}"
        
        # Generate with cache
        x_new = torch.randn(batch, 1, config.d_model)
        out2, _, new_cache = mla(x_new, kv_cache=kv_cache)
        
        if new_cache is None:
            return False, "Cache not updated"
        
        new_k, new_v = new_cache
        if new_k.shape[2] != prompt_len + 1:
            return False, f"Cache not extended: {new_k.shape[2]} != {prompt_len + 1}"
        
        return True, "KV caching works correctly"
    except Exception as e:
        return False, str(e)


def test_mla_latent_compression() -> Tuple[bool, str]:
    """Test that MLA actually compresses KV."""
    try:
        config = get_test_config()
        mla = MultiHeadLatentAttention(config)
        
        if mla.W_kv_compress is None:
            return False, "MLA not initialized"
        
        # Check compression dimensions
        kv_compress_out = config.latent_dim
        full_kv = config.num_kv_heads * (config.d_model // config.num_heads)
        
        compression_ratio = full_kv / kv_compress_out
        
        if compression_ratio <= 1:
            return False, f"No compression: ratio = {compression_ratio}"
        
        return True, f"KV compression ratio: {compression_ratio:.1f}x"
    except Exception as e:
        return False, str(e)


def test_deepseek_block_init() -> Tuple[bool, str]:
    """Test DeepSeek block initialization."""
    try:
        config = get_test_config()
        block = DeepSeekBlock(config)
        
        if block.attn_norm is None:
            return False, "attn_norm not initialized"
        if block.attention is None:
            return False, "attention not initialized"
        if block.ffn_norm is None:
            return False, "ffn_norm not initialized"
        if block.ffn is None:
            return False, "ffn not initialized"
        
        return True, "DeepSeek block initialized"
    except Exception as e:
        return False, str(e)


def test_deepseek_block_forward() -> Tuple[bool, str]:
    """Test DeepSeek block forward pass."""
    try:
        config = get_test_config()
        block = DeepSeekBlock(config)
        
        if block.attention is None:
            return False, "Block not initialized"
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        mask = create_causal_mask(seq_len)
        
        out, attn_weights, kv_cache = block(x, mask)
        
        if out.shape != x.shape:
            return False, f"Output shape {out.shape} != {x.shape}"
        
        return True, f"Block output: {out.shape}"
    except Exception as e:
        return False, str(e)


def test_deepseek_block_residual() -> Tuple[bool, str]:
    """Test that block has residual connections."""
    try:
        config = get_test_config()
        block = DeepSeekBlock(config)
        
        if block.attention is None:
            return False, "Block not initialized"
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, config.d_model)
        
        # Zero out weights to check residual
        with torch.no_grad():
            for name, param in block.named_parameters():
                if 'norm' not in name:
                    param.zero_()
        
        out, _, _ = block(x)
        
        # Output should equal input if all non-norm weights are zero
        # (because of residual connections)
        diff = (out - x).abs().mean()
        
        if diff > 0.1:
            return False, f"Residual not working: diff = {diff:.4f}"
        
        return True, "Residual connections verified"
    except Exception as e:
        return False, str(e)


def test_deepseek_block_gradient() -> Tuple[bool, str]:
    """Test gradient flow through DeepSeek block."""
    try:
        config = get_test_config()
        block = DeepSeekBlock(config)
        
        if block.attention is None:
            return False, "Block not initialized"
        
        x = torch.randn(2, 16, config.d_model, requires_grad=True)
        mask = create_causal_mask(16)
        
        out, _, _ = block(x, mask)
        loss = out.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "No gradient to input"
        if x.grad.abs().sum() == 0:
            return False, "Zero gradient"
        
        # Check gradients flow to all parameters
        grad_count = sum(1 for p in block.parameters() if p.grad is not None)
        param_count = sum(1 for p in block.parameters())
        
        if grad_count < param_count:
            return False, f"Only {grad_count}/{param_count} params have gradients"
        
        return True, f"Gradients flow to all {param_count} parameters"
    except Exception as e:
        return False, str(e)


def test_rope_init() -> Tuple[bool, str]:
    """Test RoPE initialization."""
    try:
        dim = 64
        rope = RotaryEmbedding(dim)
        
        cos, sin = rope(torch.tensor([0]), seq_len=16)
        
        if cos.shape != (16, dim // 2):
            return False, f"cos shape {cos.shape} != (16, {dim // 2})"
        if sin.shape != (16, dim // 2):
            return False, f"sin shape {sin.shape} != (16, {dim // 2})"
        
        return True, f"RoPE initialized: cos/sin shape {cos.shape}"
    except Exception as e:
        return False, str(e)


def test_apply_rope() -> Tuple[bool, str]:
    """Test applying RoPE to Q and K."""
    try:
        batch, num_heads, seq_len, head_dim = 2, 4, 16, 64
        
        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads // 2, seq_len, head_dim)
        
        rope = RotaryEmbedding(head_dim)
        cos, sin = rope(q, seq_len)
        
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        if q_rot.shape != q.shape:
            return False, f"Q shape changed: {q_rot.shape}"
        if k_rot.shape != k.shape:
            return False, f"K shape changed: {k_rot.shape}"
        
        # Rotated should be different from original
        if torch.allclose(q, q_rot):
            return False, "RoPE didn't modify Q"
        if torch.allclose(k, k_rot):
            return False, "RoPE didn't modify K"
        
        return True, "RoPE applied to Q and K"
    except Exception as e:
        return False, str(e)


def test_causal_mask() -> Tuple[bool, str]:
    """Test causal mask creation."""
    try:
        seq_len = 8
        mask = create_causal_mask(seq_len)
        
        if mask.shape != (1, 1, seq_len, seq_len):
            return False, f"Mask shape {mask.shape} != (1, 1, {seq_len}, {seq_len})"
        
        # Check lower triangular
        mask_2d = mask.squeeze()
        if not torch.allclose(mask_2d, torch.tril(torch.ones(seq_len, seq_len))):
            return False, "Mask is not lower triangular"
        
        return True, "Causal mask created correctly"
    except Exception as e:
        return False, str(e)


def test_causal_mask_with_cache() -> Tuple[bool, str]:
    """Test causal mask for cached inference."""
    try:
        q_len, kv_len = 1, 16  # Single query, multiple cached KV
        mask = create_causal_mask_with_cache(q_len, kv_len)
        
        if mask.shape != (1, 1, q_len, kv_len):
            return False, f"Mask shape {mask.shape} != (1, 1, {q_len}, {kv_len})"
        
        # Single query should attend to all previous KV
        if not torch.all(mask == 1):
            return False, "Query should attend to all cached KV"
        
        return True, "Cached mask created correctly"
    except Exception as e:
        return False, str(e)


def test_parameter_count() -> Tuple[bool, str]:
    """Test parameter counting."""
    try:
        config = get_test_config()
        block = DeepSeekBlock(config)
        
        if block.attention is None:
            return False, "Block not initialized"
        
        param_count = count_parameters(block)
        
        if param_count == 0:
            return False, "No parameters counted"
        
        # Rough check: block should have substantial params
        min_expected = config.d_model * config.d_model  # At least attention params
        if param_count < min_expected:
            return False, f"Too few parameters: {param_count}"
        
        return True, f"Block has {param_count:,} parameters"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("rms_norm_init", test_rms_norm_init),
        ("rms_norm_forward", test_rms_norm_forward),
        ("rms_norm_vs_layer_norm", test_rms_norm_vs_layer_norm),
        ("swiglu_ffn_init", test_swiglu_ffn_init),
        ("swiglu_ffn_forward", test_swiglu_ffn_forward),
        ("swiglu_gradient_flow", test_swiglu_gradient_flow),
        ("mla_init", test_mla_init),
        ("mla_forward", test_mla_forward),
        ("mla_with_mask", test_mla_with_mask),
        ("mla_kv_cache", test_mla_kv_cache),
        ("mla_latent_compression", test_mla_latent_compression),
        ("deepseek_block_init", test_deepseek_block_init),
        ("deepseek_block_forward", test_deepseek_block_forward),
        ("deepseek_block_residual", test_deepseek_block_residual),
        ("deepseek_block_gradient", test_deepseek_block_gradient),
        ("rope_init", test_rope_init),
        ("apply_rope", test_apply_rope),
        ("causal_mask", test_causal_mask),
        ("causal_mask_with_cache", test_causal_mask_with_cache),
        ("parameter_count", test_parameter_count),
    ]
    
    print(f"\n{'='*60}")
    print("Day 30: DeepSeek Block - Tests")
    print(f"{'='*60}")
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        status = "PASS" if p else "FAIL"
        print(f"  [{status}] {name}: {m}")
    
    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{len(tests)} tests passed")
    print(f"{'='*60}")
    
    return passed == len(tests)


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
