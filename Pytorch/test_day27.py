"""Test Suite for Day 27: MLA Key-Value Compression"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day27 import (
        RMSNorm, KVDownProjection, precompute_freqs_cis, apply_rotary_emb,
        DecoupledRoPEKey, MLAKVCompression, MLAQueryCompression,
        compute_mla_attention_scores
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_rmsnorm_init() -> Tuple[bool, str]:
    """Test RMSNorm initialization."""
    try:
        dim = 256
        norm = RMSNorm(dim)
        
        if norm.weight is None:
            return False, "weight not initialized"
        
        if norm.weight.shape != (dim,):
            return False, f"weight shape {norm.weight.shape} != ({dim},)"
        
        return True, "RMSNorm initialized correctly"
    except Exception as e:
        return False, str(e)


def test_rmsnorm_forward() -> Tuple[bool, str]:
    """Test RMSNorm forward pass."""
    try:
        dim = 256
        norm = RMSNorm(dim)
        
        if norm.weight is None:
            return False, "norm not initialized"
        
        batch, seq = 2, 16
        x = torch.randn(batch, seq, dim) * 10  # Large values
        out = norm(x)
        
        if out.shape != x.shape:
            return False, f"Output shape {out.shape} != {x.shape}"
        
        # Check normalization: RMS should be approximately 1
        rms = out.pow(2).mean(dim=-1).sqrt()
        if not torch.allclose(rms, torch.ones_like(rms), atol=0.1):
            return False, f"RMS not ~1 after norm: {rms.mean():.4f}"
        
        return True, f"RMSNorm working, mean RMS: {rms.mean():.4f}"
    except Exception as e:
        return False, str(e)


def test_rmsnorm_scale() -> Tuple[bool, str]:
    """Test that RMSNorm weight (scale) affects output."""
    try:
        dim = 256
        norm = RMSNorm(dim)
        
        if norm.weight is None:
            return False, "norm not initialized"
        
        x = torch.randn(2, 8, dim)
        
        # Default weight is 1
        out1 = norm(x)
        
        # Change weight
        with torch.no_grad():
            norm.weight.fill_(2.0)
        out2 = norm(x)
        
        # Output should scale by 2
        ratio = out2.abs().mean() / out1.abs().mean()
        if abs(ratio - 2.0) > 0.1:
            return False, f"Weight scaling wrong: {ratio:.2f} != 2.0"
        
        return True, "Weight scaling works correctly"
    except Exception as e:
        return False, str(e)


def test_kv_down_projection_init() -> Tuple[bool, str]:
    """Test KVDownProjection initialization."""
    try:
        d_model, d_latent = 512, 128
        proj = KVDownProjection(d_model, d_latent)
        
        if proj.down_proj is None:
            return False, "down_proj not initialized"
        if proj.norm is None:
            return False, "norm not initialized"
        
        return True, "KVDownProjection initialized"
    except Exception as e:
        return False, str(e)


def test_kv_down_projection_forward() -> Tuple[bool, str]:
    """Test KVDownProjection forward pass applies linear + norm."""
    try:
        d_model, d_latent = 512, 128
        proj = KVDownProjection(d_model, d_latent)
        
        if proj.down_proj is None:
            return False, "proj not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, d_model)
        c = proj(x)
        
        expected_shape = (batch, seq, d_latent)
        if c.shape != expected_shape:
            return False, f"Output shape {c.shape} != {expected_shape}"
        
        # Verify output is not zeros
        if c.abs().sum() == 0:
            return False, "Output is all zeros"
        
        # Verify normalization was applied (RMS should be approximately 1)
        rms = c.pow(2).mean(dim=-1).sqrt()
        if not torch.allclose(rms, torch.ones_like(rms), atol=0.2):
            return False, f"RMSNorm may not be applied correctly, mean RMS={rms.mean():.4f}"
        
        # Verify compression happened
        compression_ratio = d_model / d_latent
        if compression_ratio < 2:
            return False, f"Expected compression ratio >= 2, got {compression_ratio}"
        
        return True, f"KVDownProjection works, RMS={rms.mean():.4f}"
    except Exception as e:
        return False, str(e)


def test_precompute_freqs() -> Tuple[bool, str]:
    """Test RoPE frequency precomputation."""
    try:
        dim, max_len = 64, 128
        freqs = precompute_freqs_cis(dim, max_len)
        
        expected_shape = (max_len, dim // 2)
        if freqs.shape != expected_shape:
            return False, f"Shape {freqs.shape} != {expected_shape}"
        
        if freqs.abs().sum() == 0:
            return False, "Frequencies are all zero"
        
        # Check that frequencies are unit magnitude (complex on unit circle)
        magnitudes = freqs.abs()
        if not torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5):
            return False, "Frequencies should have unit magnitude"
        
        return True, f"Frequencies shape: {freqs.shape}"
    except Exception as e:
        return False, str(e)


def test_apply_rotary_emb() -> Tuple[bool, str]:
    """Test rotary embedding application."""
    try:
        dim, seq_len = 64, 16
        freqs = precompute_freqs_cis(dim, seq_len * 2)
        
        if freqs.abs().sum() == 0:
            return False, "Frequencies not computed"
        
        batch, num_heads = 2, 8
        x = torch.randn(batch, seq_len, num_heads, dim)
        x_rot = apply_rotary_emb(x, freqs[:seq_len])
        
        if x_rot.shape != x.shape:
            return False, f"Output shape {x_rot.shape} != {x.shape}"
        
        # Rotation should preserve magnitude
        orig_norm = x.norm()
        rot_norm = x_rot.norm()
        if abs(orig_norm - rot_norm) / orig_norm > 0.01:
            return False, f"Rotation changed magnitude significantly"
        
        return True, f"Rotary embedding applied, shape: {x_rot.shape}"
    except Exception as e:
        return False, str(e)


def test_rotary_position_difference() -> Tuple[bool, str]:
    """Test that different positions get different rotations."""
    try:
        dim, seq_len = 64, 16
        freqs = precompute_freqs_cis(dim, seq_len * 2)
        
        if freqs.abs().sum() == 0:
            return False, "Frequencies not computed"
        
        # Same vector at different positions
        x = torch.ones(1, seq_len, 1, dim)
        x_rot = apply_rotary_emb(x, freqs[:seq_len])
        
        # Different positions should have different rotations
        pos0 = x_rot[0, 0, 0]
        pos1 = x_rot[0, 1, 0]
        
        if torch.allclose(pos0, pos1, atol=1e-5):
            return False, "Different positions should have different rotations"
        
        return True, "Position-dependent rotation verified"
    except Exception as e:
        return False, str(e)


def test_decoupled_rope_key_init() -> Tuple[bool, str]:
    """Test DecoupledRoPEKey initialization."""
    try:
        d_model, d_latent, num_heads, head_dim, rope_dim = 512, 128, 8, 64, 32
        key_proj = DecoupledRoPEKey(d_model, d_latent, num_heads, head_dim, rope_dim)
        
        if key_proj.up_proj_k is None:
            return False, "up_proj_k not initialized"
        if key_proj.rope_proj is None:
            return False, "rope_proj not initialized"
        
        return True, "DecoupledRoPEKey initialized"
    except Exception as e:
        return False, str(e)


def test_decoupled_rope_key_forward() -> Tuple[bool, str]:
    """Test DecoupledRoPEKey forward pass with content and RoPE verification."""
    try:
        d_model, d_latent, num_heads, head_dim, rope_dim = 512, 128, 8, 64, 32
        key_proj = DecoupledRoPEKey(d_model, d_latent, num_heads, head_dim, rope_dim)
        
        if key_proj.up_proj_k is None:
            return False, "key_proj not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, d_model)
        c_kv = torch.randn(batch, seq, d_latent)
        
        k_content, k_rope = key_proj(x, c_kv)
        
        if k_content.shape != (batch, seq, num_heads, head_dim):
            return False, f"k_content shape wrong"
        if k_rope.shape != (batch, seq, num_heads, rope_dim):
            return False, f"k_rope shape wrong"
        
        # Verify k_content comes from c_kv (up projection)
        with torch.no_grad():
            expected_content = key_proj.up_proj_k(c_kv)
            expected_content = expected_content.view(batch, seq, num_heads, head_dim)
        
        if not torch.allclose(k_content, expected_content, atol=1e-5):
            return False, "k_content doesn't match expected up projection"
        
        # Verify k_rope has different values at different positions (rotation applied)
        pos0 = k_rope[0, 0, 0]
        pos1 = k_rope[0, 1, 0]
        if torch.allclose(pos0, pos1, atol=1e-5):
            return False, "k_rope should have different values at different positions (RoPE)"
        
        # Verify outputs are not zeros
        if k_content.abs().sum() == 0:
            return False, "k_content is all zeros"
        if k_rope.abs().sum() == 0:
            return False, "k_rope is all zeros"
        
        return True, f"Decoupled: content from latent, rope position-dependent"
    except Exception as e:
        return False, str(e)


def test_mla_kv_compression_init() -> Tuple[bool, str]:
    """Test MLAKVCompression initialization."""
    try:
        d_model, d_latent, num_heads, head_dim, rope_dim = 512, 128, 8, 64, 32
        mla = MLAKVCompression(d_model, d_latent, num_heads, head_dim, rope_dim)
        
        if mla.kv_down is None:
            return False, "kv_down not initialized"
        if mla.up_proj_k is None:
            return False, "up_proj_k not initialized"
        if mla.up_proj_v is None:
            return False, "up_proj_v not initialized"
        if mla.rope_proj is None:
            return False, "rope_proj not initialized"
        
        return True, "MLAKVCompression initialized"
    except Exception as e:
        return False, str(e)


def test_mla_kv_compression_compress() -> Tuple[bool, str]:
    """Test MLAKVCompression compress method."""
    try:
        d_model, d_latent, num_heads, head_dim, rope_dim = 512, 128, 8, 64, 32
        mla = MLAKVCompression(d_model, d_latent, num_heads, head_dim, rope_dim)
        
        if mla.kv_down is None:
            return False, "mla not initialized"
        
        batch, seq = 2, 16
        x = torch.randn(batch, seq, d_model)
        c = mla.compress(x)
        
        expected_shape = (batch, seq, d_latent)
        if c.shape != expected_shape:
            return False, f"Compressed shape {c.shape} != {expected_shape}"
        
        return True, f"Compressed: {c.shape}"
    except Exception as e:
        return False, str(e)


def test_mla_kv_compression_full_forward() -> Tuple[bool, str]:
    """Test MLAKVCompression full forward."""
    try:
        d_model, d_latent, num_heads, head_dim, rope_dim = 512, 128, 8, 64, 32
        mla = MLAKVCompression(d_model, d_latent, num_heads, head_dim, rope_dim)
        
        if mla.kv_down is None:
            return False, "mla not initialized"
        
        batch, seq = 2, 16
        x = torch.randn(batch, seq, d_model)
        c_kv, k_content, k_rope, v = mla(x)
        
        if c_kv.shape != (batch, seq, d_latent):
            return False, "c_kv shape wrong"
        if k_content.shape != (batch, seq, num_heads, head_dim):
            return False, "k_content shape wrong"
        if k_rope.shape != (batch, seq, num_heads, rope_dim):
            return False, "k_rope shape wrong"
        if v.shape != (batch, seq, num_heads, head_dim):
            return False, "v shape wrong"
        
        return True, "All outputs have correct shapes"
    except Exception as e:
        return False, str(e)


def test_mla_query_compression_init() -> Tuple[bool, str]:
    """Test MLAQueryCompression initialization."""
    try:
        d_model, d_q_latent, num_heads, head_dim, rope_dim = 512, 96, 8, 64, 32
        q_comp = MLAQueryCompression(d_model, d_q_latent, num_heads, head_dim, rope_dim)
        
        if q_comp.q_down is None:
            return False, "q_down not initialized"
        if q_comp.q_up is None:
            return False, "q_up not initialized"
        if q_comp.q_rope_proj is None:
            return False, "q_rope_proj not initialized"
        
        return True, "MLAQueryCompression initialized"
    except Exception as e:
        return False, str(e)


def test_mla_query_compression_forward() -> Tuple[bool, str]:
    """Test MLAQueryCompression forward."""
    try:
        d_model, d_q_latent, num_heads, head_dim, rope_dim = 512, 96, 8, 64, 32
        q_comp = MLAQueryCompression(d_model, d_q_latent, num_heads, head_dim, rope_dim)
        
        if q_comp.q_down is None:
            return False, "q_comp not initialized"
        
        batch, seq = 2, 16
        x = torch.randn(batch, seq, d_model)
        q_content, q_rope = q_comp(x)
        
        if q_content.shape != (batch, seq, num_heads, head_dim):
            return False, f"q_content shape wrong"
        if q_rope.shape != (batch, seq, num_heads, rope_dim):
            return False, f"q_rope shape wrong"
        
        return True, f"q_content: {q_content.shape}, q_rope: {q_rope.shape}"
    except Exception as e:
        return False, str(e)


def test_mla_attention_scores() -> Tuple[bool, str]:
    """Test combined MLA attention score computation matches formula."""
    try:
        batch, num_heads, seq_q, seq_k = 2, 8, 16, 24
        head_dim, rope_dim = 64, 32
        
        torch.manual_seed(42)
        q_content = torch.randn(batch, num_heads, seq_q, head_dim)
        q_rope = torch.randn(batch, num_heads, seq_q, rope_dim)
        k_content = torch.randn(batch, num_heads, seq_k, head_dim)
        k_rope = torch.randn(batch, num_heads, seq_k, rope_dim)
        
        scores = compute_mla_attention_scores(q_content, q_rope, k_content, k_rope)
        
        expected_shape = (batch, num_heads, seq_q, seq_k)
        if scores.shape != expected_shape:
            return False, f"Scores shape {scores.shape} != {expected_shape}"
        
        if scores.abs().sum() == 0:
            return False, "Scores are all zero"
        
        # Verify formula: scores = (q_content @ k_content.T + q_rope @ k_rope.T) * scale
        total_dim = head_dim + rope_dim
        scale = total_dim ** -0.5
        
        content_scores = torch.matmul(q_content, k_content.transpose(-2, -1))
        rope_scores = torch.matmul(q_rope, k_rope.transpose(-2, -1))
        expected_scores = (content_scores + rope_scores) * scale
        
        if not torch.allclose(scores, expected_scores, atol=1e-4):
            return False, "Scores don't match expected formula"
        
        # Verify scores have reasonable range
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            return False, "Scores contain NaN or Inf"
        
        return True, f"Scores match: (q_c @ k_c.T + q_r @ k_r.T) * scale"
    except Exception as e:
        return False, str(e)


def test_gradient_flow() -> Tuple[bool, str]:
    """Test gradient flow through MLA compression."""
    try:
        d_model, d_latent, num_heads, head_dim, rope_dim = 256, 64, 4, 32, 16
        mla = MLAKVCompression(d_model, d_latent, num_heads, head_dim, rope_dim)
        
        if mla.kv_down is None:
            return False, "mla not initialized"
        
        x = torch.randn(2, 8, d_model, requires_grad=True)
        c_kv, k_content, k_rope, v = mla(x)
        
        loss = c_kv.sum() + k_content.sum() + k_rope.sum() + v.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "No gradient to input"
        if x.grad.abs().sum() == 0:
            return False, "Gradients are zero"
        
        return True, "Gradients flow correctly"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("rmsnorm_init", test_rmsnorm_init),
        ("rmsnorm_forward", test_rmsnorm_forward),
        ("rmsnorm_scale", test_rmsnorm_scale),
        ("kv_down_projection_init", test_kv_down_projection_init),
        ("kv_down_projection_forward", test_kv_down_projection_forward),
        ("precompute_freqs", test_precompute_freqs),
        ("apply_rotary_emb", test_apply_rotary_emb),
        ("rotary_position_diff", test_rotary_position_difference),
        ("decoupled_rope_key_init", test_decoupled_rope_key_init),
        ("decoupled_rope_key_forward", test_decoupled_rope_key_forward),
        ("mla_kv_compression_init", test_mla_kv_compression_init),
        ("mla_kv_compression_compress", test_mla_kv_compression_compress),
        ("mla_kv_compression_forward", test_mla_kv_compression_full_forward),
        ("mla_query_compression_init", test_mla_query_compression_init),
        ("mla_query_compression_forward", test_mla_query_compression_forward),
        ("mla_attention_scores", test_mla_attention_scores),
        ("gradient_flow", test_gradient_flow),
    ]
    
    print(f"\n{'='*60}")
    print("Day 27: MLA Key-Value Compression - Tests")
    print("=" * 60)
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    
    print(f"\nSummary: {passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    run_all_tests()
