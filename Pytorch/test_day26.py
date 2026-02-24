"""Test Suite for Day 26: Multi-head Latent Attention (MLA) Basics"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day26 import (
        calculate_kv_cache_size, calculate_mla_cache_size,
        DownProjection, UpProjection, LowRankKVProjection,
        analyze_compression, BasicMLAAttention, measure_reconstruction_error
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_kv_cache_size_calculation() -> Tuple[bool, str]:
    """Test standard KV cache size calculation."""
    try:
        result = calculate_kv_cache_size(
            num_layers=32, seq_len=32768, num_heads=32, head_dim=128
        )
        
        if result['total_bytes'] == 0:
            return False, "Cache size not computed"
        
        # Expected: 2 * 32 * 32768 * 32 * 128 * 2 = ~17GB
        expected_gb = 2 * 32 * 32768 * 32 * 128 * 2 / (1024**3)
        if abs(result['total_gb'] - expected_gb) > 0.1:
            return False, f"Expected ~{expected_gb:.1f}GB, got {result['total_gb']:.1f}GB"
        
        return True, f"Cache size: {result['total_gb']:.2f} GB"
    except Exception as e:
        return False, str(e)


def test_mla_cache_size_calculation() -> Tuple[bool, str]:
    """Test MLA cache size calculation."""
    try:
        result = calculate_mla_cache_size(
            num_layers=32, seq_len=32768, d_latent=512
        )
        
        if result['total_bytes'] == 0:
            return False, "MLA cache size not computed"
        
        # Should be much smaller than standard
        std_result = calculate_kv_cache_size(32, 32768, 32, 128)
        if std_result['total_bytes'] > 0:
            ratio = std_result['total_bytes'] / result['total_bytes']
            if ratio < 5:
                return False, f"MLA should be >5x smaller, got {ratio:.1f}x"
        
        return True, f"MLA cache: {result['total_gb']:.2f} GB"
    except Exception as e:
        return False, str(e)


def test_down_projection_init() -> Tuple[bool, str]:
    """Test DownProjection initialization."""
    try:
        d_model, d_latent = 256, 64
        proj = DownProjection(d_model, d_latent)
        
        if proj.down_proj is None:
            return False, "down_proj not initialized"
        
        if proj.down_proj.weight.shape != (d_latent, d_model):
            return False, f"Weight shape wrong: {proj.down_proj.weight.shape}"
        
        return True, "DownProjection initialized correctly"
    except Exception as e:
        return False, str(e)


def test_down_projection_forward() -> Tuple[bool, str]:
    """Test DownProjection forward pass computes linear projection."""
    try:
        d_model, d_latent = 256, 64
        proj = DownProjection(d_model, d_latent)
        
        if proj.down_proj is None:
            return False, "down_proj not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, d_model)
        out = proj(x)
        
        expected_shape = (batch, seq, d_latent)
        if out.shape != expected_shape:
            return False, f"Output shape {out.shape} != {expected_shape}"
        
        # Verify it computes linear projection
        with torch.no_grad():
            expected = F.linear(x, proj.down_proj.weight, 
                               proj.down_proj.bias if proj.down_proj.bias is not None else None)
        
        if not torch.allclose(out, expected, atol=1e-5):
            return False, "Output doesn't match expected linear projection"
        
        # Verify output is not zeros
        if out.abs().sum() == 0:
            return False, "Output is all zeros"
        
        return True, f"Down projection matches x @ W.T, output range=[{out.min():.2f}, {out.max():.2f}]"
    except Exception as e:
        return False, str(e)


def test_up_projection_init() -> Tuple[bool, str]:
    """Test UpProjection initialization."""
    try:
        d_latent, d_output = 64, 256
        proj = UpProjection(d_latent, d_output)
        
        if proj.up_proj is None:
            return False, "up_proj not initialized"
        
        return True, "UpProjection initialized correctly"
    except Exception as e:
        return False, str(e)


def test_up_projection_forward() -> Tuple[bool, str]:
    """Test UpProjection forward pass computes linear projection."""
    try:
        d_latent, d_output = 64, 256
        proj = UpProjection(d_latent, d_output)
        
        if proj.up_proj is None:
            return False, "up_proj not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        c = torch.randn(batch, seq, d_latent)
        out = proj(c)
        
        expected_shape = (batch, seq, d_output)
        if out.shape != expected_shape:
            return False, f"Output shape {out.shape} != {expected_shape}"
        
        # Verify it computes linear projection
        with torch.no_grad():
            expected = F.linear(c, proj.up_proj.weight,
                               proj.up_proj.bias if proj.up_proj.bias is not None else None)
        
        if not torch.allclose(out, expected, atol=1e-5):
            return False, "Output doesn't match expected linear projection"
        
        # Verify output is not zeros
        if out.abs().sum() == 0:
            return False, "Output is all zeros"
        
        return True, f"Up projection matches c @ W.T, output range=[{out.min():.2f}, {out.max():.2f}]"
    except Exception as e:
        return False, str(e)


def test_low_rank_kv_projection_init() -> Tuple[bool, str]:
    """Test LowRankKVProjection initialization."""
    try:
        d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
        proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
        
        if proj.down_proj is None:
            return False, "down_proj not initialized"
        if proj.up_proj_k is None:
            return False, "up_proj_k not initialized"
        if proj.up_proj_v is None:
            return False, "up_proj_v not initialized"
        
        return True, "LowRankKVProjection initialized"
    except Exception as e:
        return False, str(e)


def test_low_rank_kv_compress() -> Tuple[bool, str]:
    """Test KV compression computes correct linear transformation."""
    try:
        d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
        proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
        
        if proj.down_proj is None:
            return False, "Projection not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, d_model)
        c = proj.compress(x)
        
        expected_shape = (batch, seq, d_latent)
        if c.shape != expected_shape:
            return False, f"Compressed shape {c.shape} != {expected_shape}"
        
        # Verify compression is linear projection
        with torch.no_grad():
            expected = F.linear(x, proj.down_proj.weight,
                               proj.down_proj.bias if proj.down_proj.bias is not None else None)
        
        if not torch.allclose(c, expected, atol=1e-5):
            return False, "Compressed output doesn't match linear projection"
        
        # Verify compression actually reduces dimension
        input_size = x.numel()
        compressed_size = c.numel()
        compression_ratio = input_size / compressed_size
        if compression_ratio < 1.5:
            return False, f"Compression ratio {compression_ratio:.2f} too low"
        
        return True, f"Compression ratio: {compression_ratio:.1f}x"
    except Exception as e:
        return False, str(e)


def test_low_rank_kv_reconstruct() -> Tuple[bool, str]:
    """Test K, V reconstruction from latent."""
    try:
        d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
        proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
        
        if proj.down_proj is None:
            return False, "Projection not initialized"
        
        batch, seq = 2, 16
        c = torch.randn(batch, seq, d_latent)
        k, v = proj.reconstruct_kv(c)
        
        expected_shape = (batch, seq, num_heads, head_dim)
        if k.shape != expected_shape:
            return False, f"K shape {k.shape} != {expected_shape}"
        if v.shape != expected_shape:
            return False, f"V shape {v.shape} != {expected_shape}"
        
        return True, f"K, V shapes: {k.shape}"
    except Exception as e:
        return False, str(e)


def test_low_rank_kv_full_forward() -> Tuple[bool, str]:
    """Test full forward pass of LowRankKVProjection."""
    try:
        d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
        proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
        
        if proj.down_proj is None:
            return False, "Projection not initialized"
        
        batch, seq = 2, 16
        x = torch.randn(batch, seq, d_model)
        c, k, v = proj(x)
        
        if c.shape != (batch, seq, d_latent):
            return False, "Latent shape wrong"
        if k.shape != (batch, seq, num_heads, head_dim):
            return False, "K shape wrong"
        if v.shape != (batch, seq, num_heads, head_dim):
            return False, "V shape wrong"
        
        return True, "Full forward pass works"
    except Exception as e:
        return False, str(e)


def test_compression_analysis() -> Tuple[bool, str]:
    """Test compression ratio analysis."""
    try:
        d_model, d_latent, num_heads, head_dim = 4096, 512, 32, 128
        result = analyze_compression(d_model, d_latent, num_heads, head_dim)
        
        if result['compression_ratio'] == 0:
            return False, "Compression ratio not computed"
        
        # Expected: standard = 2 * 32 * 128 = 8192, mla = 512 -> ratio = 16
        expected_ratio = 2 * num_heads * head_dim / d_latent
        if abs(result['compression_ratio'] - expected_ratio) > 0.1:
            return False, f"Expected ratio ~{expected_ratio}, got {result['compression_ratio']}"
        
        return True, f"Compression ratio: {result['compression_ratio']:.1f}x"
    except Exception as e:
        return False, str(e)


def test_basic_mla_attention_init() -> Tuple[bool, str]:
    """Test BasicMLAAttention initialization."""
    try:
        d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
        attn = BasicMLAAttention(d_model, d_latent, num_heads, head_dim)
        
        if attn.W_q is None:
            return False, "W_q not initialized"
        if attn.kv_proj is None:
            return False, "kv_proj not initialized"
        if attn.W_o is None:
            return False, "W_o not initialized"
        
        return True, "BasicMLAAttention initialized"
    except Exception as e:
        return False, str(e)


def test_basic_mla_attention_forward() -> Tuple[bool, str]:
    """Test BasicMLAAttention forward pass produces valid attention."""
    try:
        d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
        attn = BasicMLAAttention(d_model, d_latent, num_heads, head_dim)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, d_model)
        output, weights, _ = attn(x)
        
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != {x.shape}"
        
        expected_weights_shape = (batch, num_heads, seq, seq)
        if weights.shape != expected_weights_shape:
            return False, f"Weights shape wrong"
        
        # Verify output is not zeros
        if output.abs().sum() == 0:
            return False, "Output is all zeros"
        
        # Verify attention weights are valid probabilities
        if (weights < 0).any():
            return False, "Attention weights have negative values"
        
        attn_sums = weights.sum(dim=-1)
        if not torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4):
            return False, "Attention weights don't sum to 1"
        
        # Verify output has reasonable values
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Output contains NaN or Inf"
        
        return True, f"MLA attention works, output std={output.std():.4f}"
    except Exception as e:
        return False, str(e)


def test_basic_mla_attention_cache() -> Tuple[bool, str]:
    """Test BasicMLAAttention with caching."""
    try:
        d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
        attn = BasicMLAAttention(d_model, d_latent, num_heads, head_dim)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        batch, seq = 2, 8
        x = torch.randn(batch, seq, d_model)
        
        # Forward with cache
        output1, _, cache = attn(x, use_cache=True)
        
        if cache is None:
            return False, "Cache not returned"
        
        if cache.shape != (batch, seq, d_latent):
            return False, f"Cache shape {cache.shape} != {(batch, seq, d_latent)}"
        
        return True, f"Cache shape: {cache.shape}"
    except Exception as e:
        return False, str(e)


def test_basic_mla_attention_incremental() -> Tuple[bool, str]:
    """Test incremental generation with cache."""
    try:
        d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
        attn = BasicMLAAttention(d_model, d_latent, num_heads, head_dim)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        batch = 2
        
        # Initial forward
        x = torch.randn(batch, 4, d_model)
        _, _, cache = attn(x, use_cache=True)
        
        if cache is None:
            return False, "Initial cache not created"
        
        # Incremental forward with single new token
        new_token = torch.randn(batch, 1, d_model)
        output, _, new_cache = attn(new_token, kv_cache=cache, use_cache=True)
        
        if output.shape != (batch, 1, d_model):
            return False, f"Incremental output shape wrong"
        
        if new_cache is None:
            return False, "Updated cache not returned"
        
        # Cache should grow by 1
        expected_cache_len = 4 + 1
        if new_cache.shape[1] != expected_cache_len:
            return False, f"Cache length {new_cache.shape[1]} != {expected_cache_len}"
        
        return True, f"Incremental generation works"
    except Exception as e:
        return False, str(e)


def test_gradient_flow() -> Tuple[bool, str]:
    """Test gradient flow through MLA."""
    try:
        d_model, d_latent, num_heads, head_dim = 256, 64, 8, 32
        attn = BasicMLAAttention(d_model, d_latent, num_heads, head_dim)
        
        if attn.W_q is None:
            return False, "Attention not initialized"
        
        x = torch.randn(2, 8, d_model, requires_grad=True)
        output, _, _ = attn(x)
        
        loss = output.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "No gradient to input"
        if x.grad.abs().sum() == 0:
            return False, "Gradients are zero"
        
        return True, "Gradients flow correctly"
    except Exception as e:
        return False, str(e)


def test_compression_vs_standard_equivalence() -> Tuple[bool, str]:
    """Test that low-rank can approximate full-rank."""
    try:
        d_model, d_latent, num_heads, head_dim = 64, 32, 4, 16
        d_kv = num_heads * head_dim
        
        proj = LowRankKVProjection(d_model, d_latent, num_heads, head_dim)
        
        if proj.down_proj is None:
            return False, "Projection not initialized"
        
        # Create "ideal" low-rank weights that perfectly factorize
        # W = W_down @ W_up should work well
        with torch.no_grad():
            # Full-rank equivalent
            full_W_k = proj.down_proj.weight.T @ proj.up_proj_k.weight.T
            
        batch, seq = 2, 8
        x = torch.randn(batch, seq, d_model)
        
        # Low-rank result
        c, k, v = proj(x)
        k_flat = k.view(batch, seq, -1)
        
        # Full-rank result using equivalent weights
        k_full = F.linear(x, full_W_k.T)
        
        # Should be exactly equal (same computation, different path)
        error = (k_flat - k_full).abs().max()
        if error > 1e-4:
            return False, f"Factorization error too large: {error}"
        
        return True, f"Low-rank factorization is exact"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("kv_cache_size", test_kv_cache_size_calculation),
        ("mla_cache_size", test_mla_cache_size_calculation),
        ("down_projection_init", test_down_projection_init),
        ("down_projection_forward", test_down_projection_forward),
        ("up_projection_init", test_up_projection_init),
        ("up_projection_forward", test_up_projection_forward),
        ("low_rank_kv_init", test_low_rank_kv_projection_init),
        ("low_rank_kv_compress", test_low_rank_kv_compress),
        ("low_rank_kv_reconstruct", test_low_rank_kv_reconstruct),
        ("low_rank_kv_forward", test_low_rank_kv_full_forward),
        ("compression_analysis", test_compression_analysis),
        ("basic_mla_init", test_basic_mla_attention_init),
        ("basic_mla_forward", test_basic_mla_attention_forward),
        ("basic_mla_cache", test_basic_mla_attention_cache),
        ("basic_mla_incremental", test_basic_mla_attention_incremental),
        ("gradient_flow", test_gradient_flow),
        ("compression_equivalence", test_compression_vs_standard_equivalence),
    ]
    
    print(f"\n{'='*60}")
    print("Day 26: Multi-head Latent Attention (MLA) Basics - Tests")
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
