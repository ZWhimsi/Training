"""Test Suite for Day 28: MLA Full Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day28 import (
        MLAConfig, MLAKVCache, MultiheadLatentAttention,
        MLATransformerBlock, MLAModel, generate_with_mla,
        compare_memory_usage
    )
    IMPORT_SUCCESS = True
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


def test_mla_config() -> Tuple[bool, str]:
    """Test MLAConfig creation."""
    try:
        config = get_test_config()
        
        if config.d_model != 256:
            return False, "d_model not set"
        if config.num_heads != 4:
            return False, "num_heads not set"
        
        return True, "Config created successfully"
    except Exception as e:
        return False, str(e)


def test_kv_cache_init() -> Tuple[bool, str]:
    """Test MLAKVCache initialization with zeros."""
    try:
        config = get_test_config()
        batch = 2
        max_len = 64
        
        cache = MLAKVCache(batch, max_len, config.d_kv_latent, 
                          config.num_heads, config.rope_dim)
        
        if cache.c_kv is None:
            return False, "c_kv not initialized"
        if cache.k_rope is None:
            return False, "k_rope not initialized"
        
        if cache.c_kv.shape != (batch, max_len, config.d_kv_latent):
            return False, "c_kv shape wrong"
        if cache.k_rope.shape != (batch, max_len, config.num_heads, config.rope_dim):
            return False, "k_rope shape wrong"
        
        # Verify cache is initialized with zeros
        if cache.c_kv.abs().sum() != 0:
            return False, "c_kv should be initialized with zeros"
        if cache.k_rope.abs().sum() != 0:
            return False, "k_rope should be initialized with zeros"
        
        # Verify seq_len starts at 0
        if cache.seq_len != 0:
            return False, f"seq_len should be 0, got {cache.seq_len}"
        
        return True, "Cache initialized with zeros, seq_len=0"
    except Exception as e:
        return False, str(e)


def test_kv_cache_update() -> Tuple[bool, str]:
    """Test MLAKVCache update stores values correctly."""
    try:
        config = get_test_config()
        batch, max_len = 2, 64
        
        cache = MLAKVCache(batch, max_len, config.d_kv_latent,
                          config.num_heads, config.rope_dim)
        
        if cache.c_kv is None:
            return False, "Cache not initialized"
        
        # Add some values
        seq_len = 8
        torch.manual_seed(42)
        c_kv = torch.randn(batch, seq_len, config.d_kv_latent)
        k_rope = torch.randn(batch, seq_len, config.num_heads, config.rope_dim)
        
        new_len = cache.update(c_kv, k_rope)
        
        if new_len != seq_len:
            return False, f"New length {new_len} != {seq_len}"
        
        # Verify values were stored correctly
        if not torch.allclose(cache.c_kv[:, :seq_len], c_kv):
            return False, "c_kv values not stored correctly"
        if not torch.allclose(cache.k_rope[:, :seq_len], k_rope):
            return False, "k_rope values not stored correctly"
        
        # Add more
        c_kv2 = torch.randn(batch, seq_len, config.d_kv_latent)
        k_rope2 = torch.randn(batch, seq_len, config.num_heads, config.rope_dim)
        new_len = cache.update(c_kv2, k_rope2)
        
        if new_len != 2 * seq_len:
            return False, f"New length {new_len} != {2 * seq_len}"
        
        # Verify second update stored correctly (appended)
        if not torch.allclose(cache.c_kv[:, seq_len:2*seq_len], c_kv2):
            return False, "Second c_kv values not appended correctly"
        
        return True, f"Cache updated correctly to length {new_len}"
    except Exception as e:
        return False, str(e)


def test_kv_cache_get() -> Tuple[bool, str]:
    """Test MLAKVCache retrieval returns correct values."""
    try:
        config = get_test_config()
        batch, max_len = 2, 64
        
        cache = MLAKVCache(batch, max_len, config.d_kv_latent,
                          config.num_heads, config.rope_dim)
        
        if cache.c_kv is None:
            return False, "Cache not initialized"
        
        seq_len = 8
        torch.manual_seed(42)
        c_kv = torch.randn(batch, seq_len, config.d_kv_latent)
        k_rope = torch.randn(batch, seq_len, config.num_heads, config.rope_dim)
        cache.update(c_kv, k_rope)
        
        cached_c, cached_k = cache.get()
        
        if cached_c is None or cached_k is None:
            return False, "Get returned None"
        
        if cached_c.shape != (batch, seq_len, config.d_kv_latent):
            return False, f"Retrieved c_kv shape wrong: {cached_c.shape}"
        
        # Verify retrieved values match what was stored
        if not torch.allclose(cached_c, c_kv):
            return False, "Retrieved c_kv values don't match stored values"
        if not torch.allclose(cached_k, k_rope):
            return False, "Retrieved k_rope values don't match stored values"
        
        return True, f"Cache retrieval returns correct values"
    except Exception as e:
        return False, str(e)


def test_mla_attention_init() -> Tuple[bool, str]:
    """Test MultiheadLatentAttention initialization."""
    try:
        config = get_test_config()
        mla = MultiheadLatentAttention(config)
        
        if mla.kv_down is None:
            return False, "kv_down not initialized"
        if mla.k_up is None:
            return False, "k_up not initialized"
        if mla.v_up is None:
            return False, "v_up not initialized"
        if mla.out_proj is None:
            return False, "out_proj not initialized"
        
        return True, "MLA attention initialized"
    except Exception as e:
        return False, str(e)


def test_mla_attention_compute_q() -> Tuple[bool, str]:
    """Test query computation."""
    try:
        config = get_test_config()
        mla = MultiheadLatentAttention(config)
        
        if mla.kv_down is None:
            return False, "MLA not initialized"
        
        batch, seq = 2, 16
        x = torch.randn(batch, seq, config.d_model)
        
        q_content, q_rope = mla.compute_q(x)
        
        if q_content.shape != (batch, config.num_heads, seq, config.head_dim):
            return False, f"q_content shape wrong: {q_content.shape}"
        if q_rope.shape != (batch, config.num_heads, seq, config.rope_dim):
            return False, f"q_rope shape wrong: {q_rope.shape}"
        
        return True, "Query computation works"
    except Exception as e:
        return False, str(e)


def test_mla_attention_compute_kv() -> Tuple[bool, str]:
    """Test KV computation."""
    try:
        config = get_test_config()
        mla = MultiheadLatentAttention(config)
        
        if mla.kv_down is None:
            return False, "MLA not initialized"
        
        batch, seq = 2, 16
        x = torch.randn(batch, seq, config.d_model)
        
        c_kv, k_content, k_rope, v = mla.compute_kv(x)
        
        if c_kv.shape != (batch, seq, config.d_kv_latent):
            return False, f"c_kv shape wrong"
        if k_content.shape != (batch, config.num_heads, seq, config.head_dim):
            return False, f"k_content shape wrong"
        if k_rope.shape != (batch, seq, config.num_heads, config.rope_dim):
            return False, f"k_rope shape wrong"
        if v.shape != (batch, config.num_heads, seq, config.head_dim):
            return False, f"v shape wrong"
        
        return True, "KV computation works"
    except Exception as e:
        return False, str(e)


def test_mla_attention_forward() -> Tuple[bool, str]:
    """Test MLA forward pass produces valid attention."""
    try:
        config = get_test_config()
        mla = MultiheadLatentAttention(config)
        
        if mla.kv_down is None:
            return False, "MLA not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, config.d_model)
        
        output, attn = mla(x)
        
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != {x.shape}"
        
        expected_attn_shape = (batch, config.num_heads, seq, seq)
        if attn.shape != expected_attn_shape:
            return False, f"Attention shape wrong"
        
        # Verify output is not zeros
        if output.abs().sum() == 0:
            return False, "Output is all zeros"
        
        # Verify attention weights are valid
        if (attn < 0).any():
            return False, "Attention weights have negative values"
        
        attn_sums = attn.sum(dim=-1)
        if not torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4):
            return False, "Attention weights don't sum to 1"
        
        # Verify output has reasonable values
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Output contains NaN or Inf"
        
        return True, f"MLA forward works, output std={output.std():.4f}"
    except Exception as e:
        return False, str(e)


def test_mla_attention_with_cache() -> Tuple[bool, str]:
    """Test MLA with caching."""
    try:
        config = get_test_config()
        mla = MultiheadLatentAttention(config)
        
        if mla.kv_down is None:
            return False, "MLA not initialized"
        
        batch = 2
        
        # Create cache
        cache = MLAKVCache(batch, 64, config.d_kv_latent,
                          config.num_heads, config.rope_dim)
        
        if cache.c_kv is None:
            return False, "Cache not initialized"
        
        # First forward (prefill)
        x1 = torch.randn(batch, 8, config.d_model)
        output1, _ = mla(x1, cache=cache, start_pos=0)
        
        if cache.seq_len != 8:
            return False, f"Cache length after prefill: {cache.seq_len}"
        
        # Second forward (decode)
        x2 = torch.randn(batch, 1, config.d_model)
        output2, attn2 = mla(x2, cache=cache, start_pos=8)
        
        if cache.seq_len != 9:
            return False, f"Cache length after decode: {cache.seq_len}"
        
        # Attention should be to all cached positions
        if attn2.shape != (batch, config.num_heads, 1, 9):
            return False, f"Decode attention shape wrong: {attn2.shape}"
        
        return True, "Caching works correctly"
    except Exception as e:
        return False, str(e)


def test_mla_transformer_block_init() -> Tuple[bool, str]:
    """Test MLATransformerBlock initialization."""
    try:
        config = get_test_config()
        block = MLATransformerBlock(config)
        
        if block.attn is None:
            return False, "attn not initialized"
        if block.attn_norm is None:
            return False, "attn_norm not initialized"
        if block.ffn is None:
            return False, "ffn not initialized"
        if block.ffn_norm is None:
            return False, "ffn_norm not initialized"
        
        return True, "Block initialized"
    except Exception as e:
        return False, str(e)


def test_mla_transformer_block_forward() -> Tuple[bool, str]:
    """Test MLATransformerBlock forward with residual verification."""
    try:
        config = get_test_config()
        block = MLATransformerBlock(config)
        
        if block.attn is None:
            return False, "Block not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, config.d_model)
        
        output, attn = block(x)
        
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != {x.shape}"
        
        # Verify output is not zeros
        if output.abs().sum() == 0:
            return False, "Output is all zeros"
        
        # Verify residual connection is working (output correlates with input)
        correlation = F.cosine_similarity(output.flatten(), x.flatten(), dim=0)
        if correlation < 0.1:
            return False, f"Residual connection may not work: correlation={correlation:.3f}"
        
        # Verify output has reasonable values
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Output contains NaN or Inf"
        
        return True, f"Block forward works, residual correlation={correlation:.3f}"
    except Exception as e:
        return False, str(e)


def test_mla_model_init() -> Tuple[bool, str]:
    """Test MLAModel initialization."""
    try:
        config = get_test_config()
        model = MLAModel(config, num_layers=2)
        
        if model.layers is None:
            return False, "layers not initialized"
        if len(model.layers) != 2:
            return False, f"Expected 2 layers, got {len(model.layers)}"
        if model.final_norm is None:
            return False, "final_norm not initialized"
        
        return True, "Model initialized with 2 layers"
    except Exception as e:
        return False, str(e)


def test_mla_model_forward() -> Tuple[bool, str]:
    """Test MLAModel forward through multiple layers."""
    try:
        config = get_test_config()
        model = MLAModel(config, num_layers=2)
        
        if model.layers is None:
            return False, "Model not initialized"
        
        batch, seq = 2, 16
        torch.manual_seed(42)
        x = torch.randn(batch, seq, config.d_model)
        
        output, attn_weights = model(x)
        
        if output.shape != x.shape:
            return False, f"Output shape wrong"
        
        # Verify output is not zeros
        if output.abs().sum() == 0:
            return False, "Output is all zeros"
        
        # Verify attention weights from all layers
        if len(attn_weights) != 2:
            return False, f"Expected attention weights from 2 layers, got {len(attn_weights)}"
        
        # Verify output has reasonable values
        if torch.isnan(output).any() or torch.isinf(output).any():
            return False, "Output contains NaN or Inf"
        
        # Verify final norm was applied (output should be normalized)
        output_rms = output.pow(2).mean(dim=-1).sqrt().mean()
        if output_rms > 10:
            return False, f"Output may not be normalized, RMS={output_rms:.2f}"
        
        return True, f"Model forward works through 2 layers, output RMS={output_rms:.2f}"
    except Exception as e:
        return False, str(e)


def test_generation() -> Tuple[bool, str]:
    """Test generation with MLA produces valid sequence."""
    try:
        config = get_test_config()
        model = MLAModel(config, num_layers=2)
        
        if model.layers is None:
            return False, "Model not initialized"
        
        batch = 2
        prompt_len = 8
        max_new = 4
        
        torch.manual_seed(42)
        prompt = torch.randn(batch, prompt_len, config.d_model)
        
        all_hidden, caches = generate_with_mla(model, prompt, max_new_tokens=max_new)
        
        expected_len = prompt_len + max_new
        if all_hidden.shape[1] != expected_len:
            return False, f"Generated length {all_hidden.shape[1]} != {expected_len}"
        
        # Verify hidden states are not zeros
        if all_hidden.abs().sum() == 0:
            return False, "Generated hidden states are all zeros"
        
        # Verify caches were populated
        if len(caches) != 2:  # 2 layers
            return False, f"Expected caches for 2 layers, got {len(caches)}"
        
        for layer_idx, cache in caches.items():
            if cache.seq_len != expected_len:
                return False, f"Layer {layer_idx} cache length {cache.seq_len} != {expected_len}"
        
        # Verify output has reasonable values
        if torch.isnan(all_hidden).any() or torch.isinf(all_hidden).any():
            return False, "Generated hidden states contain NaN or Inf"
        
        return True, f"Generated {max_new} tokens with caching, total length={expected_len}"
    except Exception as e:
        return False, str(e)


def test_memory_comparison() -> Tuple[bool, str]:
    """Test memory comparison calculation with expected formula."""
    try:
        config = get_test_config()
        seq_len = 1024
        num_layers = 8
        
        mem = compare_memory_usage(config, seq_len=seq_len, num_layers=num_layers)
        
        if mem['memory_reduction'] == 0:
            return False, "Memory comparison not computed"
        
        # Verify the calculation formula
        # Standard: 2 * num_heads * head_dim per layer per token (K and V)
        # MLA: d_kv_latent + num_heads * rope_dim per layer per token
        d_kv = config.num_heads * config.head_dim
        std_per_token = 2 * d_kv
        mla_per_token = config.d_kv_latent + config.num_heads * config.rope_dim
        
        expected_reduction = std_per_token / mla_per_token
        
        # MLA should use less memory
        if mem['memory_reduction'] < 1.5:
            return False, f"Expected >1.5x reduction, got {mem['memory_reduction']:.2f}x"
        
        # Verify standard cache elements
        expected_std_elements = num_layers * seq_len * std_per_token
        if mem['standard_cache_elements'] != expected_std_elements:
            return False, f"Standard elements {mem['standard_cache_elements']} != {expected_std_elements}"
        
        return True, f"Memory reduction: {mem['memory_reduction']:.1f}x (expected ~{expected_reduction:.1f}x)"
    except Exception as e:
        return False, str(e)


def test_gradient_flow() -> Tuple[bool, str]:
    """Test gradient flow through MLA model."""
    try:
        config = get_test_config()
        model = MLAModel(config, num_layers=2)
        
        if model.layers is None:
            return False, "Model not initialized"
        
        x = torch.randn(2, 8, config.d_model, requires_grad=True)
        output, _ = model(x)
        
        loss = output.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "No gradient to input"
        if x.grad.abs().sum() == 0:
            return False, "Gradients are zero"
        
        return True, "Gradients flow correctly"
    except Exception as e:
        return False, str(e)


def test_causal_mask() -> Tuple[bool, str]:
    """Test that causal masking works correctly."""
    try:
        config = get_test_config()
        mla = MultiheadLatentAttention(config)
        
        if mla.kv_down is None:
            return False, "MLA not initialized"
        
        batch, seq = 1, 8
        x = torch.randn(batch, seq, config.d_model)
        
        # Create causal mask
        mask = torch.triu(
            torch.ones(seq, seq) * float('-inf'),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        output, attn = mla(x, mask=mask)
        
        # Check that attention is causal (upper triangle is ~0)
        upper_attn = torch.triu(attn[0, 0], diagonal=1)
        if upper_attn.abs().max() > 1e-6:
            return False, "Attention is not causal"
        
        return True, "Causal masking works"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("mla_config", test_mla_config),
        ("kv_cache_init", test_kv_cache_init),
        ("kv_cache_update", test_kv_cache_update),
        ("kv_cache_get", test_kv_cache_get),
        ("mla_attention_init", test_mla_attention_init),
        ("mla_attention_compute_q", test_mla_attention_compute_q),
        ("mla_attention_compute_kv", test_mla_attention_compute_kv),
        ("mla_attention_forward", test_mla_attention_forward),
        ("mla_attention_with_cache", test_mla_attention_with_cache),
        ("mla_transformer_block_init", test_mla_transformer_block_init),
        ("mla_transformer_block_forward", test_mla_transformer_block_forward),
        ("mla_model_init", test_mla_model_init),
        ("mla_model_forward", test_mla_model_forward),
        ("generation", test_generation),
        ("memory_comparison", test_memory_comparison),
        ("gradient_flow", test_gradient_flow),
        ("causal_mask", test_causal_mask),
    ]
    
    print(f"\n{'='*60}")
    print("Day 28: MLA Full Implementation - Tests")
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
