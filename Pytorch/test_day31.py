"""Test Suite for Day 31: DeepSeek Math Model Assembly"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day31 import (
        DeepSeekMathConfig, TokenEmbedding, RotaryEmbedding,
        MultiHeadLatentAttention, DeepSeekBlock, DeepSeekMathModel,
        create_deepseek_math_small_config, create_deepseek_math_7b_config,
        analyze_model_stats, count_parameters_by_component,
        apply_rotary_emb
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def get_test_config():
    """Get minimal config for fast testing."""
    return DeepSeekMathConfig(
        vocab_size=1000,
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        latent_dim=16,
        d_ff=172,
        max_seq_len=128
    )


def test_config_properties() -> Tuple[bool, str]:
    """Test config computed properties."""
    try:
        config = get_test_config()
        
        expected_head_dim = config.d_model // config.num_heads
        if config.head_dim != expected_head_dim:
            return False, f"head_dim {config.head_dim} != {expected_head_dim}"
        
        return True, f"Config properties correct: head_dim={config.head_dim}"
    except Exception as e:
        return False, str(e)


def test_token_embedding_init() -> Tuple[bool, str]:
    """Test token embedding initialization."""
    try:
        config = get_test_config()
        embed = TokenEmbedding(config.vocab_size, config.d_model)
        
        if embed.embedding is None:
            return False, "Embedding not initialized"
        
        if embed.embedding.num_embeddings != config.vocab_size:
            return False, "Wrong vocab size"
        if embed.embedding.embedding_dim != config.d_model:
            return False, "Wrong embedding dim"
        
        return True, "Token embedding initialized"
    except Exception as e:
        return False, str(e)


def test_token_embedding_forward() -> Tuple[bool, str]:
    """Test token embedding forward pass."""
    try:
        config = get_test_config()
        embed = TokenEmbedding(config.vocab_size, config.d_model, scale=True)
        
        if embed.embedding is None:
            return False, "Embedding not initialized"
        
        batch, seq_len = 2, 16
        torch.manual_seed(42)
        ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        
        out = embed(ids)
        
        expected = (batch, seq_len, config.d_model)
        if out.shape != expected:
            return False, f"Output shape {out.shape} != {expected}"
        
        # Verify scaling: output should be embedding * sqrt(d_model)
        import math
        raw_embed = embed.embedding(ids)
        expected_out = raw_embed * math.sqrt(config.d_model)
        
        if not torch.allclose(out, expected_out, atol=1e-5):
            return False, "Embedding not scaled by sqrt(d_model)"
        
        return True, f"Embedding verified with scaling"
    except Exception as e:
        return False, str(e)


def test_token_embedding_scaling() -> Tuple[bool, str]:
    """Test embedding scaling by sqrt(d_model)."""
    try:
        config = get_test_config()
        
        embed_scaled = TokenEmbedding(config.vocab_size, config.d_model, scale=True)
        embed_unscaled = TokenEmbedding(config.vocab_size, config.d_model, scale=False)
        
        if embed_scaled.embedding is None or embed_unscaled.embedding is None:
            return False, "Embeddings not initialized"
        
        # Copy weights
        with torch.no_grad():
            embed_unscaled.embedding.weight.copy_(embed_scaled.embedding.weight)
        
        ids = torch.randint(0, config.vocab_size, (2, 16))
        
        out_scaled = embed_scaled(ids)
        out_unscaled = embed_unscaled(ids)
        
        # Scaled should be larger by sqrt(d_model)
        expected_ratio = (config.d_model) ** 0.5
        actual_ratio = out_scaled.abs().mean() / out_unscaled.abs().mean()
        
        if abs(actual_ratio - expected_ratio) > 0.1:
            return False, f"Scale ratio {actual_ratio:.2f} != {expected_ratio:.2f}"
        
        return True, f"Scaling verified: ratio={actual_ratio:.2f}"
    except Exception as e:
        return False, str(e)


def test_rope_init() -> Tuple[bool, str]:
    """Test RoPE initialization."""
    try:
        dim = 32
        max_len = 128
        base = 10000.0
        rope = RotaryEmbedding(dim, max_len, base)
        
        cos, sin = rope.get_cos_sin(16)
        
        if cos.shape != (16, dim // 2):
            return False, f"cos shape {cos.shape} != (16, {dim // 2})"
        if sin.shape != (16, dim // 2):
            return False, f"sin shape {sin.shape} != (16, {dim // 2})"
        
        # Verify cos² + sin² = 1
        sum_sq = cos ** 2 + sin ** 2
        if not torch.allclose(sum_sq, torch.ones_like(sum_sq), atol=1e-5):
            return False, "cos² + sin² should equal 1"
        
        # Verify position 0: cos=1, sin=0 for all frequencies
        if not torch.allclose(cos[0], torch.ones(dim // 2), atol=1e-5):
            return False, "cos(0) should be 1 for all frequencies"
        if not torch.allclose(sin[0], torch.zeros(dim // 2), atol=1e-5):
            return False, "sin(0) should be 0 for all frequencies"
        
        return True, f"RoPE verified: cos²+sin²=1, position 0 correct"
    except Exception as e:
        return False, str(e)


def test_rope_offset() -> Tuple[bool, str]:
    """Test RoPE with position offset."""
    try:
        dim = 32
        rope = RotaryEmbedding(dim, 128)
        
        # Get rotations for positions 0-15
        cos1, sin1 = rope.get_cos_sin(16, offset=0)
        
        # Get rotations for positions 8-23
        cos2, sin2 = rope.get_cos_sin(16, offset=8)
        
        # They should be different
        if torch.allclose(cos1, cos2):
            return False, "Different offsets should give different rotations"
        
        # Overlapping region (8-15) should match
        cos1_overlap = rope.get_cos_sin(8, offset=8)[0]
        cos2_overlap = cos2[:8]
        
        if not torch.allclose(cos1_overlap, cos2_overlap, atol=1e-5):
            return False, "Overlapping positions should match"
        
        return True, "RoPE offset handling correct"
    except Exception as e:
        return False, str(e)


def test_apply_rotary_emb() -> Tuple[bool, str]:
    """Test applying rotary embeddings."""
    try:
        batch, heads, seq_len, head_dim = 2, 4, 16, 32
        
        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads // 2, seq_len, head_dim)
        
        rope = RotaryEmbedding(head_dim, 128)
        cos, sin = rope.get_cos_sin(seq_len)
        
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        
        if q_rot.shape != q.shape:
            return False, f"Q shape changed: {q_rot.shape}"
        if k_rot.shape != k.shape:
            return False, f"K shape changed: {k_rot.shape}"
        
        # Should be different from original
        if torch.allclose(q, q_rot):
            return False, "RoPE didn't modify Q"
        
        # RoPE is a rotation, so it should preserve vector norms
        q_norm = q.norm(dim=-1)
        q_rot_norm = q_rot.norm(dim=-1)
        if not torch.allclose(q_norm, q_rot_norm, atol=1e-4):
            return False, "RoPE should preserve Q vector norms"
        
        k_norm = k.norm(dim=-1)
        k_rot_norm = k_rot.norm(dim=-1)
        if not torch.allclose(k_norm, k_rot_norm, atol=1e-4):
            return False, "RoPE should preserve K vector norms"
        
        return True, "RoPE verified with norm preservation"
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
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, config.d_model)
        
        rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
        cos, sin = rope.get_cos_sin(seq_len)
        
        out, cache = mla(x, cos, sin)
        
        if out.shape != x.shape:
            return False, f"Output shape {out.shape} != {x.shape}"
        
        # Verify output is different from input
        if torch.allclose(out, x, atol=1e-3):
            return False, "MLA output too similar to input"
        
        # Verify KV compression is being used
        latent = mla.W_kv_compress(x)
        if latent.shape[-1] != config.latent_dim:
            return False, f"KV compression not using latent_dim={config.latent_dim}"
        
        # Verify cache is returned
        if cache is None:
            return False, "Cache should be returned"
        
        k_cache, v_cache = cache
        expected_cache_shape = (batch, config.num_kv_heads, seq_len, config.head_dim)
        if k_cache.shape != expected_cache_shape:
            return False, f"K cache shape {k_cache.shape} != {expected_cache_shape}"
        
        return True, f"MLA verified with KV compression"
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
        torch.manual_seed(42)
        rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
        
        # Prefill
        prompt_len = 8
        x1 = torch.randn(batch, prompt_len, config.d_model)
        cos1, sin1 = rope.get_cos_sin(prompt_len)
        out1, cache = mla(x1, cos1, sin1)
        
        if cache is None:
            return False, "Cache not returned"
        
        k_cache, v_cache = cache
        expected_shape = (batch, config.num_kv_heads, prompt_len, config.head_dim)
        if k_cache.shape != expected_shape:
            return False, f"Cache shape {k_cache.shape} != {expected_shape}"
        
        # Decode step
        x2 = torch.randn(batch, 1, config.d_model)
        cos2, sin2 = rope.get_cos_sin(1, offset=prompt_len)
        out2, new_cache = mla(x2, cos2, sin2, kv_cache=cache)
        
        if new_cache is None:
            return False, "Cache not updated"
        
        new_k, new_v = new_cache
        if new_k.shape[2] != prompt_len + 1:
            return False, f"Cache not extended: {new_k.shape[2]}"
        
        # Verify cached portion is preserved
        if not torch.allclose(new_k[:, :, :prompt_len, :], k_cache, atol=1e-5):
            return False, "K cache not preserved during extension"
        if not torch.allclose(new_v[:, :, :prompt_len, :], v_cache, atol=1e-5):
            return False, "V cache not preserved during extension"
        
        return True, "MLA caching verified with value preservation"
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
        torch.manual_seed(42)
        x = torch.randn(batch, seq_len, config.d_model)
        
        rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
        cos, sin = rope.get_cos_sin(seq_len)
        
        out, cache = block(x, cos, sin)
        
        if out.shape != x.shape:
            return False, f"Output shape {out.shape} != {x.shape}"
        
        # Verify block transforms input
        if torch.allclose(out, x, atol=1e-3):
            return False, "Block output too similar to input"
        
        # Verify cache is returned
        if cache is None:
            return False, "Block should return KV cache"
        
        k_cache, v_cache = cache
        expected_cache_shape = (batch, config.num_kv_heads, seq_len, config.head_dim)
        if k_cache.shape != expected_cache_shape:
            return False, f"Cache shape {k_cache.shape} != {expected_cache_shape}"
        
        return True, f"Block verified with cache"
    except Exception as e:
        return False, str(e)


def test_model_init() -> Tuple[bool, str]:
    """Test full model initialization."""
    try:
        config = get_test_config()
        model = DeepSeekMathModel(config)
        
        if model.token_embedding is None:
            return False, "token_embedding not initialized"
        if model.rope is None:
            return False, "rope not initialized"
        if model.layers is None:
            return False, "layers not initialized"
        if model.final_norm is None:
            return False, "final_norm not initialized"
        
        if len(model.layers) != config.num_layers:
            return False, f"Wrong layer count: {len(model.layers)}"
        
        return True, f"Model initialized with {config.num_layers} layers"
    except Exception as e:
        return False, str(e)


def test_model_forward() -> Tuple[bool, str]:
    """Test full model forward pass."""
    try:
        config = get_test_config()
        model = DeepSeekMathModel(config)
        
        if model.layers is None:
            return False, "Model not initialized"
        
        batch, seq_len = 2, 16
        torch.manual_seed(42)
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        
        logits, caches = model(input_ids)
        
        expected_logits = (batch, seq_len, config.vocab_size)
        if logits.shape != expected_logits:
            return False, f"Logits shape {logits.shape} != {expected_logits}"
        
        if len(caches) != config.num_layers:
            return False, f"Cache count {len(caches)} != {config.num_layers}"
        
        # Verify logits are reasonable (not all zeros, not all same)
        if logits.abs().sum() == 0:
            return False, "Logits are all zeros"
        
        logits_std = logits.std()
        if logits_std < 1e-6:
            return False, "Logits have no variance"
        
        # Verify softmax produces valid probabilities
        probs = F.softmax(logits, dim=-1)
        prob_sums = probs.sum(dim=-1)
        if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5):
            return False, "Softmax of logits doesn't sum to 1"
        
        return True, f"Model verified with valid logits"
    except Exception as e:
        return False, str(e)


def test_model_with_cache() -> Tuple[bool, str]:
    """Test model generation with KV cache."""
    try:
        config = get_test_config()
        model = DeepSeekMathModel(config)
        
        if model.layers is None:
            return False, "Model not initialized"
        
        batch = 2
        torch.manual_seed(42)
        
        # Prefill
        prompt_len = 8
        prompt = torch.randint(0, config.vocab_size, (batch, prompt_len))
        logits1, caches = model(prompt)
        
        # Verify caches
        for i, cache in enumerate(caches):
            if cache is None:
                return False, f"Layer {i} cache is None"
        
        # Decode step
        new_token = torch.randint(0, config.vocab_size, (batch, 1))
        logits2, new_caches = model(new_token, kv_caches=caches, start_pos=prompt_len)
        
        if logits2.shape != (batch, 1, config.vocab_size):
            return False, f"Decode logits shape wrong: {logits2.shape}"
        
        # Verify cache is extended
        for i, (old_cache, new_cache) in enumerate(zip(caches, new_caches)):
            old_k, old_v = old_cache
            new_k, new_v = new_cache
            
            if new_k.shape[2] != prompt_len + 1:
                return False, f"Layer {i} cache not extended"
            
            # Old cache should be preserved in new cache
            if not torch.allclose(new_k[:, :, :prompt_len, :], old_k, atol=1e-5):
                return False, f"Layer {i} K cache not preserved"
        
        return True, "Cached generation verified with cache extension"
    except Exception as e:
        return False, str(e)


def test_model_gradient_flow() -> Tuple[bool, str]:
    """Test gradient flow through model."""
    try:
        config = get_test_config()
        model = DeepSeekMathModel(config)
        
        if model.layers is None:
            return False, "Model not initialized"
        
        batch, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        target_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        
        logits, _ = model(input_ids)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            target_ids.view(-1)
        )
        
        loss.backward()
        
        # Check gradients exist
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        param_count = sum(1 for p in model.parameters())
        
        if grad_count < param_count * 0.9:  # Allow some parameters without grad
            return False, f"Only {grad_count}/{param_count} have gradients"
        
        return True, f"Gradients flow to {grad_count}/{param_count} params"
    except Exception as e:
        return False, str(e)


def test_weight_tying() -> Tuple[bool, str]:
    """Test embedding weight tying."""
    try:
        config = get_test_config()
        config.tie_word_embeddings = True
        
        model = DeepSeekMathModel(config)
        
        if model.token_embedding is None:
            return False, "Model not initialized"
        
        # LM head should be None if tied
        if not config.tie_word_embeddings:
            return False, "Config should have tie_word_embeddings=True"
        
        # Output should use embedding weights
        batch, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        logits, _ = model(input_ids)
        
        if logits.shape[-1] != config.vocab_size:
            return False, f"Output vocab mismatch: {logits.shape[-1]}"
        
        return True, "Weight tying configured"
    except Exception as e:
        return False, str(e)


def test_small_config() -> Tuple[bool, str]:
    """Test small configuration creation."""
    try:
        config = create_deepseek_math_small_config()
        
        if config.d_model > 1024:
            return False, "Small config d_model too large"
        if config.num_layers > 12:
            return False, "Small config too many layers"
        
        return True, f"Small config: {config.d_model}d, {config.num_layers}L"
    except Exception as e:
        return False, str(e)


def test_7b_config() -> Tuple[bool, str]:
    """Test 7B configuration creation."""
    try:
        config = create_deepseek_math_7b_config()
        
        # 7B config should be substantial
        if config.d_model < 2048:
            return False, "7B config d_model too small"
        if config.num_layers < 20:
            return False, "7B config too few layers"
        
        # Verify head_dim is reasonable
        if config.head_dim < 64:
            return False, f"Head dim too small: {config.head_dim}"
        
        # Verify KV compression (latent_dim < full KV size)
        full_kv_size = config.num_kv_heads * config.head_dim
        if config.latent_dim >= full_kv_size:
            return False, "7B config should use KV compression"
        
        return True, f"7B config: {config.d_model}d, {config.num_layers}L, KV compression {full_kv_size}->{config.latent_dim}"
    except Exception as e:
        return False, str(e)


def test_model_stats() -> Tuple[bool, str]:
    """Test model statistics computation."""
    try:
        config = get_test_config()
        model = DeepSeekMathModel(config)
        
        if model.layers is None:
            return False, "Model not initialized"
        
        stats = analyze_model_stats(model)
        
        if stats['total_parameters'] == 0:
            return False, "Parameters not counted"
        if stats['memory_mb_fp16'] == 0:
            return False, "Memory not computed"
        
        return True, f"Stats computed: {stats['total_parameters']:,} params"
    except Exception as e:
        return False, str(e)


def test_parameter_breakdown() -> Tuple[bool, str]:
    """Test parameter count by component."""
    try:
        config = get_test_config()
        model = DeepSeekMathModel(config)
        
        if model.layers is None:
            return False, "Model not initialized"
        
        counts = count_parameters_by_component(model)
        
        if len(counts) == 0:
            return False, "No components counted"
        
        total_from_counts = sum(counts.values())
        if total_from_counts == 0:
            return False, "Component counts are zero"
        
        return True, f"Components: {list(counts.keys())}"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("config_properties", test_config_properties),
        ("token_embedding_init", test_token_embedding_init),
        ("token_embedding_forward", test_token_embedding_forward),
        ("token_embedding_scaling", test_token_embedding_scaling),
        ("rope_init", test_rope_init),
        ("rope_offset", test_rope_offset),
        ("apply_rotary_emb", test_apply_rotary_emb),
        ("mla_init", test_mla_init),
        ("mla_forward", test_mla_forward),
        ("mla_kv_cache", test_mla_kv_cache),
        ("deepseek_block_init", test_deepseek_block_init),
        ("deepseek_block_forward", test_deepseek_block_forward),
        ("model_init", test_model_init),
        ("model_forward", test_model_forward),
        ("model_with_cache", test_model_with_cache),
        ("model_gradient_flow", test_model_gradient_flow),
        ("weight_tying", test_weight_tying),
        ("small_config", test_small_config),
        ("7b_config", test_7b_config),
        ("model_stats", test_model_stats),
        ("parameter_breakdown", test_parameter_breakdown),
    ]
    
    print(f"\n{'='*60}")
    print("Day 31: DeepSeek Math Model Assembly - Tests")
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
