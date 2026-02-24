"""Test Suite for Day 31: DeepSeek Math Model Assembly"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
try:
    from day31 import (
        DeepSeekMathConfig, TokenEmbedding, RotaryEmbedding,
        MultiHeadLatentAttention, DeepSeekBlock, DeepSeekMathModel,
        create_deepseek_math_small_config, create_deepseek_math_7b_config,
        analyze_model_stats, count_parameters_by_component,
        apply_rotary_emb
    )
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
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


def test_config_properties():
    """Test config computed properties."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    
    expected_head_dim = config.d_model // config.num_heads
    assert config.head_dim == expected_head_dim, f"head_dim {config.head_dim} != {expected_head_dim}"


def test_token_embedding_init():
    """Test token embedding initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    embed = TokenEmbedding(config.vocab_size, config.d_model)
    
    assert embed.embedding is not None, "Embedding not initialized"
    
    assert embed.embedding.num_embeddings == config.vocab_size, "Wrong vocab size"
    assert embed.embedding.embedding_dim == config.d_model, "Wrong embedding dim"


def test_token_embedding_forward():
    """Test token embedding forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    embed = TokenEmbedding(config.vocab_size, config.d_model, scale=True)
    
    assert embed.embedding is not None, "Embedding not initialized"
    
    batch, seq_len = 2, 16
    torch.manual_seed(42)
    ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    
    out = embed(ids)
    
    expected = (batch, seq_len, config.d_model)
    assert out.shape == expected, f"Output shape {out.shape} != {expected}"
    
    import math
    raw_embed = embed.embedding(ids)
    expected_out = raw_embed * math.sqrt(config.d_model)
    
    assert torch.allclose(out, expected_out, atol=1e-5), "Embedding not scaled by sqrt(d_model)"


def test_token_embedding_scaling():
    """Test embedding scaling by sqrt(d_model)."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    
    embed_scaled = TokenEmbedding(config.vocab_size, config.d_model, scale=True)
    embed_unscaled = TokenEmbedding(config.vocab_size, config.d_model, scale=False)
    
    assert embed_scaled.embedding is not None and embed_unscaled.embedding is not None, "Embeddings not initialized"
    
    with torch.no_grad():
        embed_unscaled.embedding.weight.copy_(embed_scaled.embedding.weight)
    
    ids = torch.randint(0, config.vocab_size, (2, 16))
    
    out_scaled = embed_scaled(ids)
    out_unscaled = embed_unscaled(ids)
    
    expected_ratio = (config.d_model) ** 0.5
    actual_ratio = out_scaled.abs().mean() / out_unscaled.abs().mean()
    
    assert abs(actual_ratio - expected_ratio) <= 0.1, f"Scale ratio {actual_ratio:.2f} != {expected_ratio:.2f}"


def test_rope_init():
    """Test RoPE initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim = 32
    max_len = 128
    base = 10000.0
    rope = RotaryEmbedding(dim, max_len, base)
    
    cos, sin = rope.get_cos_sin(16)
    
    assert cos.shape == (16, dim // 2), f"cos shape {cos.shape} != (16, {dim // 2})"
    assert sin.shape == (16, dim // 2), f"sin shape {sin.shape} != (16, {dim // 2})"
    
    sum_sq = cos ** 2 + sin ** 2
    assert torch.allclose(sum_sq, torch.ones_like(sum_sq), atol=1e-5), "cos² + sin² should equal 1"
    
    assert torch.allclose(cos[0], torch.ones(dim // 2), atol=1e-5), "cos(0) should be 1 for all frequencies"
    assert torch.allclose(sin[0], torch.zeros(dim // 2), atol=1e-5), "sin(0) should be 0 for all frequencies"


def test_rope_offset():
    """Test RoPE with position offset."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim = 32
    rope = RotaryEmbedding(dim, 128)
    
    cos1, sin1 = rope.get_cos_sin(16, offset=0)
    
    cos2, sin2 = rope.get_cos_sin(16, offset=8)
    
    assert not torch.allclose(cos1, cos2), "Different offsets should give different rotations"
    
    cos1_overlap = rope.get_cos_sin(8, offset=8)[0]
    cos2_overlap = cos2[:8]
    
    assert torch.allclose(cos1_overlap, cos2_overlap, atol=1e-5), "Overlapping positions should match"


def test_apply_rotary_emb():
    """Test applying rotary embeddings."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, heads, seq_len, head_dim = 2, 4, 16, 32
    
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads // 2, seq_len, head_dim)
    
    rope = RotaryEmbedding(head_dim, 128)
    cos, sin = rope.get_cos_sin(seq_len)
    
    q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
    
    assert q_rot.shape == q.shape, f"Q shape changed: {q_rot.shape}"
    assert k_rot.shape == k.shape, f"K shape changed: {k_rot.shape}"
    
    assert not torch.allclose(q, q_rot), "RoPE didn't modify Q"
    
    q_norm = q.norm(dim=-1)
    q_rot_norm = q_rot.norm(dim=-1)
    assert torch.allclose(q_norm, q_rot_norm, atol=1e-4), "RoPE should preserve Q vector norms"
    
    k_norm = k.norm(dim=-1)
    k_rot_norm = k_rot.norm(dim=-1)
    assert torch.allclose(k_norm, k_rot_norm, atol=1e-4), "RoPE should preserve K vector norms"


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
    
    rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
    cos, sin = rope.get_cos_sin(seq_len)
    
    out, cache = mla(x, cos, sin)
    
    assert out.shape == x.shape, f"Output shape {out.shape} != {x.shape}"
    
    assert not torch.allclose(out, x, atol=1e-3), "MLA output too similar to input"
    
    latent = mla.W_kv_compress(x)
    assert latent.shape[-1] == config.latent_dim, f"KV compression not using latent_dim={config.latent_dim}"
    
    assert cache is not None, "Cache should be returned"
    
    k_cache, v_cache = cache
    expected_cache_shape = (batch, config.num_kv_heads, seq_len, config.head_dim)
    assert k_cache.shape == expected_cache_shape, f"K cache shape {k_cache.shape} != {expected_cache_shape}"


def test_mla_kv_cache():
    """Test MLA KV caching."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    mla = MultiHeadLatentAttention(config)
    
    assert mla.W_q is not None, "MLA not initialized"
    
    batch = 2
    torch.manual_seed(42)
    rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
    
    prompt_len = 8
    x1 = torch.randn(batch, prompt_len, config.d_model)
    cos1, sin1 = rope.get_cos_sin(prompt_len)
    out1, cache = mla(x1, cos1, sin1)
    
    assert cache is not None, "Cache not returned"
    
    k_cache, v_cache = cache
    expected_shape = (batch, config.num_kv_heads, prompt_len, config.head_dim)
    assert k_cache.shape == expected_shape, f"Cache shape {k_cache.shape} != {expected_shape}"
    
    x2 = torch.randn(batch, 1, config.d_model)
    cos2, sin2 = rope.get_cos_sin(1, offset=prompt_len)
    out2, new_cache = mla(x2, cos2, sin2, kv_cache=cache)
    
    assert new_cache is not None, "Cache not updated"
    
    new_k, new_v = new_cache
    assert new_k.shape[2] == prompt_len + 1, f"Cache not extended: {new_k.shape[2]}"
    
    assert torch.allclose(new_k[:, :, :prompt_len, :], k_cache, atol=1e-5), "K cache not preserved during extension"
    assert torch.allclose(new_v[:, :, :prompt_len, :], v_cache, atol=1e-5), "V cache not preserved during extension"


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
    
    rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
    cos, sin = rope.get_cos_sin(seq_len)
    
    out, cache = block(x, cos, sin)
    
    assert out.shape == x.shape, f"Output shape {out.shape} != {x.shape}"
    
    assert not torch.allclose(out, x, atol=1e-3), "Block output too similar to input"
    
    assert cache is not None, "Block should return KV cache"
    
    k_cache, v_cache = cache
    expected_cache_shape = (batch, config.num_kv_heads, seq_len, config.head_dim)
    assert k_cache.shape == expected_cache_shape, f"Cache shape {k_cache.shape} != {expected_cache_shape}"


def test_model_init():
    """Test full model initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    model = DeepSeekMathModel(config)
    
    assert model.token_embedding is not None, "token_embedding not initialized"
    assert model.rope is not None, "rope not initialized"
    assert model.layers is not None, "layers not initialized"
    assert model.final_norm is not None, "final_norm not initialized"
    
    assert len(model.layers) == config.num_layers, f"Wrong layer count: {len(model.layers)}"


def test_model_forward():
    """Test full model forward pass."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    model = DeepSeekMathModel(config)
    
    assert model.layers is not None, "Model not initialized"
    
    batch, seq_len = 2, 16
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    
    logits, caches = model(input_ids)
    
    expected_logits = (batch, seq_len, config.vocab_size)
    assert logits.shape == expected_logits, f"Logits shape {logits.shape} != {expected_logits}"
    
    assert len(caches) == config.num_layers, f"Cache count {len(caches)} != {config.num_layers}"
    
    assert logits.abs().sum() != 0, "Logits are all zeros"
    
    logits_std = logits.std()
    assert logits_std >= 1e-6, "Logits have no variance"
    
    probs = F.softmax(logits, dim=-1)
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "Softmax of logits doesn't sum to 1"


def test_model_with_cache():
    """Test model generation with KV cache."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    model = DeepSeekMathModel(config)
    
    assert model.layers is not None, "Model not initialized"
    
    batch = 2
    torch.manual_seed(42)
    
    prompt_len = 8
    prompt = torch.randint(0, config.vocab_size, (batch, prompt_len))
    logits1, caches = model(prompt)
    
    for i, cache in enumerate(caches):
        assert cache is not None, f"Layer {i} cache is None"
    
    new_token = torch.randint(0, config.vocab_size, (batch, 1))
    logits2, new_caches = model(new_token, kv_caches=caches, start_pos=prompt_len)
    
    assert logits2.shape == (batch, 1, config.vocab_size), f"Decode logits shape wrong: {logits2.shape}"
    
    for i, (old_cache, new_cache) in enumerate(zip(caches, new_caches)):
        old_k, old_v = old_cache
        new_k, new_v = new_cache
        
        assert new_k.shape[2] == prompt_len + 1, f"Layer {i} cache not extended"
        
        assert torch.allclose(new_k[:, :, :prompt_len, :], old_k, atol=1e-5), f"Layer {i} K cache not preserved"


def test_model_gradient_flow():
    """Test gradient flow through model."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    model = DeepSeekMathModel(config)
    
    assert model.layers is not None, "Model not initialized"
    
    batch, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    target_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    
    logits, _ = model(input_ids)
    
    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        target_ids.view(-1)
    )
    
    loss.backward()
    
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    param_count = sum(1 for p in model.parameters())
    
    assert grad_count >= param_count * 0.9, f"Only {grad_count}/{param_count} have gradients"


def test_weight_tying():
    """Test embedding weight tying."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    config.tie_word_embeddings = True
    
    model = DeepSeekMathModel(config)
    
    assert model.token_embedding is not None, "Model not initialized"
    
    assert config.tie_word_embeddings, "Config should have tie_word_embeddings=True"
    
    batch, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    logits, _ = model(input_ids)
    
    assert logits.shape[-1] == config.vocab_size, f"Output vocab mismatch: {logits.shape[-1]}"


def test_small_config():
    """Test small configuration creation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = create_deepseek_math_small_config()
    
    assert config.d_model <= 1024, "Small config d_model too large"
    assert config.num_layers <= 12, "Small config too many layers"


def test_7b_config():
    """Test 7B configuration creation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = create_deepseek_math_7b_config()
    
    assert config.d_model >= 2048, "7B config d_model too small"
    assert config.num_layers >= 20, "7B config too few layers"
    
    assert config.head_dim >= 64, f"Head dim too small: {config.head_dim}"
    
    full_kv_size = config.num_kv_heads * config.head_dim
    assert config.latent_dim < full_kv_size, "7B config should use KV compression"


def test_model_stats():
    """Test model statistics computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    model = DeepSeekMathModel(config)
    
    assert model.layers is not None, "Model not initialized"
    
    stats = analyze_model_stats(model)
    
    assert stats['total_parameters'] != 0, "Parameters not counted"
    assert stats['memory_mb_fp16'] != 0, "Memory not computed"


def test_parameter_breakdown():
    """Test parameter count by component."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    config = get_test_config()
    model = DeepSeekMathModel(config)
    
    assert model.layers is not None, "Model not initialized"
    
    counts = count_parameters_by_component(model)
    
    assert len(counts) != 0, "No components counted"
    
    total_from_counts = sum(counts.values())
    assert total_from_counts != 0, "Component counts are zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
