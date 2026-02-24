"""Test Suite for Day 22: Grouped Query Attention (GQA)"""

import torch
import pytest
import torch.nn as nn
try:
    from day22 import (
        compute_kv_memory_savings, repeat_kv, GQAProjection,
        GroupedQueryAttention, MultiQueryAttention, StandardMultiHeadAttention,
        compare_attention_memory, GQATransformerBlock
    )
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_memory_savings_calculation():
    """Test memory savings calculation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = compute_kv_memory_savings(32, 8)
    
    assert result['mha_kv_heads'] == 32, f"MHA heads wrong: {result['mha_kv_heads']}"
    assert result['gqa_kv_heads'] == 8, f"GQA heads wrong: {result['gqa_kv_heads']}"
    assert result['savings_ratio'] == 4.0, f"Savings ratio wrong: {result['savings_ratio']}"
    assert result['heads_per_group'] == 4, f"Heads per group wrong: {result['heads_per_group']}"

def test_repeat_kv_shape():
    """Test repeat_kv output shape and memory layout."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, num_kv_heads, seq_len, head_dim = 2, 4, 10, 64
    num_repeats = 4
    
    x = torch.randn(batch, num_kv_heads, seq_len, head_dim)
    output = repeat_kv(x, num_repeats)
    
    expected_shape = (batch, num_kv_heads * num_repeats, seq_len, head_dim)
    assert output.shape == expected_shape, f"Shape {output.shape} != {expected_shape}"
    
    assert output.is_contiguous(), "Output is not contiguous"

def test_repeat_kv_values():
    """Test that repeat_kv correctly duplicates values."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, num_kv_heads, seq_len, head_dim = 2, 2, 4, 8
    num_repeats = 3
    
    x = torch.randn(batch, num_kv_heads, seq_len, head_dim)
    output = repeat_kv(x, num_repeats)
    
    for i in range(num_kv_heads):
        for j in range(num_repeats):
            output_idx = i * num_repeats + j
            assert torch.allclose(output[:, output_idx], x[:, i]), f"Head {output_idx} doesn't match source head {i}"

def test_repeat_kv_no_repeat():
    """Test repeat_kv with num_repeats=1 returns same values."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    x = torch.randn(2, 4, 10, 64)
    output = repeat_kv(x, 1)
    
    assert output.shape == x.shape, "Shape changed with num_repeats=1"
    
    assert torch.allclose(output, x, atol=1e-6), "Output differs from input with num_repeats=1"

def test_gqa_projection_shapes():
    """Test GQAProjection output shapes and computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 256, 8, 2
    proj = GQAProjection(d_model, num_heads, num_kv_heads)
    
    assert proj.W_q is not None, "W_q not initialized"
    assert proj.W_k is not None, "W_k not initialized"
    assert proj.W_v is not None, "W_v not initialized"
    
    batch, seq_len = 2, 16
    x = torch.randn(batch, seq_len, d_model)
    q, k, v = proj(x)
    
    head_dim = d_model // num_heads
    expected_q = (batch, num_heads, seq_len, head_dim)
    expected_kv = (batch, num_kv_heads, seq_len, head_dim)
    
    assert q.shape == expected_q, f"Q shape {q.shape} != {expected_q}"
    assert k.shape == expected_kv, f"K shape {k.shape} != {expected_kv}"
    assert v.shape == expected_kv, f"V shape {v.shape} != {expected_kv}"
    
    x2 = torch.randn(batch, seq_len, d_model)
    q2, k2, v2 = proj(x2)
    
    assert not torch.allclose(q, q2, atol=1e-4), "Different inputs produce same Q"

def test_gqa_projection_params():
    """Test that GQA projection has fewer KV parameters than MHA."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 256, 8, 2
    proj = GQAProjection(d_model, num_heads, num_kv_heads)
    
    assert proj.W_q is not None and proj.W_k is not None, "Projections not initialized"
    
    q_params = proj.W_q.weight.numel()
    k_params = proj.W_k.weight.numel()
    v_params = proj.W_v.weight.numel()
    
    expected_q = d_model * d_model
    expected_kv = d_model * (num_kv_heads * (d_model // num_heads))
    
    assert q_params == expected_q, f"Q params {q_params} != {expected_q}"
    assert k_params == expected_kv, f"K params {k_params} != {expected_kv}"

def test_gqa_forward():
    """Test GroupedQueryAttention forward pass and output projection."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 256, 8, 2
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout=0.0)
    
    assert gqa.projection is not None, "GQA projection not initialized"
    assert gqa.W_o is not None, "Output projection W_o not initialized"
    
    batch, seq_len = 2, 16
    x = torch.randn(batch, seq_len, d_model)
    output, attn = gqa(x)
    
    assert output.shape == x.shape, f"Output shape {output.shape} != input {x.shape}"
    
    expected_attn = (batch, num_heads, seq_len, seq_len)
    assert attn.shape == expected_attn, f"Attention shape {attn.shape} != {expected_attn}"
    
    assert not torch.allclose(output, x, atol=1e-3), "Output identical to input (no attention applied)"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Output contains NaN or Inf"

def test_gqa_attention_valid():
    """Test that GQA attention weights are valid (sum to 1)."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 128, 4, 2
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
    
    assert gqa.projection is not None, "GQA not initialized"
    
    x = torch.randn(2, 8, d_model)
    _, attn = gqa(x)
    
    attn_sum = attn.sum(dim=-1)
    assert torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5), "Attention doesn't sum to 1"

def test_gqa_with_mask():
    """Test GQA with causal mask."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 128, 4, 2
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
    
    assert gqa.projection is not None, "GQA not initialized"
    
    batch, seq_len = 2, 8
    x = torch.randn(batch, seq_len, d_model)
    
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    output, attn = gqa(x, mask)
    
    upper = torch.triu(attn, diagonal=1)
    assert upper.abs().max() <= 1e-5, "Causal mask not applied correctly"

def test_mqa_is_gqa_special_case():
    """Test that MQA is GQA with 1 KV head."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 128, 4
    mqa = MultiQueryAttention(d_model, num_heads)
    
    assert mqa.attention is not None, "MQA attention not initialized"
    
    assert mqa.attention.num_kv_heads == 1, f"MQA should have 1 KV head, got {mqa.attention.num_kv_heads}"
    
    x = torch.randn(2, 8, d_model)
    output, attn = mqa(x)
    
    assert output.shape == x.shape, "Output shape wrong"

def test_gqa_vs_mha_output_shape():
    """Test that GQA and MHA produce same output shape with valid attention."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 128, 4
    num_kv_heads = 2
    
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads, dropout=0.0)
    mha = StandardMultiHeadAttention(d_model, num_heads, dropout=0.0)
    
    assert gqa.projection is not None, "GQA not initialized"
    
    x = torch.randn(2, 8, d_model)
    
    gqa_out, gqa_attn = gqa(x)
    mha_out, mha_attn = mha(x)
    
    assert gqa_out.shape == mha_out.shape, f"Output shapes differ: GQA={gqa_out.shape}, MHA={mha_out.shape}"
    assert gqa_attn.shape == mha_attn.shape, "Attention shapes differ"
    
    gqa_sum = gqa_attn.sum(dim=-1)
    mha_sum = mha_attn.sum(dim=-1)
    
    assert torch.allclose(gqa_sum, torch.ones_like(gqa_sum), atol=1e-5), "GQA attention doesn't sum to 1"
    assert torch.allclose(mha_sum, torch.ones_like(mha_sum), atol=1e-5), "MHA attention doesn't sum to 1"

def test_memory_comparison():
    """Test memory comparison function."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = compare_attention_memory(256, 8, seq_len=1024, batch_size=32)
    
    assert result['mha_kv_bytes'] != 0, "MHA memory not computed"
    assert result['gqa_kv_bytes'] != 0, "GQA memory not computed"
    assert result['mqa_kv_bytes'] != 0, "MQA memory not computed"
    
    assert result['mha_kv_bytes'] >= result['gqa_kv_bytes'], "MHA should use more memory than GQA"
    assert result['gqa_kv_bytes'] >= result['mqa_kv_bytes'], "GQA should use more memory than MQA"

def test_gqa_transformer_block():
    """Test GQA Transformer block with residual connections."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 128, 4, 2
    block = GQATransformerBlock(d_model, num_heads, num_kv_heads, dropout=0.0)
    
    assert block.attention is not None, "Attention not initialized"
    assert block.norm1 is not None, "norm1 not initialized"
    assert block.norm2 is not None, "norm2 not initialized"
    assert block.ffn is not None, "ffn not initialized"
    
    x = torch.randn(2, 8, d_model)
    output = block(x)
    
    assert output.shape == x.shape, f"Output shape {output.shape} != {x.shape}"
    
    correlation = torch.corrcoef(torch.stack([x.flatten(), output.flatten()]))[0, 1]
    assert correlation >= 0.05, f"Weak residual: corr={correlation:.4f}"
    
    assert not torch.allclose(output, x, atol=1e-3), "Output identical to input"

def test_gqa_gradient_flow():
    """Test that gradients flow through GQA."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, num_kv_heads = 128, 4, 2
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
    
    assert gqa.projection is not None, "GQA not initialized"
    
    x = torch.randn(2, 8, d_model, requires_grad=True)
    output, _ = gqa(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient to input"
    assert x.grad.abs().sum() != 0, "Gradients are zero"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
