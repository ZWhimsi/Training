"""Test Suite for Day 22: Grouped Query Attention (GQA)"""

import torch
import torch.nn as nn
import sys
from typing import Tuple

try:
    from day22 import (
        compute_kv_memory_savings, repeat_kv, GQAProjection,
        GroupedQueryAttention, MultiQueryAttention, StandardMultiHeadAttention,
        compare_attention_memory, GQATransformerBlock
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_memory_savings_calculation() -> Tuple[bool, str]:
    """Test memory savings calculation."""
    try:
        result = compute_kv_memory_savings(32, 8)
        
        if result['mha_kv_heads'] != 32:
            return False, f"MHA heads wrong: {result['mha_kv_heads']}"
        if result['gqa_kv_heads'] != 8:
            return False, f"GQA heads wrong: {result['gqa_kv_heads']}"
        if result['savings_ratio'] != 4.0:
            return False, f"Savings ratio wrong: {result['savings_ratio']}"
        if result['heads_per_group'] != 4:
            return False, f"Heads per group wrong: {result['heads_per_group']}"
        
        return True, "Memory savings calculated correctly"
    except Exception as e:
        return False, str(e)


def test_repeat_kv_shape() -> Tuple[bool, str]:
    """Test repeat_kv output shape."""
    try:
        batch, num_kv_heads, seq_len, head_dim = 2, 4, 10, 64
        num_repeats = 4
        
        x = torch.randn(batch, num_kv_heads, seq_len, head_dim)
        output = repeat_kv(x, num_repeats)
        
        expected_shape = (batch, num_kv_heads * num_repeats, seq_len, head_dim)
        if output.shape != expected_shape:
            return False, f"Shape {output.shape} != {expected_shape}"
        
        return True, f"Output shape correct: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_repeat_kv_values() -> Tuple[bool, str]:
    """Test that repeat_kv correctly duplicates values."""
    try:
        batch, num_kv_heads, seq_len, head_dim = 2, 2, 4, 8
        num_repeats = 3
        
        x = torch.randn(batch, num_kv_heads, seq_len, head_dim)
        output = repeat_kv(x, num_repeats)
        
        # Check that values are correctly repeated
        # Output head 0 should equal input head 0
        # Output head 1 should equal input head 0 (repeat)
        # Output head 2 should equal input head 0 (repeat)
        # Output head 3 should equal input head 1
        # etc.
        
        for i in range(num_kv_heads):
            for j in range(num_repeats):
                output_idx = i * num_repeats + j
                if not torch.allclose(output[:, output_idx], x[:, i]):
                    return False, f"Head {output_idx} doesn't match source head {i}"
        
        return True, "Values correctly repeated"
    except Exception as e:
        return False, str(e)


def test_repeat_kv_no_repeat() -> Tuple[bool, str]:
    """Test repeat_kv with num_repeats=1 (no-op)."""
    try:
        x = torch.randn(2, 4, 10, 64)
        output = repeat_kv(x, 1)
        
        if output.shape != x.shape:
            return False, f"Shape changed with num_repeats=1"
        
        return True, "No-op case handled correctly"
    except Exception as e:
        return False, str(e)


def test_gqa_projection_shapes() -> Tuple[bool, str]:
    """Test GQAProjection output shapes."""
    try:
        d_model, num_heads, num_kv_heads = 256, 8, 2
        proj = GQAProjection(d_model, num_heads, num_kv_heads)
        
        if proj.W_q is None:
            return False, "W_q not initialized"
        if proj.W_k is None:
            return False, "W_k not initialized"
        if proj.W_v is None:
            return False, "W_v not initialized"
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, d_model)
        q, k, v = proj(x)
        
        head_dim = d_model // num_heads
        expected_q = (batch, num_heads, seq_len, head_dim)
        expected_kv = (batch, num_kv_heads, seq_len, head_dim)
        
        if q.shape != expected_q:
            return False, f"Q shape {q.shape} != {expected_q}"
        if k.shape != expected_kv:
            return False, f"K shape {k.shape} != {expected_kv}"
        if v.shape != expected_kv:
            return False, f"V shape {v.shape} != {expected_kv}"
        
        return True, f"Shapes: Q={q.shape}, K={k.shape}, V={v.shape}"
    except Exception as e:
        return False, str(e)


def test_gqa_projection_params() -> Tuple[bool, str]:
    """Test that GQA projection has fewer KV parameters than MHA."""
    try:
        d_model, num_heads, num_kv_heads = 256, 8, 2
        proj = GQAProjection(d_model, num_heads, num_kv_heads)
        
        if proj.W_q is None or proj.W_k is None:
            return False, "Projections not initialized"
        
        q_params = proj.W_q.weight.numel()
        k_params = proj.W_k.weight.numel()
        v_params = proj.W_v.weight.numel()
        
        # Q should have full size, K and V should be smaller
        expected_q = d_model * d_model
        expected_kv = d_model * (num_kv_heads * (d_model // num_heads))
        
        if q_params != expected_q:
            return False, f"Q params {q_params} != {expected_q}"
        if k_params != expected_kv:
            return False, f"K params {k_params} != {expected_kv}"
        
        savings = (expected_q - expected_kv) / expected_q * 100
        return True, f"KV params are {savings:.1f}% smaller than Q"
    except Exception as e:
        return False, str(e)


def test_gqa_forward() -> Tuple[bool, str]:
    """Test GroupedQueryAttention forward pass."""
    try:
        d_model, num_heads, num_kv_heads = 256, 8, 2
        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        
        if gqa.projection is None:
            return False, "GQA projection not initialized"
        
        batch, seq_len = 2, 16
        x = torch.randn(batch, seq_len, d_model)
        output, attn = gqa(x)
        
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != input {x.shape}"
        
        expected_attn = (batch, num_heads, seq_len, seq_len)
        if attn.shape != expected_attn:
            return False, f"Attention shape {attn.shape} != {expected_attn}"
        
        return True, "GQA forward pass works"
    except Exception as e:
        return False, str(e)


def test_gqa_attention_valid() -> Tuple[bool, str]:
    """Test that GQA attention weights are valid (sum to 1)."""
    try:
        d_model, num_heads, num_kv_heads = 128, 4, 2
        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        
        if gqa.projection is None:
            return False, "GQA not initialized"
        
        x = torch.randn(2, 8, d_model)
        _, attn = gqa(x)
        
        # Attention weights should sum to 1 along last dimension
        attn_sum = attn.sum(dim=-1)
        if not torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5):
            return False, f"Attention doesn't sum to 1"
        
        return True, "Attention weights valid"
    except Exception as e:
        return False, str(e)


def test_gqa_with_mask() -> Tuple[bool, str]:
    """Test GQA with causal mask."""
    try:
        d_model, num_heads, num_kv_heads = 128, 4, 2
        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        
        if gqa.projection is None:
            return False, "GQA not initialized"
        
        batch, seq_len = 2, 8
        x = torch.randn(batch, seq_len, d_model)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        
        output, attn = gqa(x, mask)
        
        # Check that upper triangle of attention is ~0 (masked out)
        upper = torch.triu(attn, diagonal=1)
        if upper.abs().max() > 1e-5:
            return False, "Causal mask not applied correctly"
        
        return True, "Causal mask works correctly"
    except Exception as e:
        return False, str(e)


def test_mqa_is_gqa_special_case() -> Tuple[bool, str]:
    """Test that MQA is GQA with 1 KV head."""
    try:
        d_model, num_heads = 128, 4
        mqa = MultiQueryAttention(d_model, num_heads)
        
        if mqa.attention is None:
            return False, "MQA attention not initialized"
        
        # Check that internal GQA has 1 KV head
        if mqa.attention.num_kv_heads != 1:
            return False, f"MQA should have 1 KV head, got {mqa.attention.num_kv_heads}"
        
        x = torch.randn(2, 8, d_model)
        output, attn = mqa(x)
        
        if output.shape != x.shape:
            return False, f"Output shape wrong"
        
        return True, "MQA correctly uses GQA with 1 KV head"
    except Exception as e:
        return False, str(e)


def test_gqa_vs_mha_output_shape() -> Tuple[bool, str]:
    """Test that GQA and MHA produce same output shape."""
    try:
        d_model, num_heads = 128, 4
        num_kv_heads = 2
        
        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        mha = StandardMultiHeadAttention(d_model, num_heads)
        
        if gqa.projection is None:
            return False, "GQA not initialized"
        
        x = torch.randn(2, 8, d_model)
        
        gqa_out, gqa_attn = gqa(x)
        mha_out, mha_attn = mha(x)
        
        if gqa_out.shape != mha_out.shape:
            return False, f"Output shapes differ: GQA={gqa_out.shape}, MHA={mha_out.shape}"
        if gqa_attn.shape != mha_attn.shape:
            return False, f"Attention shapes differ"
        
        return True, "GQA and MHA have same output shapes"
    except Exception as e:
        return False, str(e)


def test_memory_comparison() -> Tuple[bool, str]:
    """Test memory comparison function."""
    try:
        result = compare_attention_memory(256, 8, seq_len=1024, batch_size=32)
        
        if result['mha_kv_bytes'] == 0:
            return False, "MHA memory not computed"
        if result['gqa_kv_bytes'] == 0:
            return False, "GQA memory not computed"
        if result['mqa_kv_bytes'] == 0:
            return False, "MQA memory not computed"
        
        # MHA should use most memory
        if result['mha_kv_bytes'] <= result['gqa_kv_bytes']:
            return False, "MHA should use more memory than GQA"
        if result['gqa_kv_bytes'] <= result['mqa_kv_bytes']:
            return False, "GQA should use more memory than MQA"
        
        return True, f"MQA is {result['mqa_savings_vs_mha']:.1f}x more efficient than MHA"
    except Exception as e:
        return False, str(e)


def test_gqa_transformer_block() -> Tuple[bool, str]:
    """Test GQA Transformer block."""
    try:
        d_model, num_heads, num_kv_heads = 128, 4, 2
        block = GQATransformerBlock(d_model, num_heads, num_kv_heads)
        
        if block.attention is None:
            return False, "Attention not initialized"
        if block.norm1 is None:
            return False, "norm1 not initialized"
        
        x = torch.randn(2, 8, d_model)
        output = block(x)
        
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != {x.shape}"
        
        return True, "GQA Transformer block works"
    except Exception as e:
        return False, str(e)


def test_gqa_gradient_flow() -> Tuple[bool, str]:
    """Test that gradients flow through GQA."""
    try:
        d_model, num_heads, num_kv_heads = 128, 4, 2
        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        
        if gqa.projection is None:
            return False, "GQA not initialized"
        
        x = torch.randn(2, 8, d_model, requires_grad=True)
        output, _ = gqa(x)
        loss = output.sum()
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
        ("memory_savings_calc", test_memory_savings_calculation),
        ("repeat_kv_shape", test_repeat_kv_shape),
        ("repeat_kv_values", test_repeat_kv_values),
        ("repeat_kv_no_repeat", test_repeat_kv_no_repeat),
        ("gqa_projection_shapes", test_gqa_projection_shapes),
        ("gqa_projection_params", test_gqa_projection_params),
        ("gqa_forward", test_gqa_forward),
        ("gqa_attention_valid", test_gqa_attention_valid),
        ("gqa_with_mask", test_gqa_with_mask),
        ("mqa_special_case", test_mqa_is_gqa_special_case),
        ("gqa_vs_mha_shape", test_gqa_vs_mha_output_shape),
        ("memory_comparison", test_memory_comparison),
        ("gqa_transformer_block", test_gqa_transformer_block),
        ("gqa_gradient_flow", test_gqa_gradient_flow),
    ]
    
    print(f"\n{'='*50}\nDay 22: Grouped Query Attention - Tests\n{'='*50}")
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    run_all_tests()
