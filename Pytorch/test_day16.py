"""Test Suite for Day 16: Multi-Head Attention"""

import torch
import pytest
import torch.nn as nn
import math
try:
    from day16 import (split_heads, merge_heads, multi_head_attention_scores,
                       MultiHeadAttention, MultiHeadSelfAttention, CrossAttention)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_split_heads_shape():
    """Test that split_heads produces correct output shape and values."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    batch, seq, d_model = 2, 8, 64
    num_heads = 4
    d_k = d_model // num_heads
    x = torch.randn(batch, seq, d_model)
    
    result = split_heads(x, num_heads)
    
    assert result is not None, "split_heads returned None"
    
    expected_shape = (batch, num_heads, seq, d_k)
    assert result.shape == expected_shape, f"Shape {result.shape}, expected {expected_shape}"
    
    for h in range(num_heads):
        expected_slice = x[:, :, h*d_k:(h+1)*d_k]
        actual_slice = result[:, h, :, :]
        assert torch.allclose(actual_slice, expected_slice, atol=1e-6), f"Values for head {h} don't match expected"

def test_merge_heads_shape():
    """Test that merge_heads produces correct output shape and values."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    batch, num_heads, seq, d_k = 2, 4, 8, 16
    d_model = num_heads * d_k
    x = torch.randn(batch, num_heads, seq, d_k)
    
    result = merge_heads(x)
    
    assert result is not None, "merge_heads returned None"
    
    expected_shape = (batch, seq, d_model)
    assert result.shape == expected_shape, f"Shape {result.shape}, expected {expected_shape}"
    
    for h in range(num_heads):
        expected_slice = x[:, h, :, :]
        actual_slice = result[:, :, h*d_k:(h+1)*d_k]
        assert torch.allclose(actual_slice, expected_slice, atol=1e-6), f"Values for head {h} don't match expected"

def test_split_merge_inverse():
    """Test that merge_heads(split_heads(x)) == x."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq, d_model = 2, 8, 64
    num_heads = 4
    x = torch.randn(batch, seq, d_model)
    
    split = split_heads(x, num_heads)
    assert split is not None, "split_heads returned None"
        
    merged = merge_heads(split)
    assert merged is not None, "merge_heads returned None"
    
    assert torch.allclose(x, merged, atol=1e-6), "merge(split(x)) != x"

def test_mha_scores_shape():
    """Test multi-head attention scores computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    batch, num_heads, seq, d_k = 2, 4, 8, 16
    Q = torch.randn(batch, num_heads, seq, d_k)
    K = torch.randn(batch, num_heads, seq, d_k)
    V = torch.randn(batch, num_heads, seq, d_k)
    
    output, weights = multi_head_attention_scores(Q, K, V)
    
    assert output is not None, "output is None"
    assert weights is not None, "weights is None"
    
    assert output.shape == (batch, num_heads, seq, d_k), f"Output shape {output.shape} incorrect"
    assert weights.shape == (batch, num_heads, seq, seq), f"Weights shape {weights.shape} incorrect"
    
    expected_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    expected_weights = torch.softmax(expected_scores, dim=-1)
    expected_output = torch.matmul(expected_weights, V)
    
    assert torch.allclose(weights, expected_weights, atol=1e-5), f"Weights mismatch: max diff {(weights - expected_weights).abs().max():.6f}"
    
    assert torch.allclose(output, expected_output, atol=1e-5), f"Output mismatch: max diff {(output - expected_output).abs().max():.6f}"

def test_mha_weights_sum_to_one():
    """Test that attention weights sum to 1."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, num_heads, seq, d_k = 2, 4, 8, 16
    Q = torch.randn(batch, num_heads, seq, d_k)
    K = torch.randn(batch, num_heads, seq, d_k)
    V = torch.randn(batch, num_heads, seq, d_k)
    
    _, weights = multi_head_attention_scores(Q, K, V)
    
    assert weights is not None, "weights is None"
    
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Weights don't sum to 1"

def test_multihead_attention_module():
    """Test the MultiHeadAttention module."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    d_model, num_heads = 64, 4
    d_k = d_model // num_heads
    batch, seq = 2, 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    assert mha.W_q is not None, "W_q not initialized"
    assert mha.W_k is not None, "W_k not initialized"
    assert mha.W_v is not None, "W_v not initialized"
    assert mha.W_o is not None, "W_o not initialized"
    
    x = torch.randn(batch, seq, d_model)
    output, weights = mha(x, x, x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    assert weights.shape == (batch, num_heads, seq, seq), f"Weights shape {weights.shape} incorrect"
    
    with torch.no_grad():
        Q = mha.W_q(x)
        K = mha.W_k(x)
        V = mha.W_v(x)
        
        Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)
        K = K.view(batch, seq, num_heads, d_k).transpose(1, 2)
        V = V.view(batch, seq, num_heads, d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        expected_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(expected_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, d_model)
        expected = mha.W_o(attn_out)
    
    assert torch.allclose(output, expected, atol=1e-5), f"Output mismatch: max diff {(output - expected).abs().max():.6f}"

def test_self_attention_module():
    """Test the MultiHeadSelfAttention module."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    d_model, num_heads = 64, 4
    batch, seq = 2, 8
    
    self_attn = MultiHeadSelfAttention(d_model, num_heads)
    
    assert self_attn.attention is not None, "attention module not initialized"
    
    x = torch.randn(batch, seq, d_model)
    output, weights = self_attn(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    
    expected_output, expected_weights = self_attn.attention(x, x, x)
    
    assert torch.allclose(output, expected_output, atol=1e-5), f"Output mismatch with attention(x,x,x): max diff {(output - expected_output).abs().max():.6f}"
    
    if weights is not None and expected_weights is not None:
        assert torch.allclose(weights, expected_weights, atol=1e-5), "Weights mismatch with attention(x,x,x)"

def test_cross_attention():
    """Test cross-attention with different sequence lengths."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    d_model, num_heads = 64, 4
    batch = 2
    seq_q, seq_k = 6, 10
    
    cross_attn = CrossAttention(d_model, num_heads)
    
    query = torch.randn(batch, seq_q, d_model)
    kv = torch.randn(batch, seq_k, d_model)
    
    output, weights = cross_attn(query, kv)
    
    assert output is not None, "output is None"
    assert output.shape == (batch, seq_q, d_model), f"Output shape {output.shape} incorrect"
    assert weights is None or weights.shape == (batch, num_heads, seq_q, seq_k), f"Weights shape {weights.shape} incorrect"
    
    expected_output, expected_weights = cross_attn.attention(query, kv, kv)
    
    assert torch.allclose(output, expected_output, atol=1e-5), f"Output mismatch with attention(q, kv, kv): max diff {(output - expected_output).abs().max():.6f}"

def test_mha_vs_reference():
    """Test against a manual reference implementation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 64, 4
    d_k = d_model // num_heads
    batch, seq = 2, 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq, d_model)
    
    our_output, _ = mha(x, x, x)
    
    assert our_output is not None, "output is None"
    
    with torch.no_grad():
        Q = mha.W_q(x)
        K = mha.W_k(x)
        V = mha.W_v(x)
        
        Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)
        K = K.view(batch, seq, num_heads, d_k).transpose(1, 2)
        V = V.view(batch, seq, num_heads, d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, V)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, d_model)
        ref_output = mha.W_o(attn_out)
    
    assert torch.allclose(our_output, ref_output, atol=1e-5), "Doesn't match reference implementation"

def test_different_head_counts():
    """Test with various head configurations."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    d_model = 64
    batch, seq = 2, 8
    
    for num_heads in [1, 2, 4, 8]:
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch, seq, d_model)
        
        output, weights = mha(x, x, x)
        
        assert output is not None and output.shape == x.shape, f"Failed for num_heads={num_heads}"
        assert weights.shape[1] == num_heads, f"Wrong head count in weights for num_heads={num_heads}"
        
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), f"Weights don't sum to 1 for num_heads={num_heads}"
        
        assert not torch.allclose(output, torch.zeros_like(output)), f"Output is all zeros for num_heads={num_heads}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
