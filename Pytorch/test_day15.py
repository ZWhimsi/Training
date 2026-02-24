"""Test Suite for Day 15: Self-Attention"""

import torch
import pytest
import torch.nn.functional as F
try:
    from day15 import (scaled_dot_product_attention, SelfAttention, 
                       create_causal_mask, causal_attention)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_attention_basic():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    assert output is not None, "output is None"
    assert weights is not None, "weights is None"
    
    assert output.shape == (2, 4, 8), f"Output shape {output.shape}, expected (2,4,8)"
    assert weights.shape == (2, 4, 4), f"Weights shape {weights.shape}, expected (2,4,4)"
    
    d_k = Q.shape[-1]
    expected_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    expected_weights = F.softmax(expected_scores, dim=-1)
    expected_output = torch.matmul(expected_weights, V)
    
    assert torch.allclose(weights, expected_weights, atol=1e-5), f"Weights mismatch: max diff {(weights - expected_weights).abs().max():.6f}"
    
    assert torch.allclose(output, expected_output, atol=1e-5), f"Output mismatch: max diff {(output - expected_output).abs().max():.6f}"

def test_attention_weights_sum_to_one():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    Q = torch.randn(1, 4, 8)
    K = torch.randn(1, 4, 8)
    V = torch.randn(1, 4, 8)
    
    _, weights = scaled_dot_product_attention(Q, K, V)
    
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Weights don't sum to 1"

def test_attention_vs_pytorch():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)
    
    our_output, _ = scaled_dot_product_attention(Q, K, V)
    
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    ref_output = torch.matmul(weights, V)
    
    assert torch.allclose(our_output, ref_output, atol=1e-5), "Doesn't match reference"

def test_self_attention_module():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    d_model = 16
    module = SelfAttention(d_model)
    
    assert module.W_q is not None, "W_q not initialized"
    
    x = torch.randn(2, 8, d_model)
    output = module(x)
    
    assert output is not None, "output is None"
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    
    with torch.no_grad():
        Q = module.W_q(x)
        K = module.W_k(x)
        V = module.W_v(x)
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        expected = module.W_o(attn_output)
    
    assert torch.allclose(output, expected, atol=1e-5), f"Output doesn't match expected: max diff {(output - expected).abs().max():.6f}"

def test_causal_mask():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    mask = create_causal_mask(4)
    
    assert mask is not None, "mask is None"
    
    expected = torch.tensor([[[1., 0., 0., 0.],
                              [1., 1., 0., 0.],
                              [1., 1., 1., 0.],
                              [1., 1., 1., 1.]]])
    
    assert torch.equal(mask, expected), "Mask incorrect"

def test_causal_attention():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    Q = torch.randn(1, 4, 8)
    K = torch.randn(1, 4, 8)
    V = torch.randn(1, 4, 8)
    
    output, weights = causal_attention(Q, K, V)
    
    upper = torch.triu(weights[0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-5), "Future positions have non-zero attention"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
