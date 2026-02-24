"""Test Suite for Day 15: Self-Attention"""

import torch
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day15 import (scaled_dot_product_attention, SelfAttention, 
                       create_causal_mask, causal_attention)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_attention_basic() -> Tuple[bool, str]:
    try:
        torch.manual_seed(42)
        Q = torch.randn(2, 4, 8)
        K = torch.randn(2, 4, 8)
        V = torch.randn(2, 4, 8)
        
        output, weights = scaled_dot_product_attention(Q, K, V)
        
        if output is None:
            return False, "output is None"
        if weights is None:
            return False, "weights is None"
        
        if output.shape != (2, 4, 8):
            return False, f"Output shape {output.shape}, expected (2,4,8)"
        if weights.shape != (2, 4, 4):
            return False, f"Weights shape {weights.shape}, expected (2,4,4)"
        
        # Validate actual computation
        d_k = Q.shape[-1]
        expected_scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        expected_weights = F.softmax(expected_scores, dim=-1)
        expected_output = torch.matmul(expected_weights, V)
        
        if not torch.allclose(weights, expected_weights, atol=1e-5):
            return False, f"Weights mismatch: max diff {(weights - expected_weights).abs().max():.6f}"
        
        if not torch.allclose(output, expected_output, atol=1e-5):
            return False, f"Output mismatch: max diff {(output - expected_output).abs().max():.6f}"
        
        return True, f"Shapes and values OK"
    except Exception as e:
        return False, str(e)


def test_attention_weights_sum_to_one() -> Tuple[bool, str]:
    try:
        Q = torch.randn(1, 4, 8)
        K = torch.randn(1, 4, 8)
        V = torch.randn(1, 4, 8)
        
        _, weights = scaled_dot_product_attention(Q, K, V)
        
        # Each row should sum to 1
        row_sums = weights.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            return False, "Weights don't sum to 1"
        
        return True, "Weights sum to 1 ✓"
    except Exception as e:
        return False, str(e)


def test_attention_vs_pytorch() -> Tuple[bool, str]:
    try:
        Q = torch.randn(2, 4, 8)
        K = torch.randn(2, 4, 8)
        V = torch.randn(2, 4, 8)
        
        our_output, _ = scaled_dot_product_attention(Q, K, V)
        
        # Manual PyTorch reference
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        weights = F.softmax(scores, dim=-1)
        ref_output = torch.matmul(weights, V)
        
        if not torch.allclose(our_output, ref_output, atol=1e-5):
            return False, "Doesn't match reference"
        
        return True, "Matches PyTorch reference"
    except Exception as e:
        return False, str(e)


def test_self_attention_module() -> Tuple[bool, str]:
    try:
        torch.manual_seed(42)
        d_model = 16
        module = SelfAttention(d_model)
        
        if module.W_q is None:
            return False, "W_q not initialized"
        
        x = torch.randn(2, 8, d_model)
        output = module(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape mismatch: {output.shape} vs {x.shape}"
        
        # Validate computation: Q, K, V projections -> attention -> output projection
        with torch.no_grad():
            Q = module.W_q(x)
            K = module.W_k(x)
            V = module.W_v(x)
            d_k = Q.shape[-1]
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)
            expected = module.W_o(attn_output)
        
        if not torch.allclose(output, expected, atol=1e-5):
            return False, f"Output doesn't match expected: max diff {(output - expected).abs().max():.6f}"
        
        return True, f"SelfAttention module OK"
    except Exception as e:
        return False, str(e)


def test_causal_mask() -> Tuple[bool, str]:
    try:
        mask = create_causal_mask(4)
        
        if mask is None:
            return False, "mask is None"
        
        expected = torch.tensor([[[1., 0., 0., 0.],
                                  [1., 1., 0., 0.],
                                  [1., 1., 1., 0.],
                                  [1., 1., 1., 1.]]])
        
        if not torch.equal(mask, expected):
            return False, f"Mask incorrect"
        
        return True, "Causal mask correct (lower triangular)"
    except Exception as e:
        return False, str(e)


def test_causal_attention() -> Tuple[bool, str]:
    try:
        Q = torch.randn(1, 4, 8)
        K = torch.randn(1, 4, 8)
        V = torch.randn(1, 4, 8)
        
        output, weights = causal_attention(Q, K, V)
        
        # Check that future positions have zero attention
        # Position 0 should only attend to position 0
        # Position 1 should only attend to positions 0, 1
        
        # Upper triangle should be zero
        upper = torch.triu(weights[0], diagonal=1)
        if not torch.allclose(upper, torch.zeros_like(upper), atol=1e-5):
            return False, "Future positions have non-zero attention"
        
        return True, "Causal attention masks future"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("attention_basic", test_attention_basic),
        ("weights_sum", test_attention_weights_sum_to_one),
        ("vs_pytorch", test_attention_vs_pytorch),
        ("self_attention_module", test_self_attention_module),
        ("causal_mask", test_causal_mask),
        ("causal_attention", test_causal_attention),
    ]
    
    print(f"\n{'='*50}\nDay 15: Self-Attention - Tests\n{'='*50}")
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
