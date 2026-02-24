"""Test Suite for Day 16: Multi-Head Attention"""

import torch
import torch.nn as nn
import sys
from typing import Tuple

try:
    from day16 import (split_heads, merge_heads, multi_head_attention_scores,
                       MultiHeadAttention, MultiHeadSelfAttention, CrossAttention)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_split_heads_shape() -> Tuple[bool, str]:
    """Test that split_heads produces correct output shape and values."""
    try:
        torch.manual_seed(42)
        batch, seq, d_model = 2, 8, 64
        num_heads = 4
        d_k = d_model // num_heads
        x = torch.randn(batch, seq, d_model)
        
        result = split_heads(x, num_heads)
        
        if result is None:
            return False, "split_heads returned None"
        
        expected_shape = (batch, num_heads, seq, d_k)
        if result.shape != expected_shape:
            return False, f"Shape {result.shape}, expected {expected_shape}"
        
        # Validate actual values: result[b, h, s, :] should equal x[b, s, h*d_k:(h+1)*d_k]
        for h in range(num_heads):
            expected_slice = x[:, :, h*d_k:(h+1)*d_k]  # [batch, seq, d_k]
            actual_slice = result[:, h, :, :]  # [batch, seq, d_k]
            if not torch.allclose(actual_slice, expected_slice, atol=1e-6):
                return False, f"Values for head {h} don't match expected"
        
        return True, f"Shape and values correct"
    except Exception as e:
        return False, str(e)


def test_merge_heads_shape() -> Tuple[bool, str]:
    """Test that merge_heads produces correct output shape and values."""
    try:
        torch.manual_seed(42)
        batch, num_heads, seq, d_k = 2, 4, 8, 16
        d_model = num_heads * d_k
        x = torch.randn(batch, num_heads, seq, d_k)
        
        result = merge_heads(x)
        
        if result is None:
            return False, "merge_heads returned None"
        
        expected_shape = (batch, seq, d_model)
        if result.shape != expected_shape:
            return False, f"Shape {result.shape}, expected {expected_shape}"
        
        # Validate actual values: result[b, s, h*d_k:(h+1)*d_k] should equal x[b, h, s, :]
        for h in range(num_heads):
            expected_slice = x[:, h, :, :]  # [batch, seq, d_k]
            actual_slice = result[:, :, h*d_k:(h+1)*d_k]  # [batch, seq, d_k]
            if not torch.allclose(actual_slice, expected_slice, atol=1e-6):
                return False, f"Values for head {h} don't match expected"
        
        return True, f"Shape and values correct"
    except Exception as e:
        return False, str(e)


def test_split_merge_inverse() -> Tuple[bool, str]:
    """Test that merge_heads(split_heads(x)) == x."""
    try:
        batch, seq, d_model = 2, 8, 64
        num_heads = 4
        x = torch.randn(batch, seq, d_model)
        
        split = split_heads(x, num_heads)
        if split is None:
            return False, "split_heads returned None"
            
        merged = merge_heads(split)
        if merged is None:
            return False, "merge_heads returned None"
        
        if not torch.allclose(x, merged, atol=1e-6):
            return False, "merge(split(x)) != x"
        
        return True, "split/merge are inverse operations"
    except Exception as e:
        return False, str(e)


def test_mha_scores_shape() -> Tuple[bool, str]:
    """Test multi-head attention scores computation."""
    try:
        import math
        torch.manual_seed(42)
        batch, num_heads, seq, d_k = 2, 4, 8, 16
        Q = torch.randn(batch, num_heads, seq, d_k)
        K = torch.randn(batch, num_heads, seq, d_k)
        V = torch.randn(batch, num_heads, seq, d_k)
        
        output, weights = multi_head_attention_scores(Q, K, V)
        
        if output is None:
            return False, "output is None"
        if weights is None:
            return False, "weights is None"
        
        if output.shape != (batch, num_heads, seq, d_k):
            return False, f"Output shape {output.shape} incorrect"
        if weights.shape != (batch, num_heads, seq, seq):
            return False, f"Weights shape {weights.shape} incorrect"
        
        # Validate computation against reference
        expected_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        expected_weights = torch.softmax(expected_scores, dim=-1)
        expected_output = torch.matmul(expected_weights, V)
        
        if not torch.allclose(weights, expected_weights, atol=1e-5):
            return False, f"Weights mismatch: max diff {(weights - expected_weights).abs().max():.6f}"
        
        if not torch.allclose(output, expected_output, atol=1e-5):
            return False, f"Output mismatch: max diff {(output - expected_output).abs().max():.6f}"
        
        return True, f"Shapes and values correct"
    except Exception as e:
        return False, str(e)


def test_mha_weights_sum_to_one() -> Tuple[bool, str]:
    """Test that attention weights sum to 1."""
    try:
        batch, num_heads, seq, d_k = 2, 4, 8, 16
        Q = torch.randn(batch, num_heads, seq, d_k)
        K = torch.randn(batch, num_heads, seq, d_k)
        V = torch.randn(batch, num_heads, seq, d_k)
        
        _, weights = multi_head_attention_scores(Q, K, V)
        
        if weights is None:
            return False, "weights is None"
        
        row_sums = weights.sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            return False, "Weights don't sum to 1"
        
        return True, "Weights sum to 1 for each query"
    except Exception as e:
        return False, str(e)


def test_multihead_attention_module() -> Tuple[bool, str]:
    """Test the MultiHeadAttention module."""
    try:
        import math
        torch.manual_seed(42)
        d_model, num_heads = 64, 4
        d_k = d_model // num_heads
        batch, seq = 2, 8
        
        mha = MultiHeadAttention(d_model, num_heads)
        
        if mha.W_q is None:
            return False, "W_q not initialized"
        if mha.W_k is None:
            return False, "W_k not initialized"
        if mha.W_v is None:
            return False, "W_v not initialized"
        if mha.W_o is None:
            return False, "W_o not initialized"
        
        x = torch.randn(batch, seq, d_model)
        output, weights = mha(x, x, x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != input shape {x.shape}"
        if weights.shape != (batch, num_heads, seq, seq):
            return False, f"Weights shape {weights.shape} incorrect"
        
        # Validate computation against manual reference
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
        
        if not torch.allclose(output, expected, atol=1e-5):
            return False, f"Output mismatch: max diff {(output - expected).abs().max():.6f}"
        
        return True, "MultiHeadAttention module works correctly"
    except Exception as e:
        return False, str(e)


def test_self_attention_module() -> Tuple[bool, str]:
    """Test the MultiHeadSelfAttention module."""
    try:
        torch.manual_seed(42)
        d_model, num_heads = 64, 4
        batch, seq = 2, 8
        
        self_attn = MultiHeadSelfAttention(d_model, num_heads)
        
        if self_attn.attention is None:
            return False, "attention module not initialized"
        
        x = torch.randn(batch, seq, d_model)
        output, weights = self_attn(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Output shape {output.shape} != input shape {x.shape}"
        
        # Validate that self-attention equals attention(x, x, x)
        expected_output, expected_weights = self_attn.attention(x, x, x)
        
        if not torch.allclose(output, expected_output, atol=1e-5):
            return False, f"Output mismatch with attention(x,x,x): max diff {(output - expected_output).abs().max():.6f}"
        
        if weights is not None and expected_weights is not None:
            if not torch.allclose(weights, expected_weights, atol=1e-5):
                return False, f"Weights mismatch with attention(x,x,x)"
        
        return True, "MultiHeadSelfAttention works correctly"
    except Exception as e:
        return False, str(e)


def test_cross_attention() -> Tuple[bool, str]:
    """Test cross-attention with different sequence lengths."""
    try:
        torch.manual_seed(42)
        d_model, num_heads = 64, 4
        batch = 2
        seq_q, seq_k = 6, 10  # Different lengths!
        
        cross_attn = CrossAttention(d_model, num_heads)
        
        query = torch.randn(batch, seq_q, d_model)
        kv = torch.randn(batch, seq_k, d_model)
        
        output, weights = cross_attn(query, kv)
        
        if output is None:
            return False, "output is None"
        if output.shape != (batch, seq_q, d_model):
            return False, f"Output shape {output.shape} incorrect"
        if weights is not None and weights.shape != (batch, num_heads, seq_q, seq_k):
            return False, f"Weights shape {weights.shape} incorrect"
        
        # Validate that cross-attention equals attention(query, kv, kv)
        expected_output, expected_weights = cross_attn.attention(query, kv, kv)
        
        if not torch.allclose(output, expected_output, atol=1e-5):
            return False, f"Output mismatch with attention(q, kv, kv): max diff {(output - expected_output).abs().max():.6f}"
        
        return True, f"Cross-attention works (q={seq_q}, kv={seq_k})"
    except Exception as e:
        return False, str(e)


def test_mha_vs_reference() -> Tuple[bool, str]:
    """Test against a manual reference implementation."""
    try:
        import math
        d_model, num_heads = 64, 4
        d_k = d_model // num_heads
        batch, seq = 2, 8
        
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch, seq, d_model)
        
        our_output, _ = mha(x, x, x)
        
        if our_output is None:
            return False, "output is None"
        
        # Manual reference
        with torch.no_grad():
            Q = mha.W_q(x)
            K = mha.W_k(x)
            V = mha.W_v(x)
            
            # Split heads manually
            Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)
            K = K.view(batch, seq, num_heads, d_k).transpose(1, 2)
            V = V.view(batch, seq, num_heads, d_k).transpose(1, 2)
            
            # Attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
            weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(weights, V)
            
            # Merge heads
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, d_model)
            ref_output = mha.W_o(attn_out)
        
        if not torch.allclose(our_output, ref_output, atol=1e-5):
            return False, "Doesn't match reference implementation"
        
        return True, "Matches manual reference implementation"
    except Exception as e:
        return False, str(e)


def test_different_head_counts() -> Tuple[bool, str]:
    """Test with various head configurations."""
    try:
        import math
        torch.manual_seed(42)
        d_model = 64
        batch, seq = 2, 8
        
        for num_heads in [1, 2, 4, 8]:
            mha = MultiHeadAttention(d_model, num_heads)
            x = torch.randn(batch, seq, d_model)
            
            output, weights = mha(x, x, x)
            
            if output is None or output.shape != x.shape:
                return False, f"Failed for num_heads={num_heads}"
            if weights.shape[1] != num_heads:
                return False, f"Wrong head count in weights for num_heads={num_heads}"
            
            # Validate attention weights sum to 1
            row_sums = weights.sum(dim=-1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
                return False, f"Weights don't sum to 1 for num_heads={num_heads}"
            
            # Validate output has non-trivial values (not all zeros or same as input)
            if torch.allclose(output, torch.zeros_like(output)):
                return False, f"Output is all zeros for num_heads={num_heads}"
        
        return True, "Works with 1, 2, 4, 8 heads"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("split_heads_shape", test_split_heads_shape),
        ("merge_heads_shape", test_merge_heads_shape),
        ("split_merge_inverse", test_split_merge_inverse),
        ("mha_scores_shape", test_mha_scores_shape),
        ("mha_weights_sum", test_mha_weights_sum_to_one),
        ("multihead_attention_module", test_multihead_attention_module),
        ("self_attention_module", test_self_attention_module),
        ("cross_attention", test_cross_attention),
        ("mha_vs_reference", test_mha_vs_reference),
        ("different_head_counts", test_different_head_counts),
    ]
    
    print(f"\n{'='*50}\nDay 16: Multi-Head Attention - Tests\n{'='*50}")
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
