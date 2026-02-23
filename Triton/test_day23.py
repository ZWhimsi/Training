"""Test Suite for Day 23: Attention Score Computation"""

import torch
import math
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day23 import attention_scores, apply_causal_mask, mha_attention_scores
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_attention_scores() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len, head_dim = 64, 32
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        
        result = attention_scores(Q, K)
        scale = 1.0 / math.sqrt(head_dim)
        expected = (Q @ K.T) * scale
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "scores OK"
    except Exception as e:
        return False, str(e)


def test_causal_mask() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len = 64
        scores = torch.randn(seq_len, seq_len, device='cuda')
        
        result = apply_causal_mask(scores.clone())
        
        # Check upper triangle is -inf
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                if not torch.isinf(result[i, j]) or result[i, j] > 0:
                    return False, f"scores[{i},{j}] should be -inf"
        
        # Check lower triangle preserved
        for i in range(seq_len):
            for j in range(i+1):
                if torch.isinf(result[i, j]):
                    return False, f"scores[{i},{j}] should not be -inf"
        
        return True, "causal mask OK"
    except Exception as e:
        return False, str(e)


def test_mha_scores() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        n_heads, seq_len, head_dim = 8, 64, 32
        Q = torch.randn(n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(n_heads, seq_len, head_dim, device='cuda')
        
        result = mha_attention_scores(Q, K)
        scale = 1.0 / math.sqrt(head_dim)
        expected = torch.bmm(Q, K.transpose(-2, -1)) * scale
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "multi-head OK"
    except Exception as e:
        return False, str(e)


def test_scaling() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for head_dim in [32, 64, 128]:
            seq_len = 32
            Q = torch.randn(seq_len, head_dim, device='cuda')
            K = torch.randn(seq_len, head_dim, device='cuda')
            
            result = attention_scores(Q, K)
            scale = 1.0 / math.sqrt(head_dim)
            expected = (Q @ K.T) * scale
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-3:
                return False, f"Failed at d={head_dim}"
        
        return True, "scaling OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("attention_scores", test_attention_scores),
        ("causal_mask", test_causal_mask),
        ("mha_scores", test_mha_scores),
        ("scaling", test_scaling),
    ]
    
    print(f"\n{'='*50}\nDay 23: Attention Scores - Tests\n{'='*50}")
    
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        return
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    run_all_tests()
