"""Test Suite for Day 27: Memory-Efficient Attention"""

import torch
import math
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day27 import single_query_attention, memory_efficient_attention
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_attention(Q, K, V):
    """Reference attention implementation."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.T) * scale
    weights = torch.softmax(scores, dim=-1)
    return weights @ V


def test_single_query() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len, head_dim = 64, 32
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        query_idx = 5
        result = single_query_attention(Q, K, V, query_idx)
        
        expected_full = reference_attention(Q, K, V)
        expected = expected_full[query_idx]
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "single query OK"
    except Exception as e:
        return False, str(e)


def test_memory_efficient() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len, head_dim = 64, 32
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        result = memory_efficient_attention(Q, K, V)
        expected = reference_attention(Q, K, V)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "memory efficient OK"
    except Exception as e:
        return False, str(e)


def test_numerical_stability() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len, head_dim = 64, 32
        Q = torch.randn(seq_len, head_dim, device='cuda') * 10
        K = torch.randn(seq_len, head_dim, device='cuda') * 10
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        result = memory_efficient_attention(Q, K, V)
        
        if torch.isnan(result).any():
            return False, "NaN in output"
        if torch.isinf(result).any():
            return False, "Inf in output"
        
        expected = reference_attention(Q, K, V)
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-2:
            return False, f"Error: {max_err:.6f}"
        
        return True, "stable OK"
    except Exception as e:
        return False, str(e)


def test_various_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for seq_len, head_dim in [(32, 32), (64, 64), (128, 32)]:
            Q = torch.randn(seq_len, head_dim, device='cuda')
            K = torch.randn(seq_len, head_dim, device='cuda')
            V = torch.randn(seq_len, head_dim, device='cuda')
            
            result = memory_efficient_attention(Q, K, V)
            expected = reference_attention(Q, K, V)
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-2:
                return False, f"Failed at {seq_len}x{head_dim}"
        
        return True, "various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("single_query", test_single_query),
        ("memory_efficient", test_memory_efficient),
        ("numerical_stability", test_numerical_stability),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 27: Memory-Efficient Attention - Tests\n{'='*50}")
    
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
