"""Test Suite for Day 29: IO-Aware Attention"""

import torch
import math
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day29 import io_aware_attention, attention_with_stats
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_attention(Q, K, V):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    return torch.softmax((Q @ K.T) * scale, dim=-1) @ V


def test_io_aware_attention() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len, head_dim = 128, 64
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        result = io_aware_attention(Q, K, V)
        expected = reference_attention(Q, K, V)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "io_aware OK"
    except Exception as e:
        return False, str(e)


def test_with_stats() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len, head_dim = 128, 64
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        output, L, M = attention_with_stats(Q, K, V)
        expected = reference_attention(Q, K, V)
        
        max_err = (output - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Output error: {max_err:.6f}"
        
        # L and M should have valid values
        if torch.isnan(L).any() or torch.isnan(M).any():
            return False, "NaN in stats"
        
        return True, "with_stats OK"
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
        
        result = io_aware_attention(Q, K, V)
        
        if torch.isnan(result).any():
            return False, "NaN in output"
        if torch.isinf(result).any():
            return False, "Inf in output"
        
        return True, "stable OK"
    except Exception as e:
        return False, str(e)


def test_various_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for seq_len, head_dim in [(64, 32), (128, 64), (256, 64)]:
            Q = torch.randn(seq_len, head_dim, device='cuda')
            K = torch.randn(seq_len, head_dim, device='cuda')
            V = torch.randn(seq_len, head_dim, device='cuda')
            
            result = io_aware_attention(Q, K, V)
            expected = reference_attention(Q, K, V)
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-2:
                return False, f"Failed at {seq_len}x{head_dim}"
        
        return True, "various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("io_aware_attention", test_io_aware_attention),
        ("with_stats", test_with_stats),
        ("numerical_stability", test_numerical_stability),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 29: IO-Aware Attention - Tests\n{'='*50}")
    
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
