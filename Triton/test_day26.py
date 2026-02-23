"""Test Suite for Day 26: Blocked Softmax"""

import torch
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day26 import compute_row_max, safe_blocked_softmax, blocked_attention
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_row_max() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(64, 256, device='cuda')
        result = compute_row_max(x)
        expected = x.max(dim=-1).values
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "row max OK"
    except Exception as e:
        return False, str(e)


def test_safe_softmax() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(64, 256, device='cuda')
        result = safe_blocked_softmax(x)
        expected = torch.softmax(x, dim=-1)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        
        # Verify rows sum to 1
        row_sums = result.sum(dim=-1)
        sum_err = (row_sums - 1.0).abs().max().item()
        if sum_err > 1e-4:
            return False, f"Sum error: {sum_err:.6f}"
        
        return True, "blocked softmax OK"
    except Exception as e:
        return False, str(e)


def test_blocked_attention() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len, head_dim = 64, 32
        scores = torch.randn(seq_len, seq_len, device='cuda')
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        result = blocked_attention(scores, V)
        expected = torch.softmax(scores, dim=-1) @ V
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "blocked attention OK"
    except Exception as e:
        return False, str(e)


def test_numerical_stability() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        # Large values that could cause overflow
        x = torch.randn(32, 128, device='cuda') * 50
        
        result = safe_blocked_softmax(x)
        
        if torch.isnan(result).any():
            return False, "NaN in output"
        if torch.isinf(result).any():
            return False, "Inf in output"
        
        expected = torch.softmax(x, dim=-1)
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        
        return True, "numerically stable"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("row_max", test_row_max),
        ("safe_softmax", test_safe_softmax),
        ("blocked_attention", test_blocked_attention),
        ("numerical_stability", test_numerical_stability),
    ]
    
    print(f"\n{'='*50}\nDay 26: Blocked Softmax - Tests\n{'='*50}")
    
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
