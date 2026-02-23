"""Test Suite for Day 24: Online Softmax"""

import torch
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day24 import online_softmax, scaled_online_softmax, causal_online_softmax
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_online_softmax() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(64, 256, device='cuda')
        
        result = online_softmax(x)
        expected = torch.softmax(x, dim=-1)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        
        # Check rows sum to 1
        row_sums = result.sum(dim=-1)
        sum_err = (row_sums - 1.0).abs().max().item()
        if sum_err > 1e-4:
            return False, f"Row sums not 1: {sum_err:.6f}"
        
        return True, "online softmax OK"
    except Exception as e:
        return False, str(e)


def test_scaled_softmax() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(64, 256, device='cuda')
        scale = 0.125
        
        result = scaled_online_softmax(x, scale)
        expected = torch.softmax(x * scale, dim=-1)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "scaled softmax OK"
    except Exception as e:
        return False, str(e)


def test_causal_softmax() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        n = 32
        x = torch.randn(n, n, device='cuda')
        
        result = causal_online_softmax(x)
        
        # Check upper triangle is 0
        for i in range(n):
            for j in range(i + 1, n):
                if result[i, j].abs() > 1e-6:
                    return False, f"result[{i},{j}] should be 0"
        
        # Check each row sums to 1 (over valid positions)
        for i in range(n):
            row_sum = result[i, :i+1].sum()
            if abs(row_sum - 1.0) > 1e-4:
                return False, f"Row {i} sum: {row_sum:.4f}"
        
        return True, "causal softmax OK"
    except Exception as e:
        return False, str(e)


def test_numerical_stability() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        # Test with large values
        x = torch.randn(32, 128, device='cuda') * 100
        
        result = online_softmax(x)
        expected = torch.softmax(x, dim=-1)
        
        # Check no NaN or Inf
        if torch.isnan(result).any() or torch.isinf(result).any():
            return False, "NaN or Inf in output"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        
        return True, "numerically stable"
    except Exception as e:
        return False, str(e)


def test_various_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for rows, cols in [(16, 64), (64, 256), (128, 1024)]:
            x = torch.randn(rows, cols, device='cuda')
            
            result = online_softmax(x)
            expected = torch.softmax(x, dim=-1)
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-3:
                return False, f"Failed at {rows}x{cols}"
        
        return True, "various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("online_softmax", test_online_softmax),
        ("scaled_softmax", test_scaled_softmax),
        ("causal_softmax", test_causal_softmax),
        ("numerical_stability", test_numerical_stability),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 24: Online Softmax - Tests\n{'='*50}")
    
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
