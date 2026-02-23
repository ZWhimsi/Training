"""Test Suite for Day 18: Strided Memory Access"""

import torch
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day18 import strided_load, strided_2d_copy, gather, scatter_add
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_strided_load() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.arange(100, device='cuda', dtype=torch.float32)
        stride = 3
        size = 30
        
        result = strided_load(x, stride, size)
        expected = x[::stride][:size]
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "strided load OK"
    except Exception as e:
        return False, str(e)


def test_strided_2d_copy() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(64, 128, device='cuda')
        
        result = strided_2d_copy(x)
        expected = x.clone()
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "2D copy OK"
    except Exception as e:
        return False, str(e)


def test_gather() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        src = torch.randn(1000, device='cuda')
        idx = torch.randint(0, 1000, (500,), device='cuda')
        
        result = gather(src, idx)
        expected = src[idx]
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "gather OK"
    except Exception as e:
        return False, str(e)


def test_scatter_add() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        values = torch.ones(100, device='cuda')
        indices = torch.randint(0, 50, (100,), device='cuda')
        
        result = scatter_add(values, indices, 50)
        
        # Manual reference
        expected = torch.zeros(50, device='cuda')
        for i in range(100):
            expected[indices[i]] += values[i]
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "scatter_add OK"
    except Exception as e:
        return False, str(e)


def test_transposed_tensor() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(64, 128, device='cuda')
        x_t = x.T.contiguous()  # Make contiguous after transpose
        
        result = strided_2d_copy(x_t)
        expected = x_t.clone()
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "transposed OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("strided_load", test_strided_load),
        ("strided_2d_copy", test_strided_2d_copy),
        ("gather", test_gather),
        ("scatter_add", test_scatter_add),
        ("transposed_tensor", test_transposed_tensor),
    ]
    
    print(f"\n{'='*50}\nDay 18: Strided Access - Tests\n{'='*50}")
    
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
