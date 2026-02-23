"""Test Suite for Day 11: Element-wise Binary Operations"""

import torch
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day11 import (vector_add, vector_sub, vector_mul, vector_div,
                           scalar_add, vector_maximum)
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_vector_add() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        
        result = vector_add(x, y)
        expected = x + y
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, f"x+y OK"
    except Exception as e:
        return False, str(e)


def test_vector_sub() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        
        result = vector_sub(x, y)
        expected = x - y
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, f"x-y OK"
    except Exception as e:
        return False, str(e)


def test_vector_mul() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        
        result = vector_mul(x, y)
        expected = x * y
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, f"x*y OK"
    except Exception as e:
        return False, str(e)


def test_vector_div() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda') + 0.1  # Avoid div by zero
        
        result = vector_div(x, y)
        expected = x / y
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, f"x/y OK"
    except Exception as e:
        return False, str(e)


def test_scalar_add() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        scalar = 3.14
        
        result = scalar_add(x, scalar)
        expected = x + scalar
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, f"x+3.14 OK"
    except Exception as e:
        return False, str(e)


def test_maximum() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        
        result = vector_maximum(x, y)
        expected = torch.maximum(x, y)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, f"max(x,y) OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("vector_add", test_vector_add),
        ("vector_sub", test_vector_sub),
        ("vector_mul", test_vector_mul),
        ("vector_div", test_vector_div),
        ("scalar_add", test_scalar_add),
        ("maximum", test_maximum),
    ]
    
    print(f"\n{'='*50}\nDay 11: Binary Operations - Tests\n{'='*50}")
    
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
