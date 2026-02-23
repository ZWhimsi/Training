"""Test Suite for Day 17: Batched Outer Products"""

import torch
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day17 import outer_product, batched_outer, scaled_outer
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_outer_product() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        a = torch.randn(64, device='cuda')
        b = torch.randn(128, device='cuda')
        
        result = outer_product(a, b)
        expected = torch.outer(a, b)
        
        if result.shape != expected.shape:
            return False, f"Shape mismatch: {result.shape} vs {expected.shape}"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "outer product OK"
    except Exception as e:
        return False, str(e)


def test_batched_outer() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        B, M, N = 8, 64, 128
        a = torch.randn(B, M, device='cuda')
        b = torch.randn(B, N, device='cuda')
        
        result = batched_outer(a, b)
        expected = torch.bmm(a.unsqueeze(-1), b.unsqueeze(1))
        
        if result.shape != (B, M, N):
            return False, f"Shape: {result.shape}"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "batched outer OK"
    except Exception as e:
        return False, str(e)


def test_scaled_outer() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        a = torch.randn(64, device='cuda')
        b = torch.randn(128, device='cuda')
        scale = 0.125  # Like 1/sqrt(d_k)
        
        result = scaled_outer(a, b, scale)
        expected = scale * torch.outer(a, b)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "scaled outer OK"
    except Exception as e:
        return False, str(e)


def test_various_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for M, N in [(32, 32), (64, 128), (100, 200)]:
            a = torch.randn(M, device='cuda')
            b = torch.randn(N, device='cuda')
            
            result = outer_product(a, b)
            expected = torch.outer(a, b)
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-4:
                return False, f"Failed at {M}x{N}"
        
        return True, "Various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("outer_product", test_outer_product),
        ("batched_outer", test_batched_outer),
        ("scaled_outer", test_scaled_outer),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 17: Outer Products - Tests\n{'='*50}")
    
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
