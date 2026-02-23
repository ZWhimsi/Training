"""Test Suite for Day 16: Tiled Matrix Multiplication"""

import torch
import sys
from typing import Tuple

try:
    from day16 import matmul, matmul_bias, batched_matmul
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_matmul_square() -> Tuple[bool, str]:
    try:
        A = torch.randn(64, 64, device='cuda')
        B = torch.randn(64, 64, device='cuda')
        result = matmul(A, B)
        expected = torch.mm(A, B)
        
        if not torch.allclose(result, expected, atol=1e-4, rtol=1e-4):
            max_diff = (result - expected).abs().max().item()
            return False, f"Max diff: {max_diff}"
        return True, "64x64 @ 64x64 OK"
    except Exception as e:
        return False, str(e)


def test_matmul_rect() -> Tuple[bool, str]:
    try:
        A = torch.randn(128, 256, device='cuda')
        B = torch.randn(256, 64, device='cuda')
        result = matmul(A, B)
        expected = torch.mm(A, B)
        
        if not torch.allclose(result, expected, atol=1e-4, rtol=1e-4):
            return False, "Mismatch"
        return True, "128x256 @ 256x64 OK"
    except Exception as e:
        return False, str(e)


def test_matmul_non_aligned() -> Tuple[bool, str]:
    try:
        A = torch.randn(100, 75, device='cuda')
        B = torch.randn(75, 50, device='cuda')
        result = matmul(A, B)
        expected = torch.mm(A, B)
        
        if not torch.allclose(result, expected, atol=1e-4, rtol=1e-4):
            return False, "Mismatch"
        return True, "Non-aligned 100x75 @ 75x50 OK"
    except Exception as e:
        return False, str(e)


def test_matmul_bias() -> Tuple[bool, str]:
    try:
        A = torch.randn(64, 128, device='cuda')
        B = torch.randn(128, 64, device='cuda')
        bias = torch.randn(64, device='cuda')
        result = matmul_bias(A, B, bias)
        expected = torch.mm(A, B) + bias
        
        if not torch.allclose(result, expected, atol=1e-4, rtol=1e-4):
            return False, "Mismatch"
        return True, "Matmul + bias OK"
    except Exception as e:
        return False, str(e)


def test_batched_matmul() -> Tuple[bool, str]:
    try:
        A = torch.randn(8, 32, 64, device='cuda')
        B = torch.randn(8, 64, 32, device='cuda')
        result = batched_matmul(A, B)
        expected = torch.bmm(A, B)
        
        if not torch.allclose(result, expected, atol=1e-4, rtol=1e-4):
            return False, "Mismatch"
        return True, "Batch 8: 32x64 @ 64x32 OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("matmul_square", test_matmul_square),
        ("matmul_rect", test_matmul_rect),
        ("matmul_non_aligned", test_matmul_non_aligned),
        ("matmul_bias", test_matmul_bias),
        ("batched_matmul", test_batched_matmul),
    ]
    
    print(f"\n{'='*50}\nDay 16: Matmul - Tests\n{'='*50}")
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
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)
    run_all_tests()
