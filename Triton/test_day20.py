"""Test Suite for Day 20: Efficient Transpose"""

import torch
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day20 import naive_transpose, tiled_transpose, batched_transpose
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_naive_transpose() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(128, 256, device='cuda')
        result = naive_transpose(x)
        expected = x.T
        
        if result.shape != expected.shape:
            return False, f"Shape: {result.shape} != {expected.shape}"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "naive transpose OK"
    except Exception as e:
        return False, str(e)


def test_tiled_transpose() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(256, 512, device='cuda')
        result = tiled_transpose(x)
        expected = x.T
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "tiled transpose OK"
    except Exception as e:
        return False, str(e)


def test_batched_transpose() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(8, 64, 128, device='cuda')
        result = batched_transpose(x)
        expected = x.transpose(1, 2)
        
        if result.shape != expected.shape:
            return False, f"Shape: {result.shape}"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "batched transpose OK"
    except Exception as e:
        return False, str(e)


def test_non_square() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for shape in [(64, 128), (100, 200), (33, 77)]:
            x = torch.randn(shape, device='cuda')
            result = tiled_transpose(x)
            expected = x.T
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-4:
                return False, f"Failed at {shape}"
        
        return True, "non-square OK"
    except Exception as e:
        return False, str(e)


def test_correctness() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(128, 128, device='cuda')
        
        result = tiled_transpose(x)
        
        # Verify: result[i,j] == x[j,i]
        for _ in range(10):
            i = torch.randint(0, 128, (1,)).item()
            j = torch.randint(0, 128, (1,)).item()
            if abs(result[i, j].item() - x[j, i].item()) > 1e-5:
                return False, f"result[{i},{j}] != x[{j},{i}]"
        
        return True, "correctness verified"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("naive_transpose", test_naive_transpose),
        ("tiled_transpose", test_tiled_transpose),
        ("batched_transpose", test_batched_transpose),
        ("non_square", test_non_square),
        ("correctness", test_correctness),
    ]
    
    print(f"\n{'='*50}\nDay 20: Transpose - Tests\n{'='*50}")
    
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
