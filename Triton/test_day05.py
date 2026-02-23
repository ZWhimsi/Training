"""
Test Suite for Day 5: Block-Level Programming
=============================================
Run: python test_day05.py
"""

import torch
import sys
from typing import Dict, Tuple

try:
    from day05 import add_vectors, two_phase_sum, transpose, find_max, softmax_numerator
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


# ============================================================================
# Reference Implementations
# ============================================================================

def reference_add(a, b):
    return a + b

def reference_sum(x):
    return x.sum().unsqueeze(0)

def reference_transpose(x):
    return x.T.contiguous()

def reference_max(x):
    return x.max().unsqueeze(0)

def reference_softmax_num(x, row_maxes):
    return torch.exp(x - row_maxes.unsqueeze(1))


# ============================================================================
# Tests
# ============================================================================

def test_add_vectors_default() -> Tuple[bool, str]:
    try:
        a = torch.randn(10000, device='cuda')
        b = torch.randn(10000, device='cuda')
        result = add_vectors(a, b)
        expected = reference_add(a, b)
        
        if result is None:
            return False, "Returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Values mismatch"
        return True, "Default block size works"
    except Exception as e:
        return False, f"Exception: {e}"


def test_add_vectors_small_block() -> Tuple[bool, str]:
    try:
        a = torch.randn(10000, device='cuda')
        b = torch.randn(10000, device='cuda')
        result = add_vectors(a, b, block_size=256)
        expected = reference_add(a, b)
        
        if result is None:
            return False, "Returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Values mismatch"
        return True, "Small block (256) works"
    except Exception as e:
        return False, f"Exception: {e}"


def test_two_phase_sum_basic() -> Tuple[bool, str]:
    try:
        x = torch.arange(100, dtype=torch.float32, device='cuda')
        result = two_phase_sum(x)
        expected = reference_sum(x)  # 0+1+...+99 = 4950
        
        if result is None:
            return False, "Returned None"
        if not torch.allclose(result, expected, atol=1e-3):
            return False, f"Expected {expected.item()}, got {result.item()}"
        return True, "Sum 0-99 = 4950 correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_two_phase_sum_large() -> Tuple[bool, str]:
    try:
        x = torch.randn(100000, device='cuda')
        result = two_phase_sum(x)
        expected = reference_sum(x)
        
        if result is None:
            return False, "Returned None"
        if not torch.allclose(result, expected, rtol=1e-3, atol=1e-3):
            return False, f"Large sum mismatch"
        return True, "100K element sum correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_transpose_square() -> Tuple[bool, str]:
    try:
        x = torch.randn(64, 64, device='cuda')
        result = transpose(x)
        expected = reference_transpose(x)
        
        if result is None:
            return False, "Returned None"
        if result.shape != expected.shape:
            return False, f"Shape: expected {expected.shape}, got {result.shape}"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Values mismatch"
        return True, "64x64 transpose correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_transpose_rect() -> Tuple[bool, str]:
    try:
        x = torch.randn(100, 50, device='cuda')
        result = transpose(x)
        expected = reference_transpose(x)
        
        if result is None:
            return False, "Returned None"
        if result.shape != expected.shape:
            return False, f"Shape: expected {expected.shape}, got {result.shape}"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Values mismatch"
        return True, "100x50 → 50x100 transpose correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_find_max_basic() -> Tuple[bool, str]:
    try:
        x = torch.tensor([-5, -2, 3, 1, 7, 2], dtype=torch.float32, device='cuda')
        result = find_max(x)
        expected = reference_max(x)  # 7
        
        if result is None:
            return False, "Returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.item()}, got {result.item()}"
        return True, "Max = 7 correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_find_max_large() -> Tuple[bool, str]:
    try:
        x = torch.randn(50000, device='cuda')
        result = find_max(x)
        expected = reference_max(x)
        
        if result is None:
            return False, "Returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Max mismatch"
        return True, "Large tensor max correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_softmax_num_basic() -> Tuple[bool, str]:
    try:
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
        row_maxes = x.max(dim=1).values
        result = softmax_numerator(x, row_maxes)
        expected = reference_softmax_num(x, row_maxes)
        
        if result is None:
            return False, "Returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Softmax numerator mismatch"
        return True, "exp(x-max) computed correctly"
    except Exception as e:
        return False, f"Exception: {e}"


def test_softmax_num_large() -> Tuple[bool, str]:
    try:
        x = torch.randn(32, 128, device='cuda')
        row_maxes = x.max(dim=1).values
        result = softmax_numerator(x, row_maxes)
        expected = reference_softmax_num(x, row_maxes)
        
        if result is None:
            return False, "Returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Large softmax numerator mismatch"
        return True, "32x128 softmax numerator correct"
    except Exception as e:
        return False, f"Exception: {e}"


# ============================================================================
# Runner
# ============================================================================

def run_all_tests():
    results = {}
    
    exercises = [
        ("Exercise 1: Configurable Add", [
            ("add_default", test_add_vectors_default),
            ("add_small_block", test_add_vectors_small_block),
        ]),
        ("Exercise 2: Two-Phase Sum", [
            ("sum_basic", test_two_phase_sum_basic),
            ("sum_large", test_two_phase_sum_large),
        ]),
        ("Exercise 3: Transpose", [
            ("transpose_square", test_transpose_square),
            ("transpose_rect", test_transpose_rect),
        ]),
        ("Exercise 4: Find Max", [
            ("max_basic", test_find_max_basic),
            ("max_large", test_find_max_large),
        ]),
        ("Exercise 5: Softmax Numerator", [
            ("softmax_basic", test_softmax_num_basic),
            ("softmax_large", test_softmax_num_large),
        ]),
    ]
    
    for name, tests in exercises:
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        for test_name, test_fn in tests:
            passed, msg = test_fn()
            results[test_name] = (passed, msg)
            print(f"  [{'PASS' if passed else 'FAIL'}] {test_name}: {msg}")
    
    return results


def print_summary(results):
    passed = sum(1 for p, _ in results.values() if p)
    total = len(results)
    print(f"\n{'='*60}\nSUMMARY: {passed}/{total} passed\n{'='*60}")


if __name__ == "__main__":
    print("Day 5: Block-Level Programming - Tests")
    
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)
    
    results = run_all_tests()
    print_summary(results)
