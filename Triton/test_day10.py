"""Test Suite for Day 10: Reduction Operations"""

import torch
import sys
from typing import Tuple

try:
    from day10 import vector_sum, row_wise_sum, vector_mean, vector_variance, large_sum
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_vector_sum() -> Tuple[bool, str]:
    try:
        x = torch.arange(100, dtype=torch.float32, device='cuda')
        result = vector_sum(x)
        expected = x.sum()
        if not torch.allclose(result, expected.unsqueeze(0), atol=1e-4):
            return False, f"Expected {expected.item()}, got {result.item()}"
        return True, f"Sum 0-99 = {result.item():.0f}"
    except Exception as e:
        return False, str(e)


def test_row_wise_sum() -> Tuple[bool, str]:
    try:
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
        result = row_wise_sum(x)
        expected = x.sum(dim=1)
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        return True, f"Row sums: {result.tolist()}"
    except Exception as e:
        return False, str(e)


def test_vector_mean() -> Tuple[bool, str]:
    try:
        x = torch.randn(1000, device='cuda')
        result = vector_mean(x)
        expected = x.mean()
        if not torch.allclose(result, expected.unsqueeze(0), atol=1e-5):
            return False, "Mean mismatch"
        return True, f"Mean computed correctly"
    except Exception as e:
        return False, str(e)


def test_vector_variance() -> Tuple[bool, str]:
    try:
        x = torch.randn(500, device='cuda')
        result = vector_variance(x)
        expected = x.var(unbiased=False)
        if not torch.allclose(result, expected.unsqueeze(0), atol=1e-4):
            return False, f"Expected {expected.item():.4f}, got {result.item():.4f}"
        return True, f"Variance correct"
    except Exception as e:
        return False, str(e)


def test_large_sum() -> Tuple[bool, str]:
    try:
        x = torch.randn(100000, device='cuda')
        result = large_sum(x)
        expected = x.sum()
        if not torch.allclose(result, expected.unsqueeze(0), rtol=1e-4):
            return False, "Large sum mismatch"
        return True, "100K element sum OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("vector_sum", test_vector_sum),
        ("row_wise_sum", test_row_wise_sum),
        ("vector_mean", test_vector_mean),
        ("vector_variance", test_vector_variance),
        ("large_sum", test_large_sum),
    ]
    
    print(f"\n{'='*50}\nDay 10: Reductions - Tests\n{'='*50}")
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
