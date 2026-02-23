"""
Test Suite for Day 7: Memory Coalescing
=======================================
"""

import torch
import sys
from typing import Tuple

try:
    from day07 import row_access, col_access, tiled_access, vectorized_op, benchmark_access_patterns
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def ref_double(x): return x * 2.0


def test_row_access() -> Tuple[bool, str]:
    try:
        x = torch.randn(64, 128, device='cuda')
        result = row_access(x)
        expected = ref_double(x)
        if result is None: return False, "None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "Row access OK"
    except Exception as e:
        return False, str(e)


def test_col_access() -> Tuple[bool, str]:
    try:
        x = torch.randn(64, 128, device='cuda')
        result = col_access(x)
        expected = ref_double(x)
        if result is None: return False, "None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "Col access OK (but slower!)"
    except Exception as e:
        return False, str(e)


def test_tiled_access() -> Tuple[bool, str]:
    try:
        x = torch.randn(100, 200, device='cuda')
        result = tiled_access(x)
        expected = ref_double(x)
        if result is None: return False, "None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "Tiled access OK"
    except Exception as e:
        return False, str(e)


def test_vectorized() -> Tuple[bool, str]:
    try:
        x = torch.randn(100000, device='cuda')
        result = vectorized_op(x)
        expected = ref_double(x)
        if result is None: return False, "None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "Vectorized OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("row_access", test_row_access),
        ("col_access", test_col_access),
        ("tiled_access", test_tiled_access),
        ("vectorized", test_vectorized),
    ]
    
    print(f"\n{'='*50}\nDay 7 Tests\n{'='*50}")
    results = {}
    for name, fn in tests:
        p, m = fn()
        results[name] = (p, m)
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    
    passed = sum(1 for p, _ in results.values() if p)
    print(f"\nSummary: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\nRunning benchmark...")
        try:
            benchmark_access_patterns()
        except Exception as e:
            print(f"Benchmark error: {e}")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)
    run_all_tests()
