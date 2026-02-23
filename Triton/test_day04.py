"""
Test Suite for Day 4: Program IDs and Grid Configuration
========================================================
Run: python test_day04.py
"""

import torch
import sys
from typing import Dict, Tuple

try:
    from day04 import row_sum, col_sum, add_matrices_2d, vector_add_cyclic, batch_scale
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


# ============================================================================
# Reference Implementations (PyTorch)
# ============================================================================

def reference_row_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=1)

def reference_col_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=0)

def reference_add_matrices(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

def reference_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b

def reference_batch_scale(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return x * scales.unsqueeze(1)


# ============================================================================
# Test Functions
# ============================================================================

def test_row_sum_basic() -> Tuple[bool, str]:
    """Test row sum."""
    try:
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
                         dtype=torch.float32, device='cuda')
        result = row_sum(x)
        expected = reference_row_sum(x)  # [6, 15, 24]
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        return True, "Row sums: [6, 15, 24] correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_row_sum_large() -> Tuple[bool, str]:
    """Test row sum with larger matrix."""
    try:
        x = torch.randn(100, 256, device='cuda')
        result = row_sum(x)
        expected = reference_row_sum(x)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-4):
            max_diff = (result - expected).abs().max().item()
            return False, f"Max diff: {max_diff}"
        return True, "100x256 matrix row sums correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_col_sum_basic() -> Tuple[bool, str]:
    """Test column sum."""
    try:
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
                         dtype=torch.float32, device='cuda')
        result = col_sum(x)
        expected = reference_col_sum(x)  # [12, 15, 18]
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        return True, "Col sums: [12, 15, 18] correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_col_sum_large() -> Tuple[bool, str]:
    """Test column sum with larger matrix."""
    try:
        x = torch.randn(256, 100, device='cuda')
        result = col_sum(x)
        expected = reference_col_sum(x)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-4):
            max_diff = (result - expected).abs().max().item()
            return False, f"Max diff: {max_diff}"
        return True, "256x100 matrix col sums correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_add_matrices_2d_basic() -> Tuple[bool, str]:
    """Test 2D matrix addition."""
    try:
        a = torch.randn(64, 64, device='cuda')
        b = torch.randn(64, 64, device='cuda')
        result = add_matrices_2d(a, b)
        expected = reference_add_matrices(a, b)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Matrix addition failed"
        return True, "64x64 matrix addition correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_add_matrices_2d_non_square() -> Tuple[bool, str]:
    """Test 2D matrix addition with non-square matrices."""
    try:
        a = torch.randn(100, 200, device='cuda')
        b = torch.randn(100, 200, device='cuda')
        result = add_matrices_2d(a, b)
        expected = reference_add_matrices(a, b)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Non-square matrix addition failed"
        return True, "100x200 non-square matrix addition correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_vector_add_cyclic() -> Tuple[bool, str]:
    """Test cyclic vector addition."""
    try:
        a = torch.randn(10000, device='cuda')
        b = torch.randn(10000, device='cuda')
        result = vector_add_cyclic(a, b)
        expected = reference_vector_add(a, b)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Vector addition failed"
        return True, "10000-element vector add correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_batch_scale_basic() -> Tuple[bool, str]:
    """Test batch scaling."""
    try:
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
        scales = torch.tensor([2.0, 0.5], device='cuda')
        result = batch_scale(x, scales)
        expected = reference_batch_scale(x, scales)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        return True, "Batch scaling: [*2, *0.5] correct"
    except Exception as e:
        return False, f"Exception: {e}"


def test_batch_scale_large() -> Tuple[bool, str]:
    """Test batch scaling with larger batches."""
    try:
        batch_size = 64
        vec_size = 128
        x = torch.randn(batch_size, vec_size, device='cuda')
        scales = torch.randn(batch_size, device='cuda')
        result = batch_scale(x, scales)
        expected = reference_batch_scale(x, scales)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Large batch scale failed"
        return True, "64x128 batch scaling correct"
    except Exception as e:
        return False, f"Exception: {e}"


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests() -> Dict[str, Tuple[bool, str]]:
    results = {}
    
    exercises = [
        ("Exercise 1: Row Sum", [
            ("row_sum_basic", test_row_sum_basic),
            ("row_sum_large", test_row_sum_large),
        ]),
        ("Exercise 2: Column Sum", [
            ("col_sum_basic", test_col_sum_basic),
            ("col_sum_large", test_col_sum_large),
        ]),
        ("Exercise 3: 2D Matrix Add", [
            ("add_2d_basic", test_add_matrices_2d_basic),
            ("add_2d_nonsquare", test_add_matrices_2d_non_square),
        ]),
        ("Exercise 4: Cyclic Vector Add", [
            ("vec_add_cyclic", test_vector_add_cyclic),
        ]),
        ("Exercise 5: Batch Scale", [
            ("batch_scale_basic", test_batch_scale_basic),
            ("batch_scale_large", test_batch_scale_large),
        ]),
    ]
    
    for exercise_name, tests in exercises:
        print(f"\n{'=' * 60}")
        print(exercise_name)
        print("=" * 60)
        
        for test_name, test_fn in tests:
            passed, msg = test_fn()
            results[test_name] = (passed, msg)
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {test_name}: {msg}")
    
    return results


def print_summary(results: Dict[str, Tuple[bool, str]]) -> None:
    total = len(results)
    passed = sum(1 for p, _ in results.values() if p)
    
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed: {passed}/{total}")
    
    if passed == total:
        print("\n  All tests passed! Day 4 complete!")


if __name__ == "__main__":
    print("Day 4: Program IDs and Grid Configuration - Test Suite")
    print("=" * 60)
    
    if not IMPORT_SUCCESS:
        print(f"\nERROR: Could not import from day04.py: {IMPORT_ERROR}")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    results = run_all_tests()
    print_summary(results)
