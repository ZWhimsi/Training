"""
Test Suite for Day 3: Masking and Bounds Checking
=================================================
Run: python test_day03.py
"""

import torch
import sys
from typing import Dict, Tuple

try:
    from day03 import safe_load, threshold, clamp, positive_sum, where
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


# ============================================================================
# Reference Implementations (PyTorch)
# ============================================================================

def reference_threshold(x: torch.Tensor, thresh: float) -> torch.Tensor:
    return torch.where(x >= thresh, x, torch.zeros_like(x))

def reference_clamp(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    return torch.clamp(x, min_val, max_val)

def reference_positive_sum(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x[x > 0]).unsqueeze(0)

def reference_where(cond: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(cond, a, b)


# ============================================================================
# Test Functions
# ============================================================================

def test_threshold_basic() -> Tuple[bool, str]:
    """Test threshold operation."""
    try:
        x = torch.tensor([-2, -1, 0, 1, 2, 3], dtype=torch.float32, device='cuda')
        result = threshold(x, 0.5)
        expected = reference_threshold(x, 0.5)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        return True, "Threshold correctly zeros values below 0.5"
    except Exception as e:
        return False, f"Exception: {e}"


def test_threshold_all_pass() -> Tuple[bool, str]:
    """Test when all values pass threshold."""
    try:
        x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device='cuda')
        result = threshold(x, 0.0)
        expected = reference_threshold(x, 0.0)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "All values should pass"
        return True, "All values pass when threshold=0"
    except Exception as e:
        return False, f"Exception: {e}"


def test_threshold_large() -> Tuple[bool, str]:
    """Test threshold with large tensor."""
    try:
        x = torch.randn(10000, device='cuda')
        result = threshold(x, 0.0)
        expected = reference_threshold(x, 0.0)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Large tensor threshold failed"
        return True, "Large tensor threshold works"
    except Exception as e:
        return False, f"Exception: {e}"


def test_clamp_basic() -> Tuple[bool, str]:
    """Test clamp operation."""
    try:
        x = torch.tensor([-3, -1, 0, 1, 3], dtype=torch.float32, device='cuda')
        result = clamp(x, -2.0, 2.0)
        expected = reference_clamp(x, -2.0, 2.0)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        return True, "Clamp correctly limits to [-2, 2]"
    except Exception as e:
        return False, f"Exception: {e}"


def test_clamp_asymmetric() -> Tuple[bool, str]:
    """Test clamp with asymmetric bounds."""
    try:
        x = torch.randn(1000, device='cuda') * 5
        result = clamp(x, -1.0, 3.0)
        expected = reference_clamp(x, -1.0, 3.0)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Asymmetric clamp failed"
        # Verify bounds
        if result.min() < -1.0 or result.max() > 3.0:
            return False, "Values outside bounds!"
        return True, "Asymmetric bounds [-1, 3] enforced"
    except Exception as e:
        return False, f"Exception: {e}"


def test_positive_sum_basic() -> Tuple[bool, str]:
    """Test positive sum."""
    try:
        x = torch.tensor([-2, -1, 1, 2, 3], dtype=torch.float32, device='cuda')
        result = positive_sum(x)
        expected = reference_positive_sum(x)  # 1 + 2 + 3 = 6
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.item()}, got {result.item()}"
        return True, "Positive sum: 1+2+3=6"
    except Exception as e:
        return False, f"Exception: {e}"


def test_positive_sum_all_negative() -> Tuple[bool, str]:
    """Test positive sum with all negative values."""
    try:
        x = torch.tensor([-5, -4, -3, -2, -1], dtype=torch.float32, device='cuda')
        result = positive_sum(x)
        expected = torch.tensor([0.0], device='cuda')
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected 0, got {result.item()}"
        return True, "All negative → sum is 0"
    except Exception as e:
        return False, f"Exception: {e}"


def test_positive_sum_large() -> Tuple[bool, str]:
    """Test positive sum with large tensor."""
    try:
        x = torch.randn(10000, device='cuda')
        result = positive_sum(x)
        expected = reference_positive_sum(x)
        
        if result is None:
            return False, "Function returned None"
        # Allow some tolerance for floating point accumulation
        if not torch.allclose(result, expected, rtol=1e-3, atol=1e-3):
            return False, f"Expected ~{expected.item():.2f}, got {result.item():.2f}"
        return True, "Large tensor positive sum works"
    except Exception as e:
        return False, f"Exception: {e}"


def test_where_basic() -> Tuple[bool, str]:
    """Test where operation."""
    try:
        cond = torch.tensor([True, False, True, False, True], device='cuda')
        a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device='cuda')
        b = torch.tensor([10, 20, 30, 40, 50], dtype=torch.float32, device='cuda')
        
        result = where(cond, a, b)
        expected = reference_where(cond, a, b)  # [1, 20, 3, 40, 5]
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        return True, "Where selects correctly from a and b"
    except Exception as e:
        return False, f"Exception: {e}"


def test_where_large() -> Tuple[bool, str]:
    """Test where with large tensors."""
    try:
        cond = torch.randint(0, 2, (10000,), device='cuda').bool()
        a = torch.randn(10000, device='cuda')
        b = torch.randn(10000, device='cuda')
        
        result = where(cond, a, b)
        expected = reference_where(cond, a, b)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Large where operation failed"
        return True, "Large tensor where works"
    except Exception as e:
        return False, f"Exception: {e}"


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests() -> Dict[str, Tuple[bool, str]]:
    results = {}
    
    exercises = [
        ("Exercise 2: Threshold", [
            ("threshold_basic", test_threshold_basic),
            ("threshold_all_pass", test_threshold_all_pass),
            ("threshold_large", test_threshold_large),
        ]),
        ("Exercise 3: Clamp", [
            ("clamp_basic", test_clamp_basic),
            ("clamp_asymmetric", test_clamp_asymmetric),
        ]),
        ("Exercise 4: Positive Sum", [
            ("pos_sum_basic", test_positive_sum_basic),
            ("pos_sum_negative", test_positive_sum_all_negative),
            ("pos_sum_large", test_positive_sum_large),
        ]),
        ("Exercise 5: Where", [
            ("where_basic", test_where_basic),
            ("where_large", test_where_large),
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
        print("\n  All tests passed! Day 3 complete!")
    else:
        print(f"\n  {total - passed} test(s) remaining.")


if __name__ == "__main__":
    print("Day 3: Masking and Bounds Checking - Test Suite")
    print("=" * 60)
    
    if not IMPORT_SUCCESS:
        print(f"\nERROR: Could not import from day03.py: {IMPORT_ERROR}")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    results = run_all_tests()
    print_summary(results)
