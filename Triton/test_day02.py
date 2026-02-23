"""
Test Suite for Day 2: Memory Operations
=======================================
Run: python test_day02.py
"""

import torch
import sys
from typing import Dict, Tuple

try:
    from day02 import copy, scaled_copy, strided_load, relu, add_relu
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


# ============================================================================
# Reference Implementations (PyTorch)
# ============================================================================

def reference_copy(src: torch.Tensor) -> torch.Tensor:
    return src.clone()

def reference_scaled_copy(src: torch.Tensor, scale: float) -> torch.Tensor:
    return src * scale

def reference_strided_load(src: torch.Tensor, stride: int) -> torch.Tensor:
    return src[::stride].clone()

def reference_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

def reference_add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(a + b)


# ============================================================================
# Test Functions
# ============================================================================

def test_copy_basic() -> Tuple[bool, str]:
    """Test basic copy."""
    try:
        x = torch.randn(1000, device='cuda')
        result = copy(x)
        expected = reference_copy(x)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Copy values don't match"
        return True, "Basic copy works"
    except Exception as e:
        return False, f"Exception: {e}"


def test_copy_large() -> Tuple[bool, str]:
    """Test copy with large tensor."""
    try:
        x = torch.randn(100000, device='cuda')
        result = copy(x)
        expected = reference_copy(x)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Large copy failed"
        return True, "Large tensor copy works (100K elements)"
    except Exception as e:
        return False, f"Exception: {e}"


def test_scaled_copy_basic() -> Tuple[bool, str]:
    """Test scaled copy."""
    try:
        x = torch.randn(1000, device='cuda')
        scale = 2.5
        result = scaled_copy(x, scale)
        expected = reference_scaled_copy(x, scale)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Scaled values don't match"
        return True, "Scaled copy works (scale=2.5)"
    except Exception as e:
        return False, f"Exception: {e}"


def test_scaled_copy_negative() -> Tuple[bool, str]:
    """Test scaled copy with negative scale."""
    try:
        x = torch.randn(1000, device='cuda')
        scale = -0.5
        result = scaled_copy(x, scale)
        expected = reference_scaled_copy(x, scale)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Negative scale failed"
        return True, "Negative scale works"
    except Exception as e:
        return False, f"Exception: {e}"


def test_strided_load_stride2() -> Tuple[bool, str]:
    """Test strided load with stride 2."""
    try:
        x = torch.arange(100, dtype=torch.float32, device='cuda')
        result = strided_load(x, stride=2)
        expected = reference_strided_load(x, stride=2)
        
        if result is None:
            return False, "Function returned None"
        if result.shape != expected.shape:
            return False, f"Shape mismatch: {result.shape} vs {expected.shape}"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Strided values don't match"
        return True, "Stride 2 works: [0,2,4,...] extracted"
    except Exception as e:
        return False, f"Exception: {e}"


def test_strided_load_stride4() -> Tuple[bool, str]:
    """Test strided load with stride 4."""
    try:
        x = torch.randn(1000, device='cuda')
        result = strided_load(x, stride=4)
        expected = reference_strided_load(x, stride=4)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Stride 4 values don't match"
        return True, "Stride 4 works"
    except Exception as e:
        return False, f"Exception: {e}"


def test_relu_basic() -> Tuple[bool, str]:
    """Test ReLU with positive and negative values."""
    try:
        x = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32, device='cuda')
        result = relu(x)
        expected = reference_relu(x)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        return True, "ReLU correctly zeros negatives"
    except Exception as e:
        return False, f"Exception: {e}"


def test_relu_large() -> Tuple[bool, str]:
    """Test ReLU with large tensor."""
    try:
        x = torch.randn(10000, device='cuda')
        result = relu(x)
        expected = reference_relu(x)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Large ReLU failed"
        return True, "Large tensor ReLU works"
    except Exception as e:
        return False, f"Exception: {e}"


def test_add_relu_basic() -> Tuple[bool, str]:
    """Test fused add+ReLU."""
    try:
        a = torch.tensor([-1, 0, 1, 2], dtype=torch.float32, device='cuda')
        b = torch.tensor([0.5, -0.5, -0.5, 0.5], dtype=torch.float32, device='cuda')
        result = add_relu(a, b)
        expected = reference_add_relu(a, b)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        return True, "Fused add+ReLU works"
    except Exception as e:
        return False, f"Exception: {e}"


def test_add_relu_large() -> Tuple[bool, str]:
    """Test fused add+ReLU with large tensors."""
    try:
        a = torch.randn(10000, device='cuda')
        b = torch.randn(10000, device='cuda')
        result = add_relu(a, b)
        expected = reference_add_relu(a, b)
        
        if result is None:
            return False, "Function returned None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Large fused operation failed"
        return True, "Large fused add+ReLU works"
    except Exception as e:
        return False, f"Exception: {e}"


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests() -> Dict[str, Tuple[bool, str]]:
    results = {}
    
    exercises = [
        ("Exercise 1: Copy", [
            ("copy_basic", test_copy_basic),
            ("copy_large", test_copy_large),
        ]),
        ("Exercise 2: Scaled Copy", [
            ("scaled_basic", test_scaled_copy_basic),
            ("scaled_negative", test_scaled_copy_negative),
        ]),
        ("Exercise 3: Strided Load", [
            ("stride2", test_strided_load_stride2),
            ("stride4", test_strided_load_stride4),
        ]),
        ("Exercise 4: ReLU", [
            ("relu_basic", test_relu_basic),
            ("relu_large", test_relu_large),
        ]),
        ("Exercise 5: Fused Add+ReLU", [
            ("add_relu_basic", test_add_relu_basic),
            ("add_relu_large", test_add_relu_large),
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
        print("\n  All tests passed! Day 2 complete!")
    else:
        print(f"\n  {total - passed} test(s) remaining.")


if __name__ == "__main__":
    print("Day 2: Memory Operations - Test Suite")
    print("=" * 60)
    
    if not IMPORT_SUCCESS:
        print(f"\nERROR: Could not import from day02.py: {IMPORT_ERROR}")
        sys.exit(1)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    results = run_all_tests()
    print_summary(results)
