"""
Test Suite for Day 1: Hello Triton
==================================
Run this file to check your implementations against PyTorch reference.

Usage:
    python test_day01.py

Each exercise is tested independently so you can see exactly what works.
"""

import torch
import sys
from typing import Dict, Tuple

# Import student implementations
try:
    from day01 import add_one, square
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def check_cuda() -> bool:
    """Check if CUDA is available."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Triton requires a GPU.")
        print("Please run on a machine with NVIDIA GPU and CUDA installed.")
        return False
    return True


# ============================================================================
# Reference Implementations (PyTorch)
# ============================================================================

def reference_add_one(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: add 1 to every element."""
    return x + 1


def reference_square(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference: square every element."""
    return x * x


# ============================================================================
# Test Functions
# ============================================================================

def test_exercise_1_add_one_basic() -> Tuple[bool, str]:
    """Test add_one with a small tensor."""
    try:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
        
        result = add_one(x)
        expected = reference_add_one(x)
        
        if result is None:
            return False, "Function returned None (not implemented yet)"
        
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Mismatch: expected {expected.tolist()}, got {result.tolist()}"
        
        return True, "Basic add_one works correctly"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def test_exercise_1_add_one_large() -> Tuple[bool, str]:
    """Test add_one with larger tensor to verify blocking."""
    try:
        x = torch.randn(10000, device='cuda')
        
        result = add_one(x)
        expected = reference_add_one(x)
        
        if result is None:
            return False, "Function returned None"
        
        if not torch.allclose(result, expected, atol=1e-5):
            max_diff = (result - expected).abs().max().item()
            return False, f"Mismatch on large tensor, max diff: {max_diff}"
        
        return True, "Large tensor (10000 elements) handled correctly"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def test_exercise_1_add_one_2d() -> Tuple[bool, str]:
    """Test add_one with 2D tensor (flattened internally)."""
    try:
        x = torch.randn(100, 100, device='cuda')
        
        result = add_one(x)
        expected = reference_add_one(x)
        
        if result is None:
            return False, "Function returned None"
        
        if result.shape != expected.shape:
            return False, f"Shape mismatch: expected {expected.shape}, got {result.shape}"
        
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Values don't match for 2D tensor"
        
        return True, "2D tensor handled correctly"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def test_exercise_3_square_basic() -> Tuple[bool, str]:
    """Test square with a small tensor."""
    try:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
        
        result = square(x)
        expected = reference_square(x)
        
        if result is None:
            return False, "Function returned None (not implemented yet)"
        
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Mismatch: expected {expected.tolist()}, got {result.tolist()}"
        
        return True, "Basic square works correctly"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def test_exercise_3_square_negative() -> Tuple[bool, str]:
    """Test square with negative numbers."""
    try:
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='cuda')
        
        result = square(x)
        expected = reference_square(x)
        
        if result is None:
            return False, "Function returned None"
        
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Mismatch with negatives: expected {expected.tolist()}, got {result.tolist()}"
        
        return True, "Negative numbers squared correctly"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def test_exercise_3_square_large() -> Tuple[bool, str]:
    """Test square with larger tensor."""
    try:
        x = torch.randn(5000, device='cuda')
        
        result = square(x)
        expected = reference_square(x)
        
        if result is None:
            return False, "Function returned None"
        
        if not torch.allclose(result, expected, atol=1e-5):
            max_diff = (result - expected).abs().max().item()
            return False, f"Mismatch on large tensor, max diff: {max_diff}"
        
        return True, "Large tensor squared correctly"
    except Exception as e:
        return False, f"Exception: {str(e)}"


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests() -> Dict[str, Tuple[bool, str]]:
    """Run all tests and return results."""
    results = {}
    
    # Exercise 1: add_one
    print("\n" + "=" * 60)
    print("Exercise 1: add_one kernel")
    print("=" * 60)
    
    results['ex1_basic'] = test_exercise_1_add_one_basic()
    results['ex1_large'] = test_exercise_1_add_one_large()
    results['ex1_2d'] = test_exercise_1_add_one_2d()
    
    for name, (passed, msg) in list(results.items())[:3]:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {msg}")
    
    # Exercise 3: square
    print("\n" + "=" * 60)
    print("Exercise 3: square kernel")
    print("=" * 60)
    
    results['ex3_basic'] = test_exercise_3_square_basic()
    results['ex3_negative'] = test_exercise_3_square_negative()
    results['ex3_large'] = test_exercise_3_square_large()
    
    for name, (passed, msg) in list(results.items())[3:]:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {msg}")
    
    return results


def print_summary(results: Dict[str, Tuple[bool, str]]) -> None:
    """Print a summary of test results."""
    total = len(results)
    passed = sum(1 for p, _ in results.values() if p)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed: {passed}/{total}")
    
    if passed == total:
        print("\n  Congratulations! All tests passed!")
        print("  You've successfully completed Day 1!")
    else:
        print(f"\n  {total - passed} test(s) still need work.")
        print("  Keep going! Check the FAIL messages above for hints.")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 1: Hello Triton - Test Suite")
    print("=" * 60)
    
    if not IMPORT_SUCCESS:
        print(f"\nERROR: Could not import from day01.py")
        print(f"Details: {IMPORT_ERROR}")
        print("\nMake sure you've implemented the functions in day01.py")
        sys.exit(1)
    
    if not check_cuda():
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    results = run_all_tests()
    print_summary(results)
