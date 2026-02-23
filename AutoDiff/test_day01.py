"""Test Suite for Day 1: Introduction to Derivatives"""

import numpy as np
import sys
from typing import Tuple

try:
    from day01 import (forward_difference, central_difference, 
                       derivative_x_squared, derivative_x_cubed,
                       derivative_sin, derivative_exp,
                       compare_derivatives, second_derivative)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_forward_difference() -> Tuple[bool, str]:
    try:
        f = lambda x: x ** 2
        result = forward_difference(f, 3.0)
        expected = 6.0  # d/dx(x²) = 2x = 6 at x=3
        
        if result is None:
            return False, "Returned None"
        if abs(result - expected) > 0.01:
            return False, f"Expected ~6.0, got {result}"
        return True, f"d/dx(x²)|₃ ≈ {result:.4f}"
    except Exception as e:
        return False, str(e)


def test_central_difference() -> Tuple[bool, str]:
    try:
        f = lambda x: x ** 2
        result = central_difference(f, 3.0)
        expected = 6.0
        
        if result is None:
            return False, "Returned None"
        if abs(result - expected) > 1e-6:
            return False, f"Expected 6.0, got {result}"
        return True, f"Central diff more accurate: {result:.8f}"
    except Exception as e:
        return False, str(e)


def test_analytical_x_squared() -> Tuple[bool, str]:
    try:
        result = derivative_x_squared(3.0)
        if result is None:
            return False, "Returned None"
        if result != 6.0:
            return False, f"Expected 6.0, got {result}"
        return True, "d/dx(x²) = 2x ✓"
    except Exception as e:
        return False, str(e)


def test_analytical_x_cubed() -> Tuple[bool, str]:
    try:
        result = derivative_x_cubed(2.0)
        expected = 12.0  # 3 * 2² = 12
        if result is None:
            return False, "Returned None"
        if result != expected:
            return False, f"Expected {expected}, got {result}"
        return True, "d/dx(x³) = 3x² ✓"
    except Exception as e:
        return False, str(e)


def test_analytical_sin() -> Tuple[bool, str]:
    try:
        result = derivative_sin(0.0)
        expected = 1.0  # cos(0) = 1
        if result is None:
            return False, "Returned None"
        if abs(result - expected) > 1e-10:
            return False, f"Expected {expected}, got {result}"
        return True, "d/dx(sin) = cos ✓"
    except Exception as e:
        return False, str(e)


def test_analytical_exp() -> Tuple[bool, str]:
    try:
        result = derivative_exp(0.0)
        expected = 1.0  # e⁰ = 1
        if result is None:
            return False, "Returned None"
        if abs(result - expected) > 1e-10:
            return False, f"Expected {expected}, got {result}"
        return True, "d/dx(eˣ) = eˣ ✓"
    except Exception as e:
        return False, str(e)


def test_compare_derivatives() -> Tuple[bool, str]:
    try:
        f = lambda x: np.sin(x)
        df = lambda x: np.cos(x)
        result = compare_derivatives(f, df, np.pi / 4)
        
        if result is None or result.get('error') is None:
            return False, "Returned None or missing error"
        if result['error'] > 1e-8:
            return False, f"Error too large: {result['error']}"
        return True, f"Comparison error: {result['error']:.2e}"
    except Exception as e:
        return False, str(e)


def test_second_derivative() -> Tuple[bool, str]:
    try:
        f = lambda x: x ** 3
        result = second_derivative(f, 2.0)
        expected = 12.0  # d²/dx²(x³) = 6x = 12 at x=2
        
        if result is None:
            return False, "Returned None"
        if abs(result - expected) > 0.1:
            return False, f"Expected ~12.0, got {result}"
        return True, f"d²/dx²(x³)|₂ ≈ {result:.2f}"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("forward_difference", test_forward_difference),
        ("central_difference", test_central_difference),
        ("analytical_x_squared", test_analytical_x_squared),
        ("analytical_x_cubed", test_analytical_x_cubed),
        ("analytical_sin", test_analytical_sin),
        ("analytical_exp", test_analytical_exp),
        ("compare_derivatives", test_compare_derivatives),
        ("second_derivative", test_second_derivative),
    ]
    
    print(f"\n{'='*50}\nDay 1: Derivatives - Tests\n{'='*50}")
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
    run_all_tests()
