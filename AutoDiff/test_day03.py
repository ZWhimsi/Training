"""Test Suite for Day 3: Partial Derivatives and Gradients"""

import numpy as np
import sys
from typing import Tuple

try:
    from day03 import (partial_x_numerical, partial_y_numerical,
                       gradient_numerical, gradient_x2_plus_y2,
                       gradient_xy, gradient_sin_cos, verify_gradient,
                       gradient_descent_step)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_partial_x() -> Tuple[bool, str]:
    try:
        f = lambda x, y: x**2 + y**2
        result = partial_x_numerical(f, 3.0, 4.0)
        expected = 6.0  # 2x at x=3
        
        if result is None:
            return False, "Returned None"
        if abs(result - expected) > 1e-4:
            return False, f"Expected {expected}, got {result}"
        return True, "∂/∂x(x²+y²) = 2x ✓"
    except Exception as e:
        return False, str(e)


def test_partial_y() -> Tuple[bool, str]:
    try:
        f = lambda x, y: x**2 + y**2
        result = partial_y_numerical(f, 3.0, 4.0)
        expected = 8.0  # 2y at y=4
        
        if result is None:
            return False, "Returned None"
        if abs(result - expected) > 1e-4:
            return False, f"Expected {expected}, got {result}"
        return True, "∂/∂y(x²+y²) = 2y ✓"
    except Exception as e:
        return False, str(e)


def test_gradient_numerical() -> Tuple[bool, str]:
    try:
        f = lambda x, y: x**2 + y**2
        result = gradient_numerical(f, 3.0, 4.0)
        expected = np.array([6.0, 8.0])
        
        if result is None:
            return False, "Returned None"
        if not np.allclose(result, expected, atol=1e-4):
            return False, f"Expected {expected}, got {result}"
        return True, "∇f = [6, 8] ✓"
    except Exception as e:
        return False, str(e)


def test_gradient_analytical() -> Tuple[bool, str]:
    try:
        result = gradient_x2_plus_y2(3.0, 4.0)
        expected = np.array([6.0, 8.0])
        
        if result is None:
            return False, "Returned None"
        if not np.allclose(result, expected):
            return False, f"Expected {expected}, got {result}"
        return True, "Analytical gradient OK"
    except Exception as e:
        return False, str(e)


def test_gradient_xy() -> Tuple[bool, str]:
    try:
        result = gradient_xy(3.0, 4.0)
        expected = np.array([4.0, 3.0])  # [y, x]
        
        if result is None:
            return False, "Returned None"
        if not np.allclose(result, expected):
            return False, f"Expected {expected}, got {result}"
        return True, "∇(xy) = [y, x] ✓"
    except Exception as e:
        return False, str(e)


def test_gradient_sin_cos() -> Tuple[bool, str]:
    try:
        x, y = np.pi/4, np.pi/4
        result = gradient_sin_cos(x, y)
        expected = np.array([
            np.cos(x) * np.cos(y),
            -np.sin(x) * np.sin(y)
        ])
        
        if result is None:
            return False, "Returned None"
        if not np.allclose(result, expected, atol=1e-6):
            return False, f"Gradient mismatch"
        return True, "∇(sin·cos) OK"
    except Exception as e:
        return False, str(e)


def test_verify_gradient() -> Tuple[bool, str]:
    try:
        result = verify_gradient()
        
        if result['match'] is None:
            return False, "Match is None"
        if not result['match']:
            return False, "Numerical and analytical don't match"
        return True, "Numerical ≈ Analytical"
    except Exception as e:
        return False, str(e)


def test_gradient_descent() -> Tuple[bool, str]:
    try:
        f = lambda x, y: x**2 + y**2
        x, y = 3.0, 4.0
        x_new, y_new = gradient_descent_step(f, x, y, learning_rate=0.1)
        
        if x_new is None or y_new is None:
            return False, "Returned None"
        
        # x_new = 3 - 0.1 * 6 = 2.4
        # y_new = 4 - 0.1 * 8 = 3.2
        if abs(x_new - 2.4) > 1e-4 or abs(y_new - 3.2) > 1e-4:
            return False, f"Expected (2.4, 3.2), got ({x_new}, {y_new})"
        
        # Verify we moved closer to minimum (0, 0)
        old_dist = x**2 + y**2
        new_dist = x_new**2 + y_new**2
        if new_dist >= old_dist:
            return False, "Didn't move toward minimum"
        
        return True, f"(3,4)→({x_new},{y_new})"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("partial_x", test_partial_x),
        ("partial_y", test_partial_y),
        ("gradient_numerical", test_gradient_numerical),
        ("gradient_analytical", test_gradient_analytical),
        ("gradient_xy", test_gradient_xy),
        ("gradient_sin_cos", test_gradient_sin_cos),
        ("verify_gradient", test_verify_gradient),
        ("gradient_descent", test_gradient_descent),
    ]
    
    print(f"\n{'='*50}\nDay 3: Gradients - Tests\n{'='*50}")
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
