"""Test Suite for Day 5: Forward vs Reverse Mode AD"""

import sys
from typing import Tuple

try:
    from day05 import (forward_mode_x_squared, forward_mode_polynomial,
                       forward_mode_two_vars, reverse_mode_manual,
                       complexity_analysis)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_forward_x_squared() -> Tuple[bool, str]:
    try:
        val, deriv = forward_mode_x_squared(3.0)
        
        if val is None or deriv is None:
            return False, "Returned None"
        if abs(val - 9.0) > 1e-6:
            return False, f"Value: expected 9, got {val}"
        if abs(deriv - 6.0) > 1e-6:
            return False, f"Derivative: expected 6, got {deriv}"
        return True, "f(3)=9, f'(3)=6"
    except Exception as e:
        return False, str(e)


def test_forward_polynomial() -> Tuple[bool, str]:
    try:
        # f(x) = x³ + 2x² + 3x + 4
        # f'(x) = 3x² + 4x + 3
        # At x=2: f(2) = 8 + 8 + 6 + 4 = 26
        #         f'(2) = 12 + 8 + 3 = 23
        val, deriv = forward_mode_polynomial(2.0)
        
        if val is None or deriv is None:
            return False, "Returned None"
        if abs(val - 26.0) > 1e-6:
            return False, f"Value: expected 26, got {val}"
        if abs(deriv - 23.0) > 1e-6:
            return False, f"Derivative: expected 23, got {deriv}"
        return True, "f(2)=26, f'(2)=23"
    except Exception as e:
        return False, str(e)


def test_forward_two_vars() -> Tuple[bool, str]:
    try:
        # f(x,y) = xy + x²
        # ∂f/∂x = y + 2x = 3 + 4 = 7 at (2, 3)
        # f(2,3) = 6 + 4 = 10
        val, df_dx = forward_mode_two_vars(2.0, 3.0)
        
        if val is None or df_dx is None:
            return False, "Returned None"
        if abs(val - 10.0) > 1e-6:
            return False, f"Value: expected 10, got {val}"
        if abs(df_dx - 7.0) > 1e-6:
            return False, f"df/dx: expected 7, got {df_dx}"
        return True, "f=10, ∂f/∂x=7"
    except Exception as e:
        return False, str(e)


def test_reverse_mode() -> Tuple[bool, str]:
    try:
        # f(x,y) = (x+y)*x = x² + xy
        # ∂f/∂x = 2x + y = 4 + 3 = 7 at (2, 3)
        # ∂f/∂y = x = 2
        # f(2,3) = 4 + 6 = 10
        result = reverse_mode_manual(2.0, 3.0)
        
        if result['df_dx'] is None or result['df_dy'] is None:
            return False, "Gradients are None"
        if abs(result['value'] - 10.0) > 1e-6:
            return False, f"Value: expected 10, got {result['value']}"
        if abs(result['df_dx'] - 7.0) > 1e-6:
            return False, f"df/dx: expected 7, got {result['df_dx']}"
        if abs(result['df_dy'] - 2.0) > 1e-6:
            return False, f"df/dy: expected 2, got {result['df_dy']}"
        return True, "Reverse mode correct"
    except Exception as e:
        return False, str(e)


def test_complexity() -> Tuple[bool, str]:
    try:
        analysis = complexity_analysis()
        
        nn = analysis['neural_net_loss']
        if nn['better_mode'] != 'reverse':
            return False, "NN should use reverse mode"
        if nn['reverse_passes'] != 1:
            return False, "Reverse needs 1 pass for 1 output"
        
        return True, "Complexity analysis correct"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("forward_x_squared", test_forward_x_squared),
        ("forward_polynomial", test_forward_polynomial),
        ("forward_two_vars", test_forward_two_vars),
        ("reverse_mode", test_reverse_mode),
        ("complexity", test_complexity),
    ]
    
    print(f"\n{'='*50}\nDay 5: Forward vs Reverse Mode - Tests\n{'='*50}")
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
