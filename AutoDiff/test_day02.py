"""Test Suite for Day 2: The Chain Rule"""

import numpy as np
import sys
from typing import Tuple

try:
    from day02 import (chain_rule_two_functions, chain_rule_three_functions,
                       numerical_chain_rule, verify_chain_rule,
                       derivative_exp_sin, derivative_polynomial_power,
                       neural_network_gradient_intuition)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_chain_rule_two() -> Tuple[bool, str]:
    try:
        # d/dx[sin(x²)] at x=2: cos(4) * 4 ≈ -2.614
        df_dg = np.cos(4)  # cos(x²) at x=2
        dg_dx = 4  # 2x at x=2
        result = chain_rule_two_functions(df_dg, dg_dx)
        expected = np.cos(4) * 4
        
        if result is None:
            return False, "Returned None"
        if abs(result - expected) > 1e-6:
            return False, f"Expected {expected}, got {result}"
        return True, "Two function chain rule OK"
    except Exception as e:
        return False, str(e)


def test_chain_rule_three() -> Tuple[bool, str]:
    try:
        result = chain_rule_three_functions(2, 3, 4)
        expected = 24  # 2 * 3 * 4
        
        if result is None:
            return False, "Returned None"
        if result != expected:
            return False, f"Expected {expected}, got {result}"
        return True, "Three function chain rule OK"
    except Exception as e:
        return False, str(e)


def test_numerical_chain_rule() -> Tuple[bool, str]:
    try:
        # d/dx[sin(x²)] at x=1
        f = np.sin
        g = lambda x: x ** 2
        result = numerical_chain_rule(f, g, 1.0)
        expected = np.cos(1) * 2  # ≈ 1.0806
        
        if result is None:
            return False, "Returned None"
        if abs(result - expected) > 1e-4:
            return False, f"Expected ~{expected:.4f}, got {result:.4f}"
        return True, f"Numerical: {result:.4f}"
    except Exception as e:
        return False, str(e)


def test_verify_chain_rule() -> Tuple[bool, str]:
    try:
        result = verify_chain_rule()
        
        if result['analytical'] is None:
            return False, "Analytical is None"
        if result['numerical'] is None:
            return False, "Numerical is None"
        
        if abs(result['analytical'] - result['numerical']) > 1e-4:
            return False, "Analytical and numerical don't match"
        return True, "Analytical matches numerical"
    except Exception as e:
        return False, str(e)


def test_exp_sin() -> Tuple[bool, str]:
    try:
        x = 0.5
        result = derivative_exp_sin(x)
        expected = np.exp(np.sin(x)) * np.cos(x)
        
        if result is None:
            return False, "Returned None"
        if abs(result - expected) > 1e-6:
            return False, f"Expected {expected}, got {result}"
        return True, "d/dx[exp(sin(x))] OK"
    except Exception as e:
        return False, str(e)


def test_polynomial_power() -> Tuple[bool, str]:
    try:
        x = 2.0
        result = derivative_polynomial_power(x)
        expected = 6 * x * (x**2 + 1)**2  # 6 * 2 * 25 = 300
        
        if result is None:
            return False, "Returned None"
        if abs(result - expected) > 1e-6:
            return False, f"Expected {expected}, got {result}"
        return True, "d/dx[(x²+1)³] OK"
    except Exception as e:
        return False, str(e)


def test_neural_network_gradient() -> Tuple[bool, str]:
    try:
        result = neural_network_gradient_intuition()
        
        if result['dloss_dw'] is None:
            return False, "dloss_dw is None"
        
        # Expected: 2 * (6-5) * 1 * 3 = 6
        expected = 6.0
        if abs(result['dloss_dw'] - expected) > 1e-6:
            return False, f"Expected {expected}, got {result['dloss_dw']}"
        return True, "NN gradient: dloss/dw = 6"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("chain_two", test_chain_rule_two),
        ("chain_three", test_chain_rule_three),
        ("numerical_chain", test_numerical_chain_rule),
        ("verify_chain", test_verify_chain_rule),
        ("exp_sin", test_exp_sin),
        ("polynomial_power", test_polynomial_power),
        ("nn_gradient", test_neural_network_gradient),
    ]
    
    print(f"\n{'='*50}\nDay 2: Chain Rule - Tests\n{'='*50}")
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
