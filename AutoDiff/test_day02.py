"""Test Suite for Day 2: The Chain Rule"""

import numpy as np
import pytest

from day02 import (chain_rule_two_functions, chain_rule_three_functions,
                   numerical_chain_rule, verify_chain_rule,
                   derivative_exp_sin, derivative_polynomial_power,
                   neural_network_gradient_intuition)


def test_chain_rule_two():
    # d/dx[sin(x²)] at x=2: cos(4) * 4 ≈ -2.614
    df_dg = np.cos(4)  # cos(x²) at x=2
    dg_dx = 4  # 2x at x=2
    result = chain_rule_two_functions(df_dg, dg_dx)
    expected = np.cos(4) * 4
    
    assert result is not None, "Returned None"
    assert abs(result - expected) <= 1e-6, f"Expected {expected}, got {result}"


def test_chain_rule_three():
    result = chain_rule_three_functions(2, 3, 4)
    expected = 24  # 2 * 3 * 4
    
    assert result is not None, "Returned None"
    assert result == expected, f"Expected {expected}, got {result}"


def test_numerical_chain_rule():
    # d/dx[sin(x²)] at x=1
    f = np.sin
    g = lambda x: x ** 2
    result = numerical_chain_rule(f, g, 1.0)
    expected = np.cos(1) * 2  # ≈ 1.0806
    
    assert result is not None, "Returned None"
    assert abs(result - expected) <= 1e-4, f"Expected ~{expected:.4f}, got {result:.4f}"


def test_verify_chain_rule():
    result = verify_chain_rule()
    
    assert result['analytical'] is not None, "Analytical is None"
    assert result['numerical'] is not None, "Numerical is None"
    assert abs(result['analytical'] - result['numerical']) <= 1e-4, "Analytical and numerical don't match"


def test_exp_sin():
    x = 0.5
    result = derivative_exp_sin(x)
    expected = np.exp(np.sin(x)) * np.cos(x)
    
    assert result is not None, "Returned None"
    assert abs(result - expected) <= 1e-6, f"Expected {expected}, got {result}"


def test_polynomial_power():
    x = 2.0
    result = derivative_polynomial_power(x)
    expected = 6 * x * (x**2 + 1)**2  # 6 * 2 * 25 = 300
    
    assert result is not None, "Returned None"
    assert abs(result - expected) <= 1e-6, f"Expected {expected}, got {result}"


def test_neural_network_gradient():
    result = neural_network_gradient_intuition()
    
    assert result['dloss_dw'] is not None, "dloss_dw is None"
    # Expected: 2 * (6-5) * 1 * 3 = 6
    expected = 6.0
    assert abs(result['dloss_dw'] - expected) <= 1e-6, f"Expected {expected}, got {result['dloss_dw']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
