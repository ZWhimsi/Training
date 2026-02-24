"""Test Suite for Day 5: Forward vs Reverse Mode AD"""

import pytest

from day05 import (forward_mode_x_squared, forward_mode_polynomial,
                   forward_mode_two_vars, reverse_mode_manual,
                   complexity_analysis)


def test_forward_x_squared():
    val, deriv = forward_mode_x_squared(3.0)
    
    assert val is not None and deriv is not None, "Returned None"
    assert abs(val - 9.0) <= 1e-6, f"Value: expected 9, got {val}"
    assert abs(deriv - 6.0) <= 1e-6, f"Derivative: expected 6, got {deriv}"


def test_forward_polynomial():
    # f(x) = x³ + 2x² + 3x + 4
    # f'(x) = 3x² + 4x + 3
    # At x=2: f(2) = 8 + 8 + 6 + 4 = 26
    #         f'(2) = 12 + 8 + 3 = 23
    val, deriv = forward_mode_polynomial(2.0)
    
    assert val is not None and deriv is not None, "Returned None"
    assert abs(val - 26.0) <= 1e-6, f"Value: expected 26, got {val}"
    assert abs(deriv - 23.0) <= 1e-6, f"Derivative: expected 23, got {deriv}"


def test_forward_two_vars():
    # f(x,y) = xy + x²
    # ∂f/∂x = y + 2x = 3 + 4 = 7 at (2, 3)
    # f(2,3) = 6 + 4 = 10
    val, df_dx = forward_mode_two_vars(2.0, 3.0)
    
    assert val is not None and df_dx is not None, "Returned None"
    assert abs(val - 10.0) <= 1e-6, f"Value: expected 10, got {val}"
    assert abs(df_dx - 7.0) <= 1e-6, f"df/dx: expected 7, got {df_dx}"


def test_reverse_mode():
    # f(x,y) = (x+y)*x = x² + xy
    # ∂f/∂x = 2x + y = 4 + 3 = 7 at (2, 3)
    # ∂f/∂y = x = 2
    # f(2,3) = 4 + 6 = 10
    result = reverse_mode_manual(2.0, 3.0)
    
    assert result['df_dx'] is not None and result['df_dy'] is not None, "Gradients are None"
    assert abs(result['value'] - 10.0) <= 1e-6, f"Value: expected 10, got {result['value']}"
    assert abs(result['df_dx'] - 7.0) <= 1e-6, f"df/dx: expected 7, got {result['df_dx']}"
    assert abs(result['df_dy'] - 2.0) <= 1e-6, f"df/dy: expected 2, got {result['df_dy']}"


def test_complexity():
    analysis = complexity_analysis()
    
    nn = analysis['neural_net_loss']
    assert nn['better_mode'] == 'reverse', "NN should use reverse mode"
    assert nn['reverse_passes'] == 1, "Reverse needs 1 pass for 1 output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
