"""Test Suite for Day 1: Introduction to Derivatives"""

import numpy as np
import pytest

from day01 import (forward_difference, central_difference, 
                   derivative_x_squared, derivative_x_cubed,
                   derivative_sin, derivative_exp,
                   compare_derivatives, second_derivative)


def test_forward_difference():
    f = lambda x: x ** 2
    result = forward_difference(f, 3.0)
    expected = 6.0  # d/dx(x²) = 2x = 6 at x=3
    
    assert result is not None, "Returned None"
    assert abs(result - expected) <= 0.01, f"Expected ~6.0, got {result}"


def test_central_difference():
    f = lambda x: x ** 2
    result = central_difference(f, 3.0)
    expected = 6.0
    
    assert result is not None, "Returned None"
    assert abs(result - expected) <= 1e-6, f"Expected 6.0, got {result}"


def test_analytical_x_squared():
    result = derivative_x_squared(3.0)
    assert result is not None, "Returned None"
    assert result == 6.0, f"Expected 6.0, got {result}"


def test_analytical_x_cubed():
    result = derivative_x_cubed(2.0)
    expected = 12.0  # 3 * 2² = 12
    assert result is not None, "Returned None"
    assert result == expected, f"Expected {expected}, got {result}"


def test_analytical_sin():
    result = derivative_sin(0.0)
    expected = 1.0  # cos(0) = 1
    assert result is not None, "Returned None"
    assert abs(result - expected) <= 1e-10, f"Expected {expected}, got {result}"


def test_analytical_exp():
    result = derivative_exp(0.0)
    expected = 1.0  # e⁰ = 1
    assert result is not None, "Returned None"
    assert abs(result - expected) <= 1e-10, f"Expected {expected}, got {result}"


def test_compare_derivatives():
    f = lambda x: np.sin(x)
    df = lambda x: np.cos(x)
    result = compare_derivatives(f, df, np.pi / 4)
    
    assert result is not None and result.get('error') is not None, "Returned None or missing error"
    assert result['error'] <= 1e-8, f"Error too large: {result['error']}"


def test_second_derivative():
    f = lambda x: x ** 3
    result = second_derivative(f, 2.0)
    expected = 12.0  # d²/dx²(x³) = 6x = 12 at x=2
    
    assert result is not None, "Returned None"
    assert abs(result - expected) <= 0.1, f"Expected ~12.0, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
