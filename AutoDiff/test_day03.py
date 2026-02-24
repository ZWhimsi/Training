"""Test Suite for Day 3: Partial Derivatives and Gradients"""

import numpy as np
import pytest

from day03 import (partial_x_numerical, partial_y_numerical,
                   gradient_numerical, gradient_x2_plus_y2,
                   gradient_xy, gradient_sin_cos, verify_gradient,
                   gradient_descent_step)


def test_partial_x():
    f = lambda x, y: x**2 + y**2
    result = partial_x_numerical(f, 3.0, 4.0)
    expected = 6.0  # 2x at x=3
    
    assert result is not None, "Returned None"
    assert abs(result - expected) <= 1e-4, f"Expected {expected}, got {result}"


def test_partial_y():
    f = lambda x, y: x**2 + y**2
    result = partial_y_numerical(f, 3.0, 4.0)
    expected = 8.0  # 2y at y=4
    
    assert result is not None, "Returned None"
    assert abs(result - expected) <= 1e-4, f"Expected {expected}, got {result}"


def test_gradient_numerical():
    f = lambda x, y: x**2 + y**2
    result = gradient_numerical(f, 3.0, 4.0)
    expected = np.array([6.0, 8.0])
    
    assert result is not None, "Returned None"
    assert np.allclose(result, expected, atol=1e-4), f"Expected {expected}, got {result}"


def test_gradient_analytical():
    result = gradient_x2_plus_y2(3.0, 4.0)
    expected = np.array([6.0, 8.0])
    
    assert result is not None, "Returned None"
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_gradient_xy():
    result = gradient_xy(3.0, 4.0)
    expected = np.array([4.0, 3.0])  # [y, x]
    
    assert result is not None, "Returned None"
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def test_gradient_sin_cos():
    x, y = np.pi/4, np.pi/4
    result = gradient_sin_cos(x, y)
    expected = np.array([
        np.cos(x) * np.cos(y),
        -np.sin(x) * np.sin(y)
    ])
    
    assert result is not None, "Returned None"
    assert np.allclose(result, expected, atol=1e-6), "Gradient mismatch"


def test_verify_gradient():
    result = verify_gradient()
    
    assert result['match'] is not None, "Match is None"
    assert result['match'], "Numerical and analytical don't match"


def test_gradient_descent():
    f = lambda x, y: x**2 + y**2
    x, y = 3.0, 4.0
    x_new, y_new = gradient_descent_step(f, x, y, learning_rate=0.1)
    
    assert x_new is not None and y_new is not None, "Returned None"
    # x_new = 3 - 0.1 * 6 = 2.4
    # y_new = 4 - 0.1 * 8 = 3.2
    assert abs(x_new - 2.4) <= 1e-4 and abs(y_new - 3.2) <= 1e-4, f"Expected (2.4, 3.2), got ({x_new}, {y_new})"
    
    # Verify we moved closer to minimum (0, 0)
    old_dist = x**2 + y**2
    new_dist = x_new**2 + y_new**2
    assert new_dist < old_dist, "Didn't move toward minimum"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
