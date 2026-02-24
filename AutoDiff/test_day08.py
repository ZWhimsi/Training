"""Test Suite for Day 8: Division and Subtraction Backward"""

import pytest

from day08 import (
    Value,
    verify_subtraction_gradients,
    verify_division_gradients,
    chain_rule_division
)


def test_negation_forward():
    """Test negation forward pass."""
    x = Value(5.0)
    y = -x
    
    assert y is not None, "Negation returned None"
    assert abs(y.data - (-5.0)) <= 1e-6, f"-5 = {y.data}, expected -5.0"


def test_negation_backward():
    """Test negation backward pass."""
    x = Value(5.0)
    y = -x
    y.backward()
    
    assert abs(x.grad - (-1.0)) <= 1e-6, f"d(-x)/dx = {x.grad}, expected -1.0"


def test_subtraction_forward():
    """Test subtraction forward pass."""
    a = Value(7.0)
    b = Value(3.0)
    c = a - b
    
    assert c is not None, "Subtraction returned None"
    assert abs(c.data - 4.0) <= 1e-6, f"7 - 3 = {c.data}, expected 4.0"


def test_subtraction_backward():
    """Test subtraction backward pass."""
    a = Value(7.0)
    b = Value(3.0)
    c = a - b
    c.backward()
    
    assert abs(a.grad - 1.0) <= 1e-6, f"d(a-b)/da = {a.grad}, expected 1.0"
    assert abs(b.grad - (-1.0)) <= 1e-6, f"d(a-b)/db = {b.grad}, expected -1.0"


def test_rsub():
    """Test reverse subtraction (number - Value)."""
    x = Value(3.0)
    y = 10 - x  # Should be 7
    
    assert y is not None, "Reverse subtraction returned None"
    assert abs(y.data - 7.0) <= 1e-6, f"10 - 3 = {y.data}, expected 7.0"
    
    y.backward()
    assert abs(x.grad - (-1.0)) <= 1e-6, f"d(10-x)/dx = {x.grad}, expected -1.0"


def test_division_forward():
    """Test division forward pass."""
    a = Value(10.0)
    b = Value(2.0)
    c = a / b
    
    assert c is not None, "Division returned None"
    assert abs(c.data - 5.0) <= 1e-6, f"10 / 2 = {c.data}, expected 5.0"


def test_division_backward():
    """Test division backward pass."""
    a = Value(6.0)
    b = Value(2.0)
    c = a / b  # = 3
    c.backward()
    
    # d(a/b)/da = 1/b = 0.5
    # d(a/b)/db = -a/b² = -6/4 = -1.5
    assert abs(a.grad - 0.5) <= 1e-6, f"d(a/b)/da = {a.grad}, expected 0.5"
    assert abs(b.grad - (-1.5)) <= 1e-6, f"d(a/b)/db = {b.grad}, expected -1.5"


def test_rtruediv():
    """Test reverse division (number / Value)."""
    x = Value(4.0)
    y = 12 / x  # Should be 3
    
    assert y is not None, "Reverse division returned None"
    assert abs(y.data - 3.0) <= 1e-6, f"12 / 4 = {y.data}, expected 3.0"
    
    y.backward()
    # d(12/x)/dx = -12/x² = -12/16 = -0.75
    assert abs(x.grad - (-0.75)) <= 1e-6, f"d(12/x)/dx = {x.grad}, expected -0.75"


def test_chain_rule_subtraction():
    """Test chain rule with subtraction."""
    x = Value(2.0)
    # f(x) = (x^2 - x) at x=2
    # f(2) = 4 - 2 = 2
    # f'(x) = 2x - 1 = 3 at x=2
    y = x ** 2 - x
    y.backward()
    
    assert abs(y.data - 2.0) <= 1e-6, f"x²-x at x=2 = {y.data}, expected 2"
    assert abs(x.grad - 3.0) <= 1e-6, f"d(x²-x)/dx = {x.grad}, expected 3"


def test_chain_rule_division():
    """Test chain rule with division."""
    result = chain_rule_division()
    
    assert result['value'] is not None, "Result is None"
    assert abs(result['value'] - 2.0) <= 1e-6, f"(x+1)/(x-1) = {result['value']}, expected 2"
    assert abs(result['gradient'] - (-0.5)) <= 1e-4, f"gradient = {result['gradient']}, expected -0.5"


def test_complex_expression():
    """Test complex expression with division and subtraction."""
    # f(x) = (2x - 1) / x at x = 2
    # f(2) = 3/2 = 1.5
    # f'(x) = [2x - (2x-1)] / x² = 1/x²
    # f'(2) = 1/4 = 0.25
    x = Value(2.0)
    y = (2 * x - 1) / x
    y.backward()
    
    assert abs(y.data - 1.5) <= 1e-6, f"f(2) = {y.data}, expected 1.5"
    assert abs(x.grad - 0.25) <= 1e-4, f"f'(2) = {x.grad}, expected 0.25"


def test_numerical_gradient_check():
    """Verify against numerical gradient."""
    def f(val):
        x = Value(val)
        return ((x + 1) / (x - 1)).data
    
    x_val = 3.0
    h = 1e-5
    numerical = (f(x_val + h) - f(x_val - h)) / (2 * h)
    
    x = Value(x_val)
    y = (x + 1) / (x - 1)
    y.backward()
    analytical = x.grad
    
    assert abs(analytical - numerical) <= 1e-4, f"Analytical={analytical}, numerical={numerical}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
