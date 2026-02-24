"""Test Suite for Day 11: Backward Pass"""

import math
import pytest

from day11 import Value, gradient_check


def test_simple_backward():
    x = Value(3.0)
    y = x ** 2
    y.backward()
    
    # d/dx(x^2) = 2x = 6 at x=3
    assert abs(x.grad - 6.0) <= 1e-5, f"x.grad = {x.grad}, expected 6.0"


def test_addition_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    
    # d(a+b)/da = 1, d(a+b)/db = 1
    assert abs(a.grad - 1.0) <= 1e-5 and abs(b.grad - 1.0) <= 1e-5, f"Grads: a={a.grad}, b={b.grad}, expected 1, 1"


def test_multiplication_backward():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    
    # d(a*b)/da = b = 3, d(a*b)/db = a = 2
    assert abs(a.grad - 3.0) <= 1e-5, f"a.grad = {a.grad}, expected 3.0"
    assert abs(b.grad - 2.0) <= 1e-5, f"b.grad = {b.grad}, expected 2.0"


def test_chain_rule():
    x = Value(2.0)
    y = (x * 2 + 1) ** 2  # (2*2+1)^2 = 25
    y.backward()
    
    # y = (2x + 1)^2
    # dy/dx = 2 * (2x + 1) * 2 = 4 * (2*2+1) = 20
    assert abs(y.data - 25.0) <= 1e-5, f"y.data = {y.data}, expected 25"
    assert abs(x.grad - 20.0) <= 1e-5, f"x.grad = {x.grad}, expected 20"


def test_relu_backward():
    # Positive input
    x1 = Value(3.0)
    y1 = x1.relu()
    y1.backward()
    
    assert abs(y1.data - 3.0) <= 1e-5, f"relu(3) = {y1.data}, expected 3"
    assert abs(x1.grad - 1.0) <= 1e-5, f"d relu(3)/dx = {x1.grad}, expected 1"
    
    # Negative input
    x2 = Value(-3.0)
    y2 = x2.relu()
    y2.backward()
    
    assert abs(y2.data - 0.0) <= 1e-5, f"relu(-3) = {y2.data}, expected 0"
    assert abs(x2.grad - 0.0) <= 1e-5, f"d relu(-3)/dx = {x2.grad}, expected 0"


def test_exp_backward():
    x = Value(1.0)
    y = x.exp()
    y.backward()
    
    expected = math.exp(1.0)
    assert abs(y.data - expected) <= 1e-5, f"exp(1) = {y.data}, expected {expected}"
    assert abs(x.grad - expected) <= 1e-5, f"d exp(1)/dx = {x.grad}, expected {expected}"


def test_tanh_backward():
    x = Value(0.5)
    y = x.tanh()
    y.backward()
    
    expected_y = math.tanh(0.5)
    expected_grad = 1 - expected_y ** 2
    
    assert abs(y.data - expected_y) <= 1e-5, f"tanh(0.5) = {y.data}, expected {expected_y}"
    assert abs(x.grad - expected_grad) <= 1e-4, f"d tanh/dx = {x.grad}, expected {expected_grad}"


def test_gradient_accumulation():
    # When a value is used multiple times, gradients should accumulate
    x = Value(2.0)
    y = x + x  # x used twice
    y.backward()
    
    # d(x + x)/dx = 2
    assert abs(x.grad - 2.0) <= 1e-5, f"d(x+x)/dx = {x.grad}, expected 2"
    
    # Reset and test multiplication
    x2 = Value(3.0)
    y2 = x2 * x2  # x^2
    y2.backward()
    
    # d(x^2)/dx = 2x = 6
    assert abs(x2.grad - 6.0) <= 1e-5, f"d(x*x)/dx = {x2.grad}, expected 6"


def test_complex_expression():
    # Test a complex expression
    x = Value(2.0)
    y = Value(3.0)
    z = (x * y + x ** 2).relu()
    z.backward()
    
    # z = relu(x*y + x^2) = relu(6 + 4) = 10
    # dz/dx = (y + 2x) * (1 if input>0 else 0) = (3 + 4) * 1 = 7
    # dz/dy = x * 1 = 2
    
    assert abs(z.data - 10.0) <= 1e-5, f"z = {z.data}, expected 10"
    assert abs(x.grad - 7.0) <= 1e-5, f"dz/dx = {x.grad}, expected 7"
    assert abs(y.grad - 2.0) <= 1e-5, f"dz/dy = {y.grad}, expected 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
