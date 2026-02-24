"""Test Suite for Day 9: Activation Functions Backward"""

import math
import pytest

from day09 import Value, compare_activations, check_activation_gradient


def test_relu_positive():
    """Test ReLU with positive input."""
    x = Value(3.0)
    y = x.relu()
    
    assert y is not None, "ReLU returned None"
    assert abs(y.data - 3.0) <= 1e-6, f"relu(3) = {y.data}, expected 3.0"
    
    y.backward()
    assert abs(x.grad - 1.0) <= 1e-6, f"d relu(3)/dx = {x.grad}, expected 1.0"


def test_relu_negative():
    """Test ReLU with negative input."""
    x = Value(-3.0)
    y = x.relu()
    
    assert y is not None, "ReLU returned None"
    assert abs(y.data - 0.0) <= 1e-6, f"relu(-3) = {y.data}, expected 0.0"
    
    y.backward()
    assert abs(x.grad - 0.0) <= 1e-6, f"d relu(-3)/dx = {x.grad}, expected 0.0"


def test_relu_zero():
    """Test ReLU at zero."""
    x = Value(0.0)
    y = x.relu()
    
    assert y is not None, "ReLU returned None"
    assert abs(y.data - 0.0) <= 1e-6, f"relu(0) = {y.data}, expected 0.0"
    
    y.backward()
    # Gradient at 0 is commonly defined as 0
    assert x.grad in [0.0, 1.0], f"d relu(0)/dx = {x.grad}, expected 0 or 1"


def test_leaky_relu_positive():
    """Test Leaky ReLU with positive input."""
    x = Value(2.0)
    y = x.leaky_relu(alpha=0.1)
    
    assert y is not None, "Leaky ReLU returned None"
    assert abs(y.data - 2.0) <= 1e-6, f"leaky_relu(2) = {y.data}, expected 2.0"
    
    y.backward()
    assert abs(x.grad - 1.0) <= 1e-6, f"grad = {x.grad}, expected 1.0"


def test_leaky_relu_negative():
    """Test Leaky ReLU with negative input."""
    x = Value(-2.0)
    alpha = 0.1
    y = x.leaky_relu(alpha=alpha)
    
    assert y is not None, "Leaky ReLU returned None"
    expected = alpha * (-2.0)  # -0.2
    assert abs(y.data - expected) <= 1e-6, f"leaky_relu(-2) = {y.data}, expected {expected}"
    
    y.backward()
    assert abs(x.grad - alpha) <= 1e-6, f"grad = {x.grad}, expected {alpha}"


def test_exp_forward():
    """Test exp forward pass."""
    x = Value(1.0)
    y = x.exp()
    
    assert y is not None, "exp returned None"
    expected = math.exp(1.0)
    assert abs(y.data - expected) <= 1e-6, f"exp(1) = {y.data}, expected {expected}"


def test_exp_backward():
    """Test exp backward pass."""
    x = Value(2.0)
    y = x.exp()
    y.backward()
    
    expected_grad = math.exp(2.0)
    assert abs(x.grad - expected_grad) <= 1e-5, f"d exp(2)/dx = {x.grad}, expected {expected_grad}"


def test_sigmoid_values():
    """Test sigmoid at known values."""
    # sigmoid(0) = 0.5
    x = Value(0.0)
    y = x.sigmoid()
    
    assert y is not None, "sigmoid returned None"
    assert abs(y.data - 0.5) <= 1e-6, f"sigmoid(0) = {y.data}, expected 0.5"


def test_sigmoid_backward():
    """Test sigmoid backward pass."""
    x = Value(0.0)
    y = x.sigmoid()
    y.backward()
    
    # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
    assert abs(x.grad - 0.25) <= 1e-6, f"sigmoid'(0) = {x.grad}, expected 0.25"


def test_sigmoid_saturation():
    """Test sigmoid behavior at extreme values."""
    # Large positive value
    x_pos = Value(10.0)
    y_pos = x_pos.sigmoid()
    
    assert abs(y_pos.data - 1.0) <= 0.001, f"sigmoid(10) = {y_pos.data}, expected ~1.0"
    
    y_pos.backward()
    # Gradient should be very small (vanishing gradient)
    assert x_pos.grad <= 0.01, f"sigmoid'(10) = {x_pos.grad}, expected ~0"
    
    # Large negative value
    x_neg = Value(-10.0)
    y_neg = x_neg.sigmoid()
    
    assert abs(y_neg.data - 0.0) <= 0.001, f"sigmoid(-10) = {y_neg.data}, expected ~0.0"


def test_tanh_values():
    """Test tanh at known values."""
    # tanh(0) = 0
    x = Value(0.0)
    y = x.tanh()
    
    assert y is not None, "tanh returned None"
    assert abs(y.data - 0.0) <= 1e-6, f"tanh(0) = {y.data}, expected 0.0"


def test_tanh_backward():
    """Test tanh backward pass."""
    x = Value(0.0)
    y = x.tanh()
    y.backward()
    
    # tanh'(0) = 1 - tanh²(0) = 1 - 0 = 1
    assert abs(x.grad - 1.0) <= 1e-6, f"tanh'(0) = {x.grad}, expected 1.0"


def test_tanh_range():
    """Test tanh output range."""
    # tanh output should be in [-1, 1]
    for val in [-100, -1, 0, 1, 100]:
        x = Value(float(val))
        y = x.tanh()
        assert -1.0001 <= y.data <= 1.0001, f"tanh({val}) = {y.data}, out of range"


def test_softplus():
    """Test softplus activation."""
    x = Value(0.0)
    y = x.softplus()
    
    assert y is not None, "softplus returned None"
    
    # softplus(0) = log(1 + e^0) = log(2) ≈ 0.693
    expected = math.log(2)
    assert abs(y.data - expected) <= 1e-5, f"softplus(0) = {y.data}, expected {expected}"
    
    y.backward()
    # d/dx softplus(x) = sigmoid(x) = 0.5 at x=0
    assert abs(x.grad - 0.5) <= 1e-5, f"softplus'(0) = {x.grad}, expected 0.5"


def test_chain_rule_activation():
    """Test chain rule with activation function."""
    # f(x) = sigmoid(x^2) at x = 1
    x = Value(1.0)
    y = (x ** 2).sigmoid()
    y.backward()
    
    # f(x) = sigmoid(x^2)
    # f'(x) = sigmoid'(x^2) * 2x = sigmoid(x^2) * (1 - sigmoid(x^2)) * 2x
    sig_1 = 1 / (1 + math.exp(-1))  # sigmoid(1)
    expected_grad = sig_1 * (1 - sig_1) * 2
    
    assert abs(x.grad - expected_grad) <= 1e-4, f"grad = {x.grad}, expected {expected_grad}"


def test_numerical_gradient_relu():
    """Verify ReLU gradient numerically."""
    analytical, numerical, matches = check_activation_gradient('relu', 2.0)
    assert matches, f"analytical={analytical}, numerical={numerical}"


def test_numerical_gradient_sigmoid():
    """Verify sigmoid gradient numerically."""
    analytical, numerical, matches = check_activation_gradient('sigmoid', 0.5)
    assert matches, f"analytical={analytical}, numerical={numerical}"


def test_numerical_gradient_tanh():
    """Verify tanh gradient numerically."""
    analytical, numerical, matches = check_activation_gradient('tanh', 0.5)
    assert matches, f"analytical={analytical}, numerical={numerical}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
