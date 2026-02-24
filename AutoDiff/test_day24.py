"""Test Suite for Day 24: More Activation Functions"""

import numpy as np
import pytest

from day24 import Tensor, Sigmoid, Tanh, LeakyReLU


def test_sigmoid_module():
    """Test Sigmoid as a Module."""
    sigmoid = Sigmoid()
    x = Tensor([0.0])
    y = sigmoid(x)
    
    assert y is not None, "Sigmoid returned None"
    assert np.isclose(y.data[0], 0.5), f"sigmoid(0) = {y.data[0]}, expected 0.5"


def test_sigmoid_extreme_values():
    """Test Sigmoid at extreme values."""
    sigmoid = Sigmoid()
    
    x_pos = Tensor([10.0])
    y_pos = sigmoid(x_pos)
    assert np.isclose(y_pos.data[0], 1.0, atol=0.001), f"sigmoid(10) = {y_pos.data[0]}"
    
    x_neg = Tensor([-10.0])
    y_neg = sigmoid(x_neg)
    assert np.isclose(y_neg.data[0], 0.0, atol=0.001), f"sigmoid(-10) = {y_neg.data[0]}"


def test_sigmoid_backward():
    """Test Sigmoid backward pass."""
    sigmoid = Sigmoid()
    x = Tensor([0.0])
    y = sigmoid(x)
    y.backward()
    
    assert x.grad is not None, "Gradient is None"
    # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.25
    assert np.isclose(x.grad[0], 0.25), f"grad = {x.grad[0]}, expected 0.25"


def test_sigmoid_no_parameters():
    """Test that Sigmoid has no parameters."""
    sigmoid = Sigmoid()
    params = list(sigmoid.parameters())
    
    assert len(params) == 0, f"Sigmoid should have 0 params, got {len(params)}"


def test_tanh_module():
    """Test Tanh as a Module."""
    tanh = Tanh()
    x = Tensor([0.0])
    y = tanh(x)
    
    assert y is not None, "Tanh returned None"
    assert np.isclose(y.data[0], 0.0), f"tanh(0) = {y.data[0]}, expected 0.0"


def test_tanh_range():
    """Test Tanh output range."""
    tanh = Tanh()
    x = Tensor([-100.0, 0.0, 100.0])
    y = tanh(x)
    
    assert y.data[0] >= -1.0001 and y.data[0] <= -0.999, f"tanh(-100) = {y.data[0]}"
    assert np.isclose(y.data[1], 0.0), f"tanh(0) = {y.data[1]}"
    assert y.data[2] >= 0.999 and y.data[2] <= 1.0001, f"tanh(100) = {y.data[2]}"


def test_tanh_backward():
    """Test Tanh backward pass."""
    tanh = Tanh()
    x = Tensor([0.0])
    y = tanh(x)
    y.backward()
    
    assert x.grad is not None, "Gradient is None"
    # tanh'(0) = 1 - tanh(0)^2 = 1
    assert np.isclose(x.grad[0], 1.0), f"grad = {x.grad[0]}, expected 1.0"


def test_tanh_no_parameters():
    """Test that Tanh has no parameters."""
    tanh = Tanh()
    params = list(tanh.parameters())
    
    assert len(params) == 0, f"Tanh should have 0 params, got {len(params)}"


def test_leaky_relu_positive():
    """Test LeakyReLU with positive input."""
    leaky = LeakyReLU(alpha=0.1)
    x = Tensor([3.0])
    y = leaky(x)
    
    assert y is not None, "LeakyReLU returned None"
    assert np.isclose(y.data[0], 3.0), f"leaky_relu(3) = {y.data[0]}, expected 3.0"


def test_leaky_relu_negative():
    """Test LeakyReLU with negative input."""
    leaky = LeakyReLU(alpha=0.1)
    x = Tensor([-3.0])
    y = leaky(x)
    
    assert y is not None, "LeakyReLU returned None"
    expected = 0.1 * (-3.0)
    assert np.isclose(y.data[0], expected), f"leaky_relu(-3) = {y.data[0]}, expected {expected}"


def test_leaky_relu_backward():
    """Test LeakyReLU backward pass."""
    alpha = 0.1
    leaky = LeakyReLU(alpha=alpha)
    
    # Positive input
    x_pos = Tensor([2.0])
    y_pos = leaky(x_pos)
    y_pos.backward()
    assert np.isclose(x_pos.grad[0], 1.0), f"grad (pos) = {x_pos.grad[0]}, expected 1.0"
    
    # Negative input
    x_neg = Tensor([-2.0])
    y_neg = leaky(x_neg)
    y_neg.backward()
    assert np.isclose(x_neg.grad[0], alpha), f"grad (neg) = {x_neg.grad[0]}, expected {alpha}"


def test_leaky_relu_default_alpha():
    """Test LeakyReLU default alpha."""
    leaky = LeakyReLU()
    x = Tensor([-1.0])
    y = leaky(x)
    
    # Default alpha is usually 0.01
    assert y.data[0] < 0, "leaky_relu(-1) should be negative"
    assert y.data[0] >= -0.1, f"leaky_relu(-1) = {y.data[0]}, alpha seems too large"


def test_leaky_relu_no_parameters():
    """Test that LeakyReLU has no learnable parameters."""
    leaky = LeakyReLU()
    params = list(leaky.parameters())
    
    assert len(params) == 0, f"LeakyReLU should have 0 params, got {len(params)}"


def test_activation_batch():
    """Test activations on batch input."""
    sigmoid = Sigmoid()
    tanh = Tanh()
    leaky = LeakyReLU(alpha=0.1)
    
    x = Tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]])
    
    y_sig = sigmoid(x)
    y_tanh = tanh(x)
    y_leaky = leaky(x)
    
    assert y_sig.shape == (2, 3), f"sigmoid shape = {y_sig.shape}"
    assert y_tanh.shape == (2, 3), f"tanh shape = {y_tanh.shape}"
    assert y_leaky.shape == (2, 3), f"leaky shape = {y_leaky.shape}"


def test_activation_chain():
    """Test chaining activations."""
    from day19 import Linear, Sequential
    
    model = Sequential(
        Linear(2, 4),
        LeakyReLU(),
        Linear(4, 2),
        Sigmoid()
    )
    
    x = Tensor([[1.0, 2.0]])
    y = model(x)
    
    assert y is not None, "Chain returned None"
    assert y.shape == (1, 2), f"shape = {y.shape}"
    # Output should be in [0, 1] due to final Sigmoid
    assert np.all(y.data >= 0) and np.all(y.data <= 1), f"y = {y.data}"


def test_activation_repr():
    """Test activation string representation."""
    sigmoid = Sigmoid()
    tanh = Tanh()
    leaky = LeakyReLU(alpha=0.2)
    
    assert repr(sigmoid) is not None and len(repr(sigmoid)) > 0, "Sigmoid repr empty"
    assert repr(tanh) is not None and len(repr(tanh)) > 0, "Tanh repr empty"
    assert repr(leaky) is not None and len(repr(leaky)) > 0, "LeakyReLU repr empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
