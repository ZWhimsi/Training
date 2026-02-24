"""Test Suite for Day 15: Tensor Activation Functions"""

import numpy as np
import pytest

from day15 import Tensor


def test_relu():
    """Test ReLU activation."""
    a = Tensor([-2, -1, 0, 1, 2])
    b = a.relu()
    
    assert b is not None and b.data is not None, "relu returned None"
    expected = [0, 0, 0, 1, 2]
    assert np.allclose(b.data, expected), f"relu = {b.data}"


def test_relu_backward():
    """Test ReLU backward pass."""
    a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    b = a.relu()
    b.backward()
    
    assert a.grad is not None, "Gradient is None"
    expected = [0, 0, 0, 1, 1]
    assert np.allclose(a.grad, expected), f"da = {a.grad}"


def test_sigmoid():
    """Test sigmoid activation."""
    a = Tensor([0.0])
    b = a.sigmoid()
    
    assert b is not None and b.data is not None, "sigmoid returned None"
    assert np.isclose(b.data[0], 0.5), f"sigmoid(0) = {b.data[0]}"


def test_sigmoid_backward():
    """Test sigmoid backward pass."""
    a = Tensor([0.0])
    b = a.sigmoid()
    b.backward()
    
    assert a.grad is not None, "Gradient is None"
    # sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
    assert np.isclose(a.grad[0], 0.25), f"da = {a.grad[0]}"


def test_tanh():
    """Test tanh activation."""
    a = Tensor([0.0])
    b = a.tanh()
    
    assert b is not None and b.data is not None, "tanh returned None"
    assert np.isclose(b.data[0], 0.0), f"tanh(0) = {b.data[0]}"


def test_tanh_backward():
    """Test tanh backward pass."""
    a = Tensor([0.0])
    b = a.tanh()
    b.backward()
    
    assert a.grad is not None, "Gradient is None"
    # tanh'(0) = 1 - tanh²(0) = 1
    assert np.isclose(a.grad[0], 1.0), f"da = {a.grad[0]}"


def test_exp():
    """Test exp activation."""
    a = Tensor([0.0, 1.0])
    b = a.exp()
    
    assert b is not None and b.data is not None, "exp returned None"
    expected = [1.0, np.e]
    assert np.allclose(b.data, expected, rtol=1e-5), f"exp = {b.data}"


def test_exp_backward():
    """Test exp backward pass."""
    a = Tensor([1.0])
    b = a.exp()
    b.backward()
    
    assert a.grad is not None, "Gradient is None"
    # d/dx exp(x) = exp(x)
    assert np.isclose(a.grad[0], np.e), f"da = {a.grad[0]}"


def test_log():
    """Test log activation."""
    a = Tensor([1.0, np.e])
    b = a.log()
    
    assert b is not None and b.data is not None, "log returned None"
    expected = [0.0, 1.0]
    assert np.allclose(b.data, expected, rtol=1e-5), f"log = {b.data}"


def test_log_backward():
    """Test log backward pass."""
    a = Tensor([2.0])
    b = a.log()
    b.backward()
    
    assert a.grad is not None, "Gradient is None"
    # d/dx log(x) = 1/x = 0.5
    assert np.isclose(a.grad[0], 0.5), f"da = {a.grad[0]}"


def test_leaky_relu():
    """Test leaky ReLU activation."""
    a = Tensor([-2, -1, 0, 1, 2])
    alpha = 0.1
    b = a.leaky_relu(alpha=alpha)
    
    assert b is not None and b.data is not None, "leaky_relu returned None"
    expected = [-0.2, -0.1, 0, 1, 2]
    assert np.allclose(b.data, expected), f"leaky_relu = {b.data}"


def test_leaky_relu_backward():
    """Test leaky ReLU backward pass."""
    a = Tensor([-2.0, 2.0])
    alpha = 0.1
    b = a.leaky_relu(alpha=alpha)
    b.backward()
    
    assert a.grad is not None, "Gradient is None"
    expected = [alpha, 1.0]
    assert np.allclose(a.grad, expected), f"da = {a.grad}"


def test_softmax():
    """Test softmax activation."""
    a = Tensor([1.0, 2.0, 3.0])
    b = a.softmax()
    
    assert b is not None and b.data is not None, "softmax returned None"
    
    # Softmax should sum to 1
    assert np.isclose(np.sum(b.data), 1.0), f"softmax sum = {np.sum(b.data)}"
    
    # Larger inputs should have larger outputs
    assert b.data[2] > b.data[1] > b.data[0], f"softmax order wrong: {b.data}"


def test_softmax_stability():
    """Test softmax numerical stability."""
    # Large values that would overflow without stability fix
    a = Tensor([1000.0, 1001.0, 1002.0])
    b = a.softmax()
    
    assert b is not None and b.data is not None, "softmax returned None"
    assert np.isfinite(b.data).all(), f"softmax has inf/nan: {b.data}"
    assert np.isclose(np.sum(b.data), 1.0), f"softmax sum = {np.sum(b.data)}"


def test_activation_chain():
    """Test chaining activations."""
    a = Tensor([[1.0, -1.0], [-2.0, 2.0]])
    b = a.relu().sigmoid()
    
    assert b is not None, "Chain returned None"
    # relu([1, -1]) = [1, 0], sigmoid([1, 0]) ≈ [0.731, 0.5]
    assert b.data[0, 1] == 0.5, f"sigmoid(relu(-1)) = {b.data[0, 1]}"


def test_activation_backward_chain():
    """Test backward pass through activation chain."""
    a = Tensor([[1.0, 2.0]])
    b = a.tanh().exp()
    b.backward()
    
    assert a.grad is not None, "Gradient is None"
    assert np.isfinite(a.grad).all(), f"Gradient has inf/nan: {a.grad}"


def test_gelu():
    """Test GELU activation if implemented."""
    a = Tensor([0.0, 1.0, -1.0])
    
    try:
        b = a.gelu()
        assert b is not None, "gelu returned None"
        # GELU(0) ≈ 0
        assert np.isclose(b.data[0], 0.0, atol=1e-3), f"gelu(0) = {b.data[0]}"
    except AttributeError:
        pytest.skip("GELU not implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
