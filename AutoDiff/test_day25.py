"""Test Suite for Day 25: GELU and Softplus"""

import numpy as np
import pytest

from day25 import Tensor, GELU, Softplus


def test_gelu_zero():
    """Test GELU at zero."""
    gelu = GELU()
    x = Tensor([0.0])
    y = gelu(x)
    
    assert y is not None, "GELU returned None"
    # GELU(0) = 0 * Phi(0) = 0 * 0.5 = 0
    assert np.isclose(y.data[0], 0.0, atol=1e-5), f"gelu(0) = {y.data[0]}, expected 0"


def test_gelu_positive():
    """Test GELU with positive input."""
    gelu = GELU()
    x = Tensor([2.0])
    y = gelu(x)
    
    assert y is not None, "GELU returned None"
    # GELU(2) ≈ 1.954
    assert np.isclose(y.data[0], 1.954, atol=0.01), f"gelu(2) = {y.data[0]}"


def test_gelu_negative():
    """Test GELU with negative input."""
    gelu = GELU()
    x = Tensor([-2.0])
    y = gelu(x)
    
    assert y is not None, "GELU returned None"
    # GELU(-2) ≈ -0.0454
    assert y.data[0] < 0, f"gelu(-2) should be negative, got {y.data[0]}"
    assert np.isclose(y.data[0], -0.0454, atol=0.01), f"gelu(-2) = {y.data[0]}"


def test_gelu_backward():
    """Test GELU backward pass."""
    gelu = GELU()
    x = Tensor([0.0])
    y = gelu(x)
    y.backward()
    
    assert x.grad is not None, "Gradient is None"
    # GELU'(0) ≈ 0.5
    assert np.isclose(x.grad[0], 0.5, atol=0.01), f"gelu'(0) = {x.grad[0]}, expected ~0.5"


def test_gelu_numerical_gradient():
    """Verify GELU gradient numerically."""
    gelu = GELU()
    x = Tensor([1.0])
    y = gelu(x)
    y.backward()
    
    analytical = x.grad[0]
    
    # Numerical gradient
    eps = 1e-5
    x_plus = Tensor([1.0 + eps])
    x_minus = Tensor([1.0 - eps])
    numerical = (gelu(x_plus).data[0] - gelu(x_minus).data[0]) / (2 * eps)
    
    assert np.isclose(analytical, numerical, atol=1e-4), f"analytical={analytical}, numerical={numerical}"


def test_gelu_no_parameters():
    """Test that GELU has no parameters."""
    gelu = GELU()
    params = list(gelu.parameters())
    
    assert len(params) == 0, f"GELU should have 0 params, got {len(params)}"


def test_softplus_zero():
    """Test Softplus at zero."""
    softplus = Softplus()
    x = Tensor([0.0])
    y = softplus(x)
    
    assert y is not None, "Softplus returned None"
    # softplus(0) = log(1 + exp(0)) = log(2) ≈ 0.693
    expected = np.log(2)
    assert np.isclose(y.data[0], expected, atol=1e-5), f"softplus(0) = {y.data[0]}, expected {expected}"


def test_softplus_positive():
    """Test Softplus with positive input."""
    softplus = Softplus()
    x = Tensor([5.0])
    y = softplus(x)
    
    assert y is not None, "Softplus returned None"
    # For large x, softplus(x) ≈ x
    assert np.isclose(y.data[0], 5.0, atol=0.01), f"softplus(5) = {y.data[0]}, expected ~5"


def test_softplus_negative():
    """Test Softplus with negative input."""
    softplus = Softplus()
    x = Tensor([-5.0])
    y = softplus(x)
    
    assert y is not None, "Softplus returned None"
    # For large negative x, softplus(x) ≈ 0
    assert y.data[0] > 0, "softplus should always be positive"
    assert np.isclose(y.data[0], 0.0, atol=0.01), f"softplus(-5) = {y.data[0]}, expected ~0"


def test_softplus_backward():
    """Test Softplus backward pass."""
    softplus = Softplus()
    x = Tensor([0.0])
    y = softplus(x)
    y.backward()
    
    assert x.grad is not None, "Gradient is None"
    # d/dx softplus(x) = sigmoid(x) = 0.5 at x=0
    assert np.isclose(x.grad[0], 0.5, atol=1e-5), f"softplus'(0) = {x.grad[0]}, expected 0.5"


def test_softplus_numerical_gradient():
    """Verify Softplus gradient numerically."""
    softplus = Softplus()
    x = Tensor([1.0])
    y = softplus(x)
    y.backward()
    
    analytical = x.grad[0]
    
    eps = 1e-5
    x_plus = Tensor([1.0 + eps])
    x_minus = Tensor([1.0 - eps])
    numerical = (softplus(x_plus).data[0] - softplus(x_minus).data[0]) / (2 * eps)
    
    assert np.isclose(analytical, numerical, atol=1e-4), f"analytical={analytical}, numerical={numerical}"


def test_softplus_stability():
    """Test Softplus numerical stability."""
    softplus = Softplus()
    
    # Very large positive value
    x_large = Tensor([100.0])
    y_large = softplus(x_large)
    assert np.isfinite(y_large.data[0]), f"softplus(100) = {y_large.data[0]}"
    assert np.isclose(y_large.data[0], 100.0, atol=0.1), f"softplus(100) = {y_large.data[0]}"
    
    # Very large negative value
    x_small = Tensor([-100.0])
    y_small = softplus(x_small)
    assert np.isfinite(y_small.data[0]), f"softplus(-100) = {y_small.data[0]}"
    assert y_small.data[0] >= 0, f"softplus should be non-negative"


def test_softplus_no_parameters():
    """Test that Softplus has no parameters."""
    softplus = Softplus()
    params = list(softplus.parameters())
    
    assert len(params) == 0, f"Softplus should have 0 params, got {len(params)}"


def test_activations_batch():
    """Test activations on batch input."""
    gelu = GELU()
    softplus = Softplus()
    
    x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    
    y_gelu = gelu(x)
    y_softplus = softplus(x)
    
    assert y_gelu.shape == (1, 5), f"GELU shape = {y_gelu.shape}"
    assert y_softplus.shape == (1, 5), f"Softplus shape = {y_softplus.shape}"


def test_softplus_vs_relu():
    """Test that Softplus is smooth approximation of ReLU."""
    from day19 import ReLU
    
    relu = ReLU()
    softplus = Softplus()
    
    x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    y_relu = relu(x)
    y_softplus = softplus(x)
    
    # For positive values, softplus ≈ relu + small constant
    diff = np.abs(y_softplus.data - y_relu.data)
    assert np.all(diff < 0.5), f"Softplus should approximate ReLU for positive x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
