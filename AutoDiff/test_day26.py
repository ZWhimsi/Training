"""Test Suite for Day 26: ELU Activation"""

import numpy as np
import pytest

from day26 import Tensor, ELU


def test_elu_positive():
    """Test ELU with positive input."""
    elu = ELU(alpha=1.0)
    x = Tensor([3.0])
    y = elu(x)
    
    assert y is not None, "ELU returned None"
    # ELU(x) = x for x > 0
    assert np.isclose(y.data[0], 3.0), f"elu(3) = {y.data[0]}, expected 3.0"


def test_elu_negative():
    """Test ELU with negative input."""
    elu = ELU(alpha=1.0)
    x = Tensor([-1.0])
    y = elu(x)
    
    assert y is not None, "ELU returned None"
    # ELU(x) = alpha * (exp(x) - 1) for x <= 0
    expected = 1.0 * (np.exp(-1.0) - 1)
    assert np.isclose(y.data[0], expected, atol=1e-5), f"elu(-1) = {y.data[0]}, expected {expected}"


def test_elu_zero():
    """Test ELU at zero."""
    elu = ELU(alpha=1.0)
    x = Tensor([0.0])
    y = elu(x)
    
    assert y is not None, "ELU returned None"
    # ELU(0) = 0
    assert np.isclose(y.data[0], 0.0), f"elu(0) = {y.data[0]}, expected 0.0"


def test_elu_different_alpha():
    """Test ELU with different alpha values."""
    x = Tensor([-1.0])
    
    elu1 = ELU(alpha=1.0)
    elu2 = ELU(alpha=2.0)
    
    y1 = elu1(x)
    y2 = elu2(x)
    
    # y2 should be 2x y1 for negative inputs
    assert np.isclose(y2.data[0], 2 * y1.data[0]), f"alpha scaling: {y1.data[0]} vs {y2.data[0]}"


def test_elu_backward_positive():
    """Test ELU backward pass for positive input."""
    elu = ELU(alpha=1.0)
    x = Tensor([2.0])
    y = elu(x)
    y.backward()
    
    assert x.grad is not None, "Gradient is None"
    # d/dx ELU(x) = 1 for x > 0
    assert np.isclose(x.grad[0], 1.0), f"elu'(2) = {x.grad[0]}, expected 1.0"


def test_elu_backward_negative():
    """Test ELU backward pass for negative input."""
    elu = ELU(alpha=1.0)
    x = Tensor([-1.0])
    y = elu(x)
    y.backward()
    
    assert x.grad is not None, "Gradient is None"
    # d/dx ELU(x) = alpha * exp(x) for x <= 0
    expected = 1.0 * np.exp(-1.0)
    assert np.isclose(x.grad[0], expected, atol=1e-5), f"elu'(-1) = {x.grad[0]}, expected {expected}"


def test_elu_numerical_gradient():
    """Verify ELU gradient numerically."""
    elu = ELU(alpha=1.0)
    
    for test_x in [-2.0, -0.5, 0.5, 2.0]:
        x = Tensor([test_x])
        y = elu(x)
        y.backward()
        
        analytical = x.grad[0]
        
        eps = 1e-5
        x_plus = Tensor([test_x + eps])
        x_minus = Tensor([test_x - eps])
        numerical = (elu(x_plus).data[0] - elu(x_minus).data[0]) / (2 * eps)
        
        assert np.isclose(analytical, numerical, atol=1e-4), f"x={test_x}: analytical={analytical}, numerical={numerical}"


def test_elu_continuity():
    """Test ELU is continuous at x=0."""
    elu = ELU(alpha=1.0)
    
    # Values approaching 0 from both sides
    x_pos = Tensor([1e-8])
    x_neg = Tensor([-1e-8])
    
    y_pos = elu(x_pos)
    y_neg = elu(x_neg)
    
    # Should be approximately equal (continuous)
    assert np.isclose(y_pos.data[0], y_neg.data[0], atol=1e-5), f"Not continuous: {y_pos.data[0]} vs {y_neg.data[0]}"


def test_elu_smoothness():
    """Test ELU derivative is continuous at x=0."""
    elu = ELU(alpha=1.0)
    
    # Derivative at 0+ should be 1
    # Derivative at 0- should be alpha * exp(0) = alpha = 1
    x_pos = Tensor([1e-6])
    y_pos = elu(x_pos)
    y_pos.backward()
    grad_pos = x_pos.grad[0]
    
    x_neg = Tensor([-1e-6])
    y_neg = elu(x_neg)
    y_neg.backward()
    grad_neg = x_neg.grad[0]
    
    assert np.isclose(grad_pos, grad_neg, atol=1e-4), f"Derivative not continuous: {grad_pos} vs {grad_neg}"


def test_elu_saturation():
    """Test ELU saturation behavior."""
    elu = ELU(alpha=1.0)
    
    x = Tensor([-100.0])
    y = elu(x)
    
    # For very negative x, ELU -> -alpha
    assert np.isclose(y.data[0], -1.0, atol=1e-5), f"elu(-100) = {y.data[0]}, expected -1.0"


def test_elu_no_parameters():
    """Test that ELU has no learnable parameters."""
    elu = ELU()
    params = list(elu.parameters())
    
    assert len(params) == 0, f"ELU should have 0 params, got {len(params)}"


def test_elu_batch():
    """Test ELU on batch input."""
    elu = ELU(alpha=1.0)
    
    x = Tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    y = elu(x)
    
    assert y.shape == (1, 5), f"shape = {y.shape}"
    
    # Check positive values unchanged
    assert np.isclose(y.data[0, 3], 1.0), f"elu(1) = {y.data[0, 3]}"
    assert np.isclose(y.data[0, 4], 2.0), f"elu(2) = {y.data[0, 4]}"
    
    # Check negative values are negative but > -alpha
    assert y.data[0, 0] > -1.0 and y.data[0, 0] < 0, f"elu(-2) = {y.data[0, 0]}"


def test_elu_default_alpha():
    """Test ELU default alpha value."""
    elu = ELU()
    
    x = Tensor([-1.0])
    y = elu(x)
    
    # Default alpha is usually 1.0
    expected_alpha1 = np.exp(-1.0) - 1
    assert np.isclose(y.data[0], expected_alpha1, atol=0.1), f"elu(-1) = {y.data[0]}"


def test_elu_repr():
    """Test ELU string representation."""
    elu = ELU(alpha=0.5)
    s = repr(elu)
    
    assert s is not None and len(s) > 0, "ELU repr empty"
    assert 'ELU' in s or '0.5' in s, f"repr = {s}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
