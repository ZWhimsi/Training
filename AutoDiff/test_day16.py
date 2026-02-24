"""Test Suite for Day 16: Loss Functions"""

import numpy as np
import pytest

from day16 import Tensor, MSELoss, CrossEntropyLoss


def test_mse_zero():
    """Test MSE with identical inputs."""
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([1.0, 2.0, 3.0])
    
    loss_fn = MSELoss()
    loss = loss_fn(pred, target)
    
    assert loss is not None, "MSE returned None"
    assert np.isclose(loss.data, 0.0), f"MSE = {loss.data}, expected 0"


def test_mse_basic():
    """Test basic MSE calculation."""
    pred = Tensor([0.0, 0.0, 0.0])
    target = Tensor([1.0, 2.0, 3.0])
    
    loss_fn = MSELoss()
    loss = loss_fn(pred, target)
    
    assert loss is not None, "MSE returned None"
    # MSE = (1 + 4 + 9) / 3 = 14/3 ≈ 4.67
    expected = (1 + 4 + 9) / 3
    assert np.isclose(loss.data, expected), f"MSE = {loss.data}, expected {expected}"


def test_mse_backward():
    """Test MSE backward pass."""
    pred = Tensor([1.0, 2.0])
    target = Tensor([2.0, 4.0])
    
    loss_fn = MSELoss()
    loss = loss_fn(pred, target)
    loss.backward()
    
    assert pred.grad is not None, "Gradient is None"
    # d/d_pred MSE = 2 * (pred - target) / n = 2 * [-1, -2] / 2 = [-1, -2]
    expected = [-1.0, -2.0]
    assert np.allclose(pred.grad, expected), f"d_pred = {pred.grad}"


def test_mse_2d():
    """Test MSE on 2D tensors."""
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    target = Tensor([[1.0, 1.0], [1.0, 1.0]])
    
    loss_fn = MSELoss()
    loss = loss_fn(pred, target)
    
    assert loss is not None, "MSE returned None"
    # MSE = (0 + 1 + 4 + 9) / 4 = 14/4 = 3.5
    expected = 3.5
    assert np.isclose(loss.data, expected), f"MSE = {loss.data}, expected {expected}"


def test_cross_entropy_perfect():
    """Test CE with perfect predictions."""
    # One-hot target [0, 1, 0], perfect logits [small, large, small]
    logits = Tensor([[-10.0, 10.0, -10.0]])
    target = Tensor([1])  # Class 1
    
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, target)
    
    assert loss is not None, "CE returned None"
    # Should be near 0 for perfect prediction
    assert loss.data <= 0.1, f"CE = {loss.data}, expected ~0"


def test_cross_entropy_basic():
    """Test basic cross entropy calculation."""
    logits = Tensor([[1.0, 2.0, 3.0]])
    target = Tensor([0])  # Target is class 0
    
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, target)
    
    assert loss is not None, "CE returned None"
    assert loss.data > 0, f"CE should be positive, got {loss.data}"
    
    # For logits [1,2,3], target 0:
    # softmax = exp([1,2,3]) / sum = [0.09, 0.24, 0.67]
    # CE = -log(0.09) ≈ 2.4
    assert 2.0 <= loss.data <= 3.0, f"CE = {loss.data}, expected ~2.4"


def test_cross_entropy_backward():
    """Test CE backward pass."""
    logits = Tensor([[1.0, 2.0, 3.0]])
    target = Tensor([2])  # Target is class 2
    
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, target)
    loss.backward()
    
    assert logits.grad is not None, "Gradient is None"
    
    # Gradient should be (softmax - one_hot)
    # For target=2, one_hot = [0, 0, 1]
    # Gradient[2] should be negative (softmax[2] - 1 < 0)
    assert logits.grad[0, 2] < 0, f"Grad at target should be negative: {logits.grad}"


def test_cross_entropy_batch():
    """Test CE on batch of predictions."""
    logits = Tensor([[1.0, 2.0], [3.0, 4.0]])  # Batch of 2
    target = Tensor([1, 0])  # Targets
    
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, target)
    
    assert loss is not None, "CE returned None"
    assert loss.data > 0, f"CE should be positive"


def test_mse_sum_reduction():
    """Test MSE with sum reduction."""
    pred = Tensor([1.0, 2.0])
    target = Tensor([0.0, 0.0])
    
    loss_fn = MSELoss(reduction='sum')
    loss = loss_fn(pred, target)
    
    assert loss is not None, "MSE returned None"
    # sum = 1 + 4 = 5
    expected = 5.0
    assert np.isclose(loss.data, expected), f"MSE sum = {loss.data}, expected {expected}"


def test_mse_none_reduction():
    """Test MSE with no reduction."""
    pred = Tensor([1.0, 2.0])
    target = Tensor([0.0, 0.0])
    
    loss_fn = MSELoss(reduction='none')
    loss = loss_fn(pred, target)
    
    assert loss is not None, "MSE returned None"
    expected = [1.0, 4.0]
    assert np.allclose(loss.data, expected), f"MSE none = {loss.data}, expected {expected}"


def test_loss_function_callable():
    """Test that loss functions are callable."""
    mse = MSELoss()
    ce = CrossEntropyLoss()
    
    assert callable(mse), "MSELoss should be callable"
    assert callable(ce), "CrossEntropyLoss should be callable"


def test_numerical_gradient_mse():
    """Verify MSE gradient numerically."""
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([1.5, 2.5, 3.5])
    
    loss_fn = MSELoss()
    loss = loss_fn(pred, target)
    loss.backward()
    
    analytical = pred.grad.copy()
    
    eps = 1e-5
    for i in range(3):
        pred_plus = pred.data.copy()
        pred_plus[i] += eps
        loss_plus = np.mean((pred_plus - target.data) ** 2)
        
        pred_minus = pred.data.copy()
        pred_minus[i] -= eps
        loss_minus = np.mean((pred_minus - target.data) ** 2)
        
        numerical = (loss_plus - loss_minus) / (2 * eps)
        
        assert abs(numerical - analytical[i]) <= 1e-4, f"Grad mismatch at {i}: {analytical[i]} vs {numerical}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
