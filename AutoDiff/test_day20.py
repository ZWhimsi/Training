"""Test Suite for Day 20: Optimizer Base and SGD"""

import numpy as np
import pytest

from day20 import Tensor, Parameter, Module, Linear, Optimizer, SGD


def test_optimizer_creation():
    """Test Optimizer creation."""
    p = Parameter([1.0, 2.0])
    opt = Optimizer([p], lr=0.1)
    
    assert opt is not None, "Optimizer returned None"
    assert hasattr(opt, 'param_groups'), "No param_groups attribute"


def test_optimizer_lr():
    """Test Optimizer learning rate."""
    p = Parameter([1.0])
    opt = Optimizer([p], lr=0.01)
    
    assert opt.param_groups[0]['lr'] == 0.01, f"lr = {opt.param_groups[0]['lr']}"


def test_sgd_creation():
    """Test SGD optimizer creation."""
    p = Parameter([1.0, 2.0])
    opt = SGD([p], lr=0.1)
    
    assert opt is not None, "SGD returned None"


def test_sgd_step():
    """Test SGD step."""
    p = Parameter([1.0, 2.0])
    p.grad = np.array([1.0, 1.0])
    
    opt = SGD([p], lr=0.1)
    opt.step()
    
    # p_new = p - lr * grad = [1, 2] - 0.1 * [1, 1] = [0.9, 1.9]
    expected = [0.9, 1.9]
    assert np.allclose(p.data, expected), f"p = {p.data}, expected {expected}"


def test_sgd_zero_grad():
    """Test SGD zero_grad."""
    p = Parameter([1.0])
    p.grad = np.array([5.0])
    
    opt = SGD([p], lr=0.1)
    opt.zero_grad()
    
    assert np.allclose(p.grad, 0), f"grad = {p.grad}, expected 0"


def test_sgd_momentum():
    """Test SGD with momentum."""
    p = Parameter([0.0])
    opt = SGD([p], lr=0.1, momentum=0.9)
    
    # First step
    p.grad = np.array([1.0])
    opt.step()
    
    # v = 0.9 * 0 + 1.0 = 1.0
    # p = 0 - 0.1 * 1.0 = -0.1
    assert np.isclose(p.data[0], -0.1), f"After step 1: p = {p.data[0]}"
    
    # Second step with same gradient
    p.grad = np.array([1.0])
    opt.step()
    
    # v = 0.9 * 1.0 + 1.0 = 1.9
    # p = -0.1 - 0.1 * 1.9 = -0.29
    assert np.isclose(p.data[0], -0.29), f"After step 2: p = {p.data[0]}"


def test_sgd_weight_decay():
    """Test SGD with weight decay."""
    p = Parameter([1.0])
    p.grad = np.array([0.0])  # Zero gradient
    
    opt = SGD([p], lr=0.1, weight_decay=0.1)
    opt.step()
    
    # With weight decay: grad = 0 + 0.1 * 1.0 = 0.1
    # p = 1.0 - 0.1 * 0.1 = 0.99
    assert np.isclose(p.data[0], 0.99), f"p = {p.data[0]}, expected 0.99"


def test_sgd_multiple_params():
    """Test SGD with multiple parameters."""
    p1 = Parameter([1.0, 2.0])
    p2 = Parameter([3.0])
    
    p1.grad = np.array([0.1, 0.2])
    p2.grad = np.array([0.3])
    
    opt = SGD([p1, p2], lr=1.0)
    opt.step()
    
    assert np.allclose(p1.data, [0.9, 1.8]), f"p1 = {p1.data}"
    assert np.allclose(p2.data, [2.7]), f"p2 = {p2.data}"


def test_sgd_with_model():
    """Test SGD with Module."""
    model = Linear(2, 1)
    opt = SGD(model.parameters(), lr=0.01)
    
    x = Tensor([[1.0, 2.0]])
    y = model(x)
    loss = y.sum()
    
    loss.backward()
    opt.step()
    
    # Weights should have changed
    assert model.weight.grad is not None, "No gradient computed"


def test_sgd_training_loop():
    """Test SGD in a training loop."""
    model = Linear(1, 1)
    model.weight.data = np.array([[0.0]])
    model.bias.data = np.array([0.0])
    
    opt = SGD(model.parameters(), lr=0.1)
    
    # Simple regression: y = 2 * x
    x = Tensor([[1.0], [2.0], [3.0]])
    target = Tensor([[2.0], [4.0], [6.0]])
    
    initial_loss = None
    final_loss = None
    
    for i in range(100):
        opt.zero_grad()
        pred = model(x)
        loss = ((pred - target) ** 2).mean()
        
        if i == 0:
            initial_loss = loss.data
        
        loss.backward()
        opt.step()
        
        final_loss = loss.data
    
    assert final_loss < initial_loss, f"Loss didn't decrease: {initial_loss} -> {final_loss}"


def test_optimizer_param_groups():
    """Test Optimizer with param groups."""
    p1 = Parameter([1.0])
    p2 = Parameter([2.0])
    
    opt = SGD([
        {'params': [p1], 'lr': 0.1},
        {'params': [p2], 'lr': 0.01}
    ])
    
    assert len(opt.param_groups) == 2, f"Should have 2 groups, got {len(opt.param_groups)}"
    assert opt.param_groups[0]['lr'] == 0.1, "Group 0 lr wrong"
    assert opt.param_groups[1]['lr'] == 0.01, "Group 1 lr wrong"


def test_sgd_nesterov():
    """Test SGD with Nesterov momentum."""
    p = Parameter([0.0])
    
    try:
        opt = SGD([p], lr=0.1, momentum=0.9, nesterov=True)
        p.grad = np.array([1.0])
        opt.step()
        # Just check it runs without error
        assert True
    except (TypeError, AttributeError):
        pytest.skip("Nesterov momentum not implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
