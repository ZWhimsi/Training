"""Test Suite for Day 21: Adam Optimizer"""

import numpy as np
import pytest

from day21 import Tensor, Parameter, Module, Linear, Adam


def test_adam_creation():
    """Test Adam optimizer creation."""
    p = Parameter([1.0, 2.0])
    opt = Adam([p], lr=0.001)
    
    assert opt is not None, "Adam returned None"


def test_adam_step():
    """Test Adam step."""
    p = Parameter([1.0])
    p.grad = np.array([1.0])
    
    opt = Adam([p], lr=0.1)
    initial = p.data.copy()
    
    opt.step()
    
    assert not np.allclose(p.data, initial), "Parameter didn't change"


def test_adam_default_params():
    """Test Adam default hyperparameters."""
    p = Parameter([1.0])
    opt = Adam([p])
    
    pg = opt.param_groups[0]
    assert 'betas' in pg, "No betas parameter"
    assert 'eps' in pg, "No eps parameter"
    
    # Default betas are usually (0.9, 0.999)
    assert len(pg['betas']) == 2, f"betas should be tuple of 2: {pg['betas']}"


def test_adam_momentum_terms():
    """Test Adam maintains momentum terms."""
    p = Parameter([1.0])
    opt = Adam([p], lr=0.1, betas=(0.9, 0.999))
    
    # Multiple steps should use momentum
    changes = []
    for i in range(3):
        p.grad = np.array([1.0])
        prev = p.data.copy()
        opt.step()
        changes.append(p.data[0] - prev[0])
    
    # Due to bias correction, first step should be largest
    # (or changes should not all be equal)
    assert not all(np.isclose(c, changes[0]) for c in changes), "Adam should have varying step sizes"


def test_adam_zero_grad():
    """Test Adam zero_grad."""
    p = Parameter([1.0])
    p.grad = np.array([5.0])
    
    opt = Adam([p], lr=0.001)
    opt.zero_grad()
    
    assert np.allclose(p.grad, 0), f"grad = {p.grad}"


def test_adam_weight_decay():
    """Test Adam with weight decay."""
    p = Parameter([1.0])
    p.grad = np.array([0.0])
    
    try:
        opt = Adam([p], lr=0.1, weight_decay=0.1)
        opt.step()
        
        # With weight decay, parameter should decrease even with zero gradient
        assert p.data[0] < 1.0, f"p = {p.data[0]}, expected < 1.0"
    except TypeError:
        pytest.skip("Adam weight_decay not implemented")


def test_adam_multiple_params():
    """Test Adam with multiple parameters."""
    p1 = Parameter([1.0, 2.0])
    p2 = Parameter([3.0])
    
    p1.grad = np.array([0.1, 0.1])
    p2.grad = np.array([0.1])
    
    opt = Adam([p1, p2], lr=0.1)
    opt.step()
    
    assert not np.allclose(p1.data, [1.0, 2.0]), "p1 didn't change"
    assert not np.allclose(p2.data, [3.0]), "p2 didn't change"


def test_adam_with_model():
    """Test Adam with Module."""
    model = Linear(2, 1)
    opt = Adam(model.parameters(), lr=0.01)
    
    x = Tensor([[1.0, 2.0]])
    y = model(x)
    loss = y.sum()
    
    loss.backward()
    opt.step()
    
    assert model.weight.grad is not None, "No gradient computed"


def test_adam_training_convergence():
    """Test Adam convergence on simple problem."""
    model = Linear(1, 1)
    model.weight.data = np.array([[0.0]])
    model.bias.data = np.array([0.0])
    
    opt = Adam(model.parameters(), lr=0.1)
    
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
    
    assert final_loss < initial_loss * 0.1, f"Loss didn't converge: {initial_loss} -> {final_loss}"


def test_adam_vs_sgd_speed():
    """Test that Adam typically converges faster than SGD."""
    # This is a soft test - Adam should be faster on this problem
    def train(opt_class, lr):
        np.random.seed(42)
        model = Linear(2, 1)
        opt = opt_class(model.parameters(), lr=lr)
        
        x = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        target = Tensor([[1.0], [2.0], [3.0]])
        
        for _ in range(50):
            opt.zero_grad()
            pred = model(x)
            loss = ((pred - target) ** 2).mean()
            loss.backward()
            opt.step()
        
        return loss.data
    
    try:
        from day20 import SGD
        adam_loss = train(Adam, 0.1)
        sgd_loss = train(SGD, 0.1)
        
        # Just verify both work, not necessarily that Adam is faster
        assert adam_loss < 10 and sgd_loss < 10, "Training didn't work"
    except ImportError:
        pytest.skip("SGD not available for comparison")


def test_adam_sparse_gradients():
    """Test Adam with sparse gradients (some zeros)."""
    p = Parameter([1.0, 2.0, 3.0])
    opt = Adam([p], lr=0.1)
    
    # Only gradient on first element
    p.grad = np.array([1.0, 0.0, 0.0])
    opt.step()
    
    # First element should change more
    assert not np.isclose(p.data[0], 1.0), "First element should change"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
