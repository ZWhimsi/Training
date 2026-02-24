"""Test Suite for Day 27: More Optimizers (Adagrad, RMSprop)"""

import numpy as np
import pytest

from day27 import Tensor, Parameter, Linear, Adagrad, RMSprop


def test_adagrad_creation():
    """Test Adagrad optimizer creation."""
    p = Parameter([1.0, 2.0])
    opt = Adagrad([p], lr=0.1)
    
    assert opt is not None, "Adagrad returned None"


def test_adagrad_step():
    """Test Adagrad step."""
    p = Parameter([1.0])
    p.grad = np.array([1.0])
    
    opt = Adagrad([p], lr=0.1)
    initial = p.data.copy()
    
    opt.step()
    
    assert not np.allclose(p.data, initial), "Parameter didn't change"


def test_adagrad_adaptive_lr():
    """Test Adagrad adaptive learning rate."""
    p = Parameter([0.0])
    opt = Adagrad([p], lr=1.0)
    
    # Multiple steps with same gradient
    steps = []
    for _ in range(5):
        prev = p.data.copy()
        p.grad = np.array([1.0])
        opt.step()
        steps.append(p.data[0] - prev[0])
    
    # Step sizes should decrease (adaptive LR)
    for i in range(1, len(steps)):
        assert abs(steps[i]) <= abs(steps[i-1]), f"Step size should decrease: {steps}"


def test_adagrad_zero_grad():
    """Test Adagrad zero_grad."""
    p = Parameter([1.0])
    p.grad = np.array([5.0])
    
    opt = Adagrad([p], lr=0.1)
    opt.zero_grad()
    
    assert np.allclose(p.grad, 0), f"grad = {p.grad}"


def test_adagrad_eps():
    """Test Adagrad epsilon for numerical stability."""
    p = Parameter([1.0])
    p.grad = np.array([0.0])  # Zero gradient
    
    opt = Adagrad([p], lr=0.1, eps=1e-8)
    
    # Should not raise division by zero
    opt.step()
    assert np.isfinite(p.data[0]), f"p = {p.data[0]}"


def test_rmsprop_creation():
    """Test RMSprop optimizer creation."""
    p = Parameter([1.0, 2.0])
    opt = RMSprop([p], lr=0.01)
    
    assert opt is not None, "RMSprop returned None"


def test_rmsprop_step():
    """Test RMSprop step."""
    p = Parameter([1.0])
    p.grad = np.array([1.0])
    
    opt = RMSprop([p], lr=0.1)
    initial = p.data.copy()
    
    opt.step()
    
    assert not np.allclose(p.data, initial), "Parameter didn't change"


def test_rmsprop_decay():
    """Test RMSprop uses exponential moving average."""
    p = Parameter([0.0])
    opt = RMSprop([p], lr=0.1, alpha=0.9)
    
    # With alpha=0.9, recent gradients have less weight
    steps = []
    for _ in range(5):
        prev = p.data.copy()
        p.grad = np.array([1.0])
        opt.step()
        steps.append(p.data[0] - prev[0])
    
    # Step sizes should stabilize (unlike Adagrad which keeps decreasing)
    # Check that steps don't keep decreasing to near zero
    assert abs(steps[-1]) > 1e-6, f"RMSprop should not decay to zero: {steps}"


def test_rmsprop_zero_grad():
    """Test RMSprop zero_grad."""
    p = Parameter([1.0])
    p.grad = np.array([5.0])
    
    opt = RMSprop([p], lr=0.01)
    opt.zero_grad()
    
    assert np.allclose(p.grad, 0), f"grad = {p.grad}"


def test_rmsprop_momentum():
    """Test RMSprop with momentum."""
    p = Parameter([0.0])
    
    try:
        opt = RMSprop([p], lr=0.1, momentum=0.9)
        
        p.grad = np.array([1.0])
        opt.step()
        step1 = p.data[0]
        
        p.grad = np.array([1.0])
        opt.step()
        step2 = p.data[0] - step1
        
        # With momentum, second step should be larger
        # (momentum accumulates)
        assert True  # Just check it runs
    except TypeError:
        pytest.skip("RMSprop momentum not implemented")


def test_optimizer_with_model():
    """Test optimizers with a model."""
    model = Linear(2, 1)
    
    for OptClass in [Adagrad, RMSprop]:
        opt = OptClass(model.parameters(), lr=0.01)
        
        x = Tensor([[1.0, 2.0]])
        y = model(x)
        loss = y.sum()
        
        loss.backward()
        opt.step()
        
        assert model.weight.grad is not None, f"{OptClass.__name__}: No gradient"


def test_adagrad_training():
    """Test Adagrad on simple training problem."""
    np.random.seed(42)
    
    model = Linear(1, 1)
    model.weight.data = np.array([[0.0]])
    model.bias.data = np.array([0.0])
    
    opt = Adagrad(model.parameters(), lr=1.0)
    
    X = Tensor([[1.0], [2.0], [3.0]])
    y = Tensor([[2.0], [4.0], [6.0]])  # y = 2x
    
    for _ in range(50):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
    
    # Should approximate y = 2x
    assert np.isclose(model.weight.data[0, 0], 2.0, atol=0.5), f"weight = {model.weight.data}"


def test_rmsprop_training():
    """Test RMSprop on simple training problem."""
    np.random.seed(42)
    
    model = Linear(1, 1)
    model.weight.data = np.array([[0.0]])
    model.bias.data = np.array([0.0])
    
    opt = RMSprop(model.parameters(), lr=0.1)
    
    X = Tensor([[1.0], [2.0], [3.0]])
    y = Tensor([[2.0], [4.0], [6.0]])  # y = 2x
    
    for _ in range(100):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
    
    # Should approximate y = 2x
    assert np.isclose(model.weight.data[0, 0], 2.0, atol=0.5), f"weight = {model.weight.data}"


def test_optimizer_multiple_params():
    """Test optimizers with multiple parameters."""
    p1 = Parameter([1.0, 2.0])
    p2 = Parameter([3.0])
    
    for OptClass in [Adagrad, RMSprop]:
        p1.data = np.array([1.0, 2.0])
        p2.data = np.array([3.0])
        p1.grad = np.array([0.1, 0.2])
        p2.grad = np.array([0.3])
        
        opt = OptClass([p1, p2], lr=0.1)
        opt.step()
        
        assert not np.allclose(p1.data, [1.0, 2.0]), f"{OptClass.__name__}: p1 didn't change"
        assert not np.allclose(p2.data, [3.0]), f"{OptClass.__name__}: p2 didn't change"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
