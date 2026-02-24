"""Test Suite for Day 28: AdamW Optimizer"""

import numpy as np
import pytest

from day28 import Tensor, Parameter, Linear, AdamW


def test_adamw_creation():
    """Test AdamW optimizer creation."""
    p = Parameter([1.0, 2.0])
    opt = AdamW([p], lr=0.001)
    
    assert opt is not None, "AdamW returned None"


def test_adamw_step():
    """Test AdamW step."""
    p = Parameter([1.0])
    p.grad = np.array([1.0])
    
    opt = AdamW([p], lr=0.1)
    initial = p.data.copy()
    
    opt.step()
    
    assert not np.allclose(p.data, initial), "Parameter didn't change"


def test_adamw_weight_decay():
    """Test AdamW weight decay is decoupled."""
    p = Parameter([1.0])
    p.grad = np.array([0.0])  # Zero gradient
    
    opt = AdamW([p], lr=0.1, weight_decay=0.1)
    opt.step()
    
    # With decoupled weight decay, p should decrease even with zero gradient
    # AdamW: p = p - lr * weight_decay * p
    expected = 1.0 - 0.1 * 0.1 * 1.0  # 0.99
    assert np.isclose(p.data[0], expected, atol=0.02), f"p = {p.data[0]}, expected ~{expected}"


def test_adamw_vs_adam_decay():
    """Test that AdamW applies weight decay differently than Adam."""
    # AdamW: decoupled weight decay (applied directly to weights)
    # Adam: L2 regularization (applied to gradients)
    
    p1 = Parameter([1.0])
    p1.grad = np.array([0.0])
    
    opt = AdamW([p1], lr=0.1, weight_decay=0.1)
    opt.step()
    
    # Weight decay should reduce the weight
    assert p1.data[0] < 1.0, f"AdamW should reduce weight, got {p1.data[0]}"


def test_adamw_default_params():
    """Test AdamW default hyperparameters."""
    p = Parameter([1.0])
    opt = AdamW([p])
    
    pg = opt.param_groups[0]
    assert 'betas' in pg, "No betas parameter"
    assert 'eps' in pg, "No eps parameter"
    assert 'weight_decay' in pg, "No weight_decay parameter"


def test_adamw_zero_grad():
    """Test AdamW zero_grad."""
    p = Parameter([1.0])
    p.grad = np.array([5.0])
    
    opt = AdamW([p], lr=0.001)
    opt.zero_grad()
    
    assert np.allclose(p.grad, 0), f"grad = {p.grad}"


def test_adamw_momentum():
    """Test AdamW maintains momentum terms."""
    p = Parameter([1.0])
    opt = AdamW([p], lr=0.1, betas=(0.9, 0.999))
    
    changes = []
    for _ in range(3):
        p.grad = np.array([1.0])
        prev = p.data.copy()
        opt.step()
        changes.append(p.data[0] - prev[0])
    
    # Due to bias correction and weight decay, steps should vary
    assert not all(np.isclose(c, changes[0]) for c in changes), "AdamW should have varying steps"


def test_adamw_with_model():
    """Test AdamW with Module."""
    model = Linear(2, 1)
    opt = AdamW(model.parameters(), lr=0.01)
    
    x = Tensor([[1.0, 2.0]])
    y = model(x)
    loss = y.sum()
    
    loss.backward()
    opt.step()
    
    assert model.weight.grad is not None, "No gradient computed"


def test_adamw_training_convergence():
    """Test AdamW convergence on simple problem."""
    np.random.seed(42)
    
    model = Linear(1, 1)
    model.weight.data = np.array([[0.0]])
    model.bias.data = np.array([0.0])
    
    opt = AdamW(model.parameters(), lr=0.1, weight_decay=0.01)
    
    X = Tensor([[1.0], [2.0], [3.0]])
    y = Tensor([[2.0], [4.0], [6.0]])  # y = 2x
    
    initial_loss = None
    final_loss = None
    
    for i in range(100):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - y) ** 2).mean()
        
        if i == 0:
            initial_loss = loss.data
        
        loss.backward()
        opt.step()
        
        final_loss = loss.data
    
    assert final_loss < initial_loss * 0.1, f"Loss didn't converge: {initial_loss} -> {final_loss}"


def test_adamw_regularization_effect():
    """Test that AdamW regularization prevents large weights."""
    np.random.seed(42)
    
    # Train with and without weight decay
    def train(weight_decay):
        model = Linear(2, 1)
        opt = AdamW(model.parameters(), lr=0.1, weight_decay=weight_decay)
        
        X = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = Tensor([[100.0], [100.0], [200.0]])  # Large targets
        
        for _ in range(100):
            opt.zero_grad()
            pred = model(X)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            opt.step()
        
        return np.abs(model.weight.data).max()
    
    weight_no_decay = train(0.0)
    weight_with_decay = train(0.1)
    
    # Weight decay should result in smaller weights
    assert weight_with_decay <= weight_no_decay, f"Weight decay should reduce weights: {weight_with_decay} vs {weight_no_decay}"


def test_adamw_multiple_params():
    """Test AdamW with multiple parameters."""
    p1 = Parameter([1.0, 2.0])
    p2 = Parameter([3.0])
    
    p1.grad = np.array([0.1, 0.1])
    p2.grad = np.array([0.1])
    
    opt = AdamW([p1, p2], lr=0.1)
    opt.step()
    
    assert not np.allclose(p1.data, [1.0, 2.0]), "p1 didn't change"
    assert not np.allclose(p2.data, [3.0]), "p2 didn't change"


def test_adamw_sparse_gradients():
    """Test AdamW with sparse gradients."""
    p = Parameter([1.0, 2.0, 3.0])
    opt = AdamW([p], lr=0.1)
    
    # Only gradient on first element
    p.grad = np.array([1.0, 0.0, 0.0])
    opt.step()
    
    # First element should change most due to gradient
    # Other elements still change due to weight decay
    changes = np.abs(p.data - np.array([1.0, 2.0, 3.0]))
    assert changes[0] > changes[1], "First element should change most"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
