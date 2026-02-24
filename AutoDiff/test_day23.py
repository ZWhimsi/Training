"""Test Suite for Day 23: Training Loop"""

import numpy as np
import pytest

from day23 import (
    Tensor, Linear, ReLU, Sequential, MSELoss,
    SGD, DataLoader, train_epoch, evaluate
)


def test_train_epoch_basic():
    """Test basic training epoch."""
    model = Linear(2, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    
    X = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    y = Tensor([[1.0], [2.0], [3.0], [4.0]])
    
    loader = DataLoader(X, y, batch_size=2)
    
    loss = train_epoch(model, loader, loss_fn, optimizer)
    
    assert loss is not None, "train_epoch returned None"
    assert loss > 0, f"Loss should be positive, got {loss}"


def test_train_epoch_decreases_loss():
    """Test that training decreases loss."""
    np.random.seed(42)
    model = Linear(2, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    
    X = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    y = Tensor([[1.0], [2.0], [3.0], [0.0]])
    
    loader = DataLoader(X, y, batch_size=4, shuffle=False)
    
    loss1 = train_epoch(model, loader, loss_fn, optimizer)
    loss2 = train_epoch(model, loader, loss_fn, optimizer)
    loss3 = train_epoch(model, loader, loss_fn, optimizer)
    
    # Loss should generally decrease
    assert loss3 < loss1, f"Loss should decrease: {loss1} -> {loss3}"


def test_evaluate_basic():
    """Test basic evaluation."""
    model = Linear(2, 1)
    model.weight.data = np.array([[1.0, 1.0]])
    model.bias.data = np.array([0.0])
    
    loss_fn = MSELoss()
    
    X = Tensor([[1.0, 1.0], [2.0, 2.0]])  # Outputs: [2, 4]
    y = Tensor([[2.0], [4.0]])  # Perfect match
    
    loader = DataLoader(X, y, batch_size=2, shuffle=False)
    
    loss = evaluate(model, loader, loss_fn)
    
    assert loss is not None, "evaluate returned None"
    assert np.isclose(loss, 0.0, atol=1e-5), f"Loss = {loss}, expected ~0"


def test_evaluate_no_gradient():
    """Test that evaluate doesn't compute gradients."""
    model = Linear(2, 1)
    loss_fn = MSELoss()
    
    X = Tensor([[1.0, 2.0]])
    y = Tensor([[1.0]])
    
    loader = DataLoader(X, y, batch_size=1)
    
    initial_grad = model.weight.grad.copy() if model.weight.grad is not None else None
    
    evaluate(model, loader, loss_fn)
    
    # Gradient should not have changed (or should be zeros)
    if initial_grad is not None:
        assert np.allclose(model.weight.grad, initial_grad), "Gradient changed during eval"


def test_train_mlp():
    """Test training a multi-layer network."""
    np.random.seed(42)
    
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1)
    )
    
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # XOR-like problem
    X = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = Tensor([[0.0], [1.0], [1.0], [0.0]])
    
    loader = DataLoader(X, y, batch_size=4, shuffle=False)
    
    initial_loss = train_epoch(model, loader, loss_fn, optimizer)
    
    for _ in range(100):
        train_epoch(model, loader, loss_fn, optimizer)
    
    final_loss = evaluate(model, loader, loss_fn)
    
    assert final_loss < initial_loss, f"Loss should decrease: {initial_loss} -> {final_loss}"


def test_training_loop_full():
    """Test full training loop."""
    np.random.seed(42)
    
    model = Linear(1, 1)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    
    # y = 2x + 1
    X_train = Tensor(np.linspace(0, 1, 20).reshape(-1, 1))
    y_train = Tensor((2 * np.linspace(0, 1, 20) + 1).reshape(-1, 1))
    
    train_loader = DataLoader(X_train, y_train, batch_size=5)
    
    losses = []
    for epoch in range(50):
        loss = train_epoch(model, train_loader, loss_fn, optimizer)
        losses.append(loss)
    
    # Loss should decrease over training
    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]} -> {losses[-1]}"
    
    # Model should approximate y = 2x + 1
    assert np.isclose(model.weight.data[0, 0], 2.0, atol=0.5), f"weight = {model.weight.data}"
    assert np.isclose(model.bias.data[0], 1.0, atol=0.5), f"bias = {model.bias.data}"


def test_batch_processing():
    """Test that batches are processed correctly."""
    model = Linear(2, 1)
    model.weight.data = np.array([[1.0, 1.0]])
    model.bias.data = np.array([0.0])
    
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.0)  # No updates
    
    X = Tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    y = Tensor([[2.0], [4.0], [6.0], [8.0]])
    
    loader = DataLoader(X, y, batch_size=2, shuffle=False)
    
    # With lr=0, loss should be 0 (perfect predictions)
    loss = evaluate(model, loader, loss_fn)
    assert np.isclose(loss, 0.0, atol=1e-5), f"Loss = {loss}, expected 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
