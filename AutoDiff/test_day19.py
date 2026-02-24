"""Test Suite for Day 19: ReLU and Sequential"""

import numpy as np
import pytest

from day19 import Tensor, Linear, ReLU, Sequential


def test_relu_module():
    """Test ReLU as a Module."""
    relu = ReLU()
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = relu(x)
    
    assert y is not None and y.data is not None, "ReLU returned None"
    expected = [0.0, 0.0, 0.0, 1.0, 2.0]
    assert np.allclose(y.data, expected), f"y = {y.data}"


def test_relu_backward():
    """Test ReLU backward pass."""
    relu = ReLU()
    x = Tensor([-1.0, 1.0])
    y = relu(x)
    y.backward()
    
    assert x.grad is not None, "Gradient is None"
    expected = [0.0, 1.0]
    assert np.allclose(x.grad, expected), f"grad = {x.grad}"


def test_relu_no_parameters():
    """Test that ReLU has no parameters."""
    relu = ReLU()
    params = list(relu.parameters())
    
    assert len(params) == 0, f"ReLU should have 0 params, got {len(params)}"


def test_sequential_empty():
    """Test empty Sequential."""
    model = Sequential()
    x = Tensor([1.0, 2.0])
    y = model(x)
    
    assert y is not None, "Sequential returned None"
    assert np.allclose(y.data, x.data), "Empty Sequential should be identity"


def test_sequential_single():
    """Test Sequential with single layer."""
    model = Sequential(ReLU())
    x = Tensor([-1.0, 1.0])
    y = model(x)
    
    assert y is not None, "Sequential returned None"
    expected = [0.0, 1.0]
    assert np.allclose(y.data, expected), f"y = {y.data}"


def test_sequential_linear_relu():
    """Test Sequential with Linear and ReLU."""
    linear = Linear(2, 2)
    linear.weight.data = np.eye(2)
    linear.bias.data = np.array([-1.0, 1.0])
    
    model = Sequential(linear, ReLU())
    x = Tensor([1.0, 0.0])
    y = model(x)
    
    # After Linear: [1-1, 0+1] = [0, 1]
    # After ReLU: [0, 1]
    assert y is not None, "Sequential returned None"
    expected = [0.0, 1.0]
    assert np.allclose(y.data, expected), f"y = {y.data}"


def test_sequential_parameters():
    """Test Sequential parameters collection."""
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2)
    )
    
    params = list(model.parameters())
    # Linear(3,4): weight + bias = 2 params
    # ReLU: 0 params
    # Linear(4,2): weight + bias = 2 params
    # Total: 4 params
    
    assert len(params) == 4, f"Should have 4 params, got {len(params)}"


def test_sequential_backward():
    """Test Sequential backward pass."""
    layer1 = Linear(2, 3)
    layer2 = Linear(3, 1)
    model = Sequential(layer1, ReLU(), layer2)
    
    x = Tensor([[1.0, 2.0]])
    y = model(x)
    y.backward()
    
    assert layer1.weight.grad is not None, "layer1 weight grad is None"
    assert layer2.weight.grad is not None, "layer2 weight grad is None"


def test_sequential_mlp():
    """Test Sequential MLP."""
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 4),
        ReLU(),
        Linear(4, 1)
    )
    
    x = Tensor([[1.0, 2.0]])
    y = model(x)
    
    assert y is not None, "Sequential returned None"
    assert y.shape == (1, 1), f"shape = {y.shape}, expected (1, 1)"


def test_sequential_len():
    """Test Sequential length."""
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 1)
    )
    
    assert len(model) == 3, f"len = {len(model)}, expected 3"


def test_sequential_getitem():
    """Test Sequential indexing."""
    linear1 = Linear(2, 3)
    relu = ReLU()
    linear2 = Linear(3, 1)
    
    model = Sequential(linear1, relu, linear2)
    
    assert model[0] is linear1, "model[0] should be linear1"
    assert model[1] is relu, "model[1] should be relu"
    assert model[2] is linear2, "model[2] should be linear2"


def test_sequential_zero_grad():
    """Test Sequential zero_grad."""
    model = Sequential(Linear(2, 3), Linear(3, 1))
    
    # Run forward/backward to get some gradients
    x = Tensor([[1.0, 2.0]])
    y = model(x)
    y.backward()
    
    model.zero_grad()
    
    for param in model.parameters():
        assert np.allclose(param.grad, 0), f"Grad not zeroed: {param.grad}"


def test_sequential_train_eval():
    """Test Sequential train/eval mode."""
    model = Sequential(Linear(2, 3), ReLU())
    
    model.train()
    assert model.training is True, "Should be in training mode"
    
    model.eval()
    assert model.training is False, "Should be in eval mode"


def test_sequential_repr():
    """Test Sequential string representation."""
    model = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))
    s = repr(model)
    
    assert s is not None, "__repr__ returned None"
    assert 'Sequential' in s or 'Linear' in s, f"repr = {s}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
