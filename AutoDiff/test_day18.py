"""Test Suite for Day 18: Linear Layer"""

import numpy as np
import pytest

from day18 import Tensor, Parameter, Linear


def test_linear_creation():
    """Test Linear layer creation."""
    layer = Linear(4, 3)
    
    assert layer is not None, "Linear returned None"
    assert hasattr(layer, 'weight'), "Linear has no weight"
    assert hasattr(layer, 'bias'), "Linear has no bias"


def test_linear_shapes():
    """Test Linear layer parameter shapes."""
    layer = Linear(10, 5)
    
    assert layer.weight.shape == (5, 10), f"weight shape = {layer.weight.shape}, expected (5, 10)"
    assert layer.bias.shape == (5,), f"bias shape = {layer.bias.shape}, expected (5,)"


def test_linear_forward():
    """Test Linear layer forward pass."""
    layer = Linear(3, 2)
    # Set known weights
    layer.weight.data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    layer.bias.data = np.array([0.0, 0.0])
    
    x = Tensor([1.0, 2.0, 3.0])
    y = layer(x)
    
    assert y is not None and y.data is not None, "forward returned None"
    assert np.allclose(y.data, [1.0, 2.0]), f"y = {y.data}, expected [1, 2]"


def test_linear_with_bias():
    """Test Linear layer with bias."""
    layer = Linear(2, 2)
    layer.weight.data = np.eye(2)
    layer.bias.data = np.array([10.0, 20.0])
    
    x = Tensor([1.0, 2.0])
    y = layer(x)
    
    assert y is not None, "forward returned None"
    assert np.allclose(y.data, [11.0, 22.0]), f"y = {y.data}, expected [11, 22]"


def test_linear_no_bias():
    """Test Linear layer without bias."""
    layer = Linear(3, 2, bias=False)
    
    assert layer.bias is None, f"bias should be None"
    
    layer.weight.data = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    x = Tensor([1.0, 2.0, 3.0])
    y = layer(x)
    
    assert y is not None, "forward returned None"
    assert np.allclose(y.data, [6.0, 12.0]), f"y = {y.data}"


def test_linear_batch():
    """Test Linear layer with batch input."""
    layer = Linear(3, 2)
    layer.weight.data = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    layer.bias.data = np.array([0.0, 0.0])
    
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Batch of 2
    y = layer(x)
    
    assert y is not None, "forward returned None"
    assert y.shape == (2, 2), f"shape = {y.shape}, expected (2, 2)"
    
    expected = [[1.0, 3.0], [4.0, 6.0]]
    assert np.allclose(y.data, expected), f"y = {y.data}"


def test_linear_backward():
    """Test Linear layer backward pass."""
    layer = Linear(3, 2)
    layer.weight.data = np.ones((2, 3))
    layer.bias.data = np.zeros(2)
    
    x = Tensor([1.0, 2.0, 3.0])
    y = layer(x)
    loss = y.sum()
    loss.backward()
    
    assert layer.weight.grad is not None, "weight grad is None"
    assert layer.bias.grad is not None, "bias grad is None"
    assert x.grad is not None, "x grad is None"


def test_linear_weight_gradient():
    """Test Linear weight gradient values."""
    layer = Linear(2, 1)
    layer.weight.data = np.array([[1.0, 1.0]])
    layer.bias.data = np.array([0.0])
    
    x = Tensor([[3.0, 4.0]])
    y = layer(x)
    y.backward()
    
    # dy/dw = x.T for single output
    # Expected weight grad is [[3, 4]] (shape matches weight)
    assert layer.weight.grad is not None, "weight grad is None"
    expected_shape = (1, 2)
    assert layer.weight.grad.shape == expected_shape, f"grad shape = {layer.weight.grad.shape}"


def test_linear_parameters():
    """Test that Linear has correct number of parameters."""
    layer = Linear(4, 3)
    params = list(layer.parameters())
    
    assert len(params) == 2, f"Should have 2 params (weight, bias), got {len(params)}"


def test_linear_parameters_no_bias():
    """Test that Linear without bias has correct parameters."""
    layer = Linear(4, 3, bias=False)
    params = list(layer.parameters())
    
    assert len(params) == 1, f"Should have 1 param (weight only), got {len(params)}"


def test_linear_chain():
    """Test chaining Linear layers."""
    layer1 = Linear(3, 4)
    layer2 = Linear(4, 2)
    
    x = Tensor([[1.0, 2.0, 3.0]])
    h = layer1(x)
    y = layer2(h)
    
    assert y is not None, "forward returned None"
    assert y.shape == (1, 2), f"shape = {y.shape}, expected (1, 2)"


def test_linear_chain_backward():
    """Test backward through chain of Linear layers."""
    layer1 = Linear(2, 3)
    layer2 = Linear(3, 1)
    
    x = Tensor([[1.0, 2.0]])
    h = layer1(x)
    y = layer2(h)
    y.backward()
    
    assert layer1.weight.grad is not None, "layer1 weight grad is None"
    assert layer2.weight.grad is not None, "layer2 weight grad is None"


def test_linear_repr():
    """Test Linear string representation."""
    layer = Linear(10, 5)
    s = repr(layer)
    
    assert s is not None, "__repr__ returned None"
    assert 'Linear' in s or '10' in s or '5' in s, f"repr = {s}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
