"""Test Suite for Day 4: Neural Network Building Blocks"""

import torch
import torch.nn as nn
import pytest

try:
    from day04 import (SimpleLayer, TwoLayerNet, count_parameters,
                       get_parameter_info, create_sequential_net, test_forward_pass)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_simple_layer():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    layer = SimpleLayer(10, 5)
    
    assert layer.weight is not None, "weight is None"
    assert layer.bias is not None, "bias is None"
    
    assert layer.weight.shape == (5, 10), f"Weight shape wrong: {layer.weight.shape}, expected (5, 10)"
    assert layer.bias.shape == (5,), f"Bias shape wrong: {layer.bias.shape}, expected (5,)"
    
    # Test forward pass with known values
    torch.manual_seed(42)
    x = torch.randn(4, 10)
    out = layer(x)
    
    assert out is not None, "forward returned None"
    assert out.shape == (4, 5), f"Output shape wrong: {out.shape}"
    
    # Verify computation: y = x @ W.T + b
    expected = x @ layer.weight.T + layer.bias
    assert torch.allclose(out, expected, atol=1e-5), f"Forward computation wrong: got {out[0]}, expected {expected[0]}"


def test_two_layer_net():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = TwoLayerNet(10, 20, 5)
    
    assert model.fc1 is not None and model.fc2 is not None, "Layers are None"
    
    # Verify layer shapes
    assert model.fc1.weight.shape == (20, 10), f"fc1 weight shape: {model.fc1.weight.shape}, expected (20, 10)"
    assert model.fc2.weight.shape == (5, 20), f"fc2 weight shape: {model.fc2.weight.shape}, expected (5, 20)"
    
    torch.manual_seed(42)
    x = torch.randn(4, 10)
    out = model(x)
    
    assert out is not None, "forward returned None"
    assert out.shape == (4, 5), f"Output shape: {out.shape}, expected (4, 5)"
    
    # Verify computation: relu(x @ W1.T + b1) @ W2.T + b2
    expected = torch.relu(model.fc1(x))
    expected = model.fc2(expected)
    assert torch.allclose(out, expected, atol=1e-5), "Forward computation wrong"


def test_count_parameters():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = nn.Linear(10, 5)
    count = count_parameters(model)
    
    assert count is not None, "Returned None"
    
    # 10*5 weights + 5 bias = 55
    expected = 55
    assert count == expected, f"Expected {expected}, got {count}"


def test_parameter_info():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = nn.Linear(10, 5)
    info = get_parameter_info(model)
    
    assert len(info) == 2, f"Expected 2 params, got {len(info)}"
    
    weight_info = [i for i in info if 'weight' in i['name']][0]
    bias_info = [i for i in info if 'bias' in i['name']][0]
    
    # Validate weight shape
    assert weight_info['shape'] is not None, "weight shape is None"
    assert weight_info['shape'] == (5, 10), f"weight shape: got {weight_info['shape']}, expected (5, 10)"
    
    # Validate bias shape
    assert bias_info['shape'] is not None, "bias shape is None"
    assert bias_info['shape'] == (5,), f"bias shape: got {bias_info['shape']}, expected (5,)"
    
    # Validate requires_grad
    assert weight_info['requires_grad'] == True, f"weight requires_grad: got {weight_info['requires_grad']}, expected True"
    assert bias_info['requires_grad'] == True, f"bias requires_grad: got {bias_info['requires_grad']}, expected True"


def test_sequential():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = create_sequential_net(10, 20, 5)
    
    assert model is not None, "Returned None"
    
    x = torch.randn(4, 10)
    out = model(x)
    
    assert out.shape == (4, 5), f"Output shape: {out.shape}"


def test_forward_pass_fn():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = test_forward_pass()
    
    assert 'error' not in result, result.get('error', '')
    
    assert result['output_shape'] == (4, 5), f"Output shape: {result['output_shape']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
