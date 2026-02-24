"""Test Suite for Day 4: Neural Network Building Blocks"""

import torch
import torch.nn as nn
import sys
from typing import Tuple

try:
    from day04 import (SimpleLayer, TwoLayerNet, count_parameters,
                       get_parameter_info, create_sequential_net, test_forward_pass)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_simple_layer() -> Tuple[bool, str]:
    try:
        layer = SimpleLayer(10, 5)
        
        if layer.weight is None:
            return False, "weight is None"
        if layer.bias is None:
            return False, "bias is None"
        
        if layer.weight.shape != (5, 10):
            return False, f"Weight shape wrong: {layer.weight.shape}, expected (5, 10)"
        if layer.bias.shape != (5,):
            return False, f"Bias shape wrong: {layer.bias.shape}, expected (5,)"
        
        # Test forward pass with known values
        torch.manual_seed(42)
        x = torch.randn(4, 10)
        out = layer(x)
        
        if out is None:
            return False, "forward returned None"
        if out.shape != (4, 5):
            return False, f"Output shape wrong: {out.shape}"
        
        # Verify computation: y = x @ W.T + b
        expected = x @ layer.weight.T + layer.bias
        if not torch.allclose(out, expected, atol=1e-5):
            return False, f"Forward computation wrong: got {out[0]}, expected {expected[0]}"
        
        return True, "SimpleLayer works"
    except Exception as e:
        return False, str(e)


def test_two_layer_net() -> Tuple[bool, str]:
    try:
        model = TwoLayerNet(10, 20, 5)
        
        if model.fc1 is None or model.fc2 is None:
            return False, "Layers are None"
        
        # Verify layer shapes
        if model.fc1.weight.shape != (20, 10):
            return False, f"fc1 weight shape: {model.fc1.weight.shape}, expected (20, 10)"
        if model.fc2.weight.shape != (5, 20):
            return False, f"fc2 weight shape: {model.fc2.weight.shape}, expected (5, 20)"
        
        torch.manual_seed(42)
        x = torch.randn(4, 10)
        out = model(x)
        
        if out is None:
            return False, "forward returned None"
        if out.shape != (4, 5):
            return False, f"Output shape: {out.shape}, expected (4, 5)"
        
        # Verify computation: relu(x @ W1.T + b1) @ W2.T + b2
        expected = torch.relu(model.fc1(x))
        expected = model.fc2(expected)
        if not torch.allclose(out, expected, atol=1e-5):
            return False, f"Forward computation wrong"
        
        return True, "TwoLayerNet forward OK"
    except Exception as e:
        return False, str(e)


def test_count_parameters() -> Tuple[bool, str]:
    try:
        model = nn.Linear(10, 5)
        count = count_parameters(model)
        
        if count is None:
            return False, "Returned None"
        
        # 10*5 weights + 5 bias = 55
        expected = 55
        if count != expected:
            return False, f"Expected {expected}, got {count}"
        
        return True, f"Linear(10,5) has {count} params"
    except Exception as e:
        return False, str(e)


def test_parameter_info() -> Tuple[bool, str]:
    try:
        model = nn.Linear(10, 5)
        info = get_parameter_info(model)
        
        if len(info) != 2:
            return False, f"Expected 2 params, got {len(info)}"
        
        weight_info = [i for i in info if 'weight' in i['name']][0]
        bias_info = [i for i in info if 'bias' in i['name']][0]
        
        # Validate weight shape
        if weight_info['shape'] is None:
            return False, "weight shape is None"
        if weight_info['shape'] != (5, 10):
            return False, f"weight shape: got {weight_info['shape']}, expected (5, 10)"
        
        # Validate bias shape
        if bias_info['shape'] is None:
            return False, "bias shape is None"
        if bias_info['shape'] != (5,):
            return False, f"bias shape: got {bias_info['shape']}, expected (5,)"
        
        # Validate requires_grad
        if weight_info['requires_grad'] != True:
            return False, f"weight requires_grad: got {weight_info['requires_grad']}, expected True"
        if bias_info['requires_grad'] != True:
            return False, f"bias requires_grad: got {bias_info['requires_grad']}, expected True"
        
        return True, "Parameter info correct"
    except Exception as e:
        return False, str(e)


def test_sequential() -> Tuple[bool, str]:
    try:
        model = create_sequential_net(10, 20, 5)
        
        if model is None:
            return False, "Returned None"
        
        x = torch.randn(4, 10)
        out = model(x)
        
        if out.shape != (4, 5):
            return False, f"Output shape: {out.shape}"
        
        return True, "Sequential model works"
    except Exception as e:
        return False, str(e)


def test_forward_pass_fn() -> Tuple[bool, str]:
    try:
        result = test_forward_pass()
        
        if 'error' in result:
            return False, result['error']
        
        if result['output_shape'] != (4, 5):
            return False, f"Output shape: {result['output_shape']}"
        
        return True, f"Forward pass: {result['input_shape']} → {result['output_shape']}"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("simple_layer", test_simple_layer),
        ("two_layer_net", test_two_layer_net),
        ("count_parameters", test_count_parameters),
        ("parameter_info", test_parameter_info),
        ("sequential", test_sequential),
        ("forward_pass", test_forward_pass_fn),
    ]
    
    print(f"\n{'='*50}\nDay 4: NN Building Blocks - Tests\n{'='*50}")
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    run_all_tests()
