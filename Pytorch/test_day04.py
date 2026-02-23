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
            return False, f"Weight shape wrong: {layer.weight.shape}"
        
        x = torch.randn(4, 10)
        out = layer(x)
        
        if out is None:
            return False, "forward returned None"
        if out.shape != (4, 5):
            return False, f"Output shape wrong: {out.shape}"
        
        return True, "SimpleLayer works"
    except Exception as e:
        return False, str(e)


def test_two_layer_net() -> Tuple[bool, str]:
    try:
        model = TwoLayerNet(10, 20, 5)
        
        if model.fc1 is None or model.fc2 is None:
            return False, "Layers are None"
        
        x = torch.randn(4, 10)
        out = model(x)
        
        if out is None:
            return False, "forward returned None"
        if out.shape != (4, 5):
            return False, f"Output shape: {out.shape}, expected (4, 5)"
        
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
        if weight_info['shape'] is None:
            return False, "Shape is None"
        if weight_info['requires_grad'] is None:
            return False, "requires_grad is None"
        
        return True, "Parameter info extracted"
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
