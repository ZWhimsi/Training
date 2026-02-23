"""Test Suite for Day 3: Autograd"""

import torch
import sys
from typing import Tuple

try:
    from day03 import (compute_gradient_simple, compute_chain_rule, 
                       compute_multi_variable, gradient_accumulation,
                       control_gradient_flow, tensor_gradients)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_gradient_simple() -> Tuple[bool, str]:
    try:
        result = compute_gradient_simple()
        
        if result['gradient'] is None:
            return False, "gradient is None"
        
        expected = 6.0  # d/dx(x^2) = 2x = 6 at x=3
        if abs(result['gradient'].item() - expected) > 1e-5:
            return False, f"Expected {expected}, got {result['gradient'].item()}"
        
        return True, f"d/dx(x^2)|_3 = {result['gradient'].item()}"
    except Exception as e:
        return False, str(e)


def test_chain_rule() -> Tuple[bool, str]:
    try:
        result = compute_chain_rule()
        
        if result['gradient'] is None:
            return False, "gradient is None"
        
        expected = 20.0  # 4 * (2*2 + 1) = 20
        if abs(result['gradient'].item() - expected) > 1e-5:
            return False, f"Expected {expected}, got {result['gradient'].item()}"
        
        return True, f"d/dx((2x+1)^2)|_2 = {result['gradient'].item()}"
    except Exception as e:
        return False, str(e)


def test_multi_variable() -> Tuple[bool, str]:
    try:
        result = compute_multi_variable()
        
        if result['grad_x'] is None or result['grad_y'] is None:
            return False, "Gradients are None"
        
        expected_x = 12.0  # 2xy at (2,3)
        expected_y = 31.0  # x^2 + 3y^2 at (2,3)
        
        if abs(result['grad_x'].item() - expected_x) > 1e-5:
            return False, f"grad_x: expected {expected_x}, got {result['grad_x'].item()}"
        if abs(result['grad_y'].item() - expected_y) > 1e-5:
            return False, f"grad_y: expected {expected_y}, got {result['grad_y'].item()}"
        
        return True, f"df/dx={result['grad_x'].item()}, df/dy={result['grad_y'].item()}"
    except Exception as e:
        return False, str(e)


def test_gradient_accumulation() -> Tuple[bool, str]:
    try:
        result = gradient_accumulation()
        
        if result['grad_after_first'] != 4.0:
            return False, f"After first: {result['grad_after_first']}, expected 4"
        
        if result['grad_after_second'] is None or result['grad_after_second'] != 8.0:
            return False, f"After second: {result['grad_after_second']}, expected 8 (accumulated)"
        
        if result['grad_after_zero'] is None or result['grad_after_zero'] != 4.0:
            return False, f"After zero: {result['grad_after_zero']}, expected 4"
        
        return True, "Accumulation: 4 → 8, then reset → 4"
    except Exception as e:
        return False, str(e)


def test_control_gradient_flow() -> Tuple[bool, str]:
    try:
        result = control_gradient_flow()
        
        if result['detached_requires_grad'] is None:
            return False, "detached_requires_grad is None"
        if result['detached_requires_grad'] != False:
            return False, "Detached should not require grad"
        
        if result['no_grad_requires'] != False:
            return False, "Inside no_grad should not require grad"
        
        if result['after_no_grad_requires'] is None or result['after_no_grad_requires'] != True:
            return False, "After no_grad should require grad again"
        
        return True, "detach and no_grad work correctly"
    except Exception as e:
        return False, str(e)


def test_tensor_gradients() -> Tuple[bool, str]:
    try:
        result = tensor_gradients()
        
        if result['gradient'] is None:
            return False, "gradient is None"
        
        # Gradient of sum(x^2) is 2x
        expected = torch.tensor([[2., 4.], [6., 8.]])
        if not torch.allclose(result['gradient'], expected):
            return False, f"Expected {expected}, got {result['gradient']}"
        
        return True, "Tensor gradient: 2x"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("gradient_simple", test_gradient_simple),
        ("chain_rule", test_chain_rule),
        ("multi_variable", test_multi_variable),
        ("gradient_accumulation", test_gradient_accumulation),
        ("control_gradient_flow", test_control_gradient_flow),
        ("tensor_gradients", test_tensor_gradients),
    ]
    
    print(f"\n{'='*50}\nDay 3: Autograd - Tests\n{'='*50}")
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
