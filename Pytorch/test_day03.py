"""Test Suite for Day 3: Autograd"""

import torch
import pytest

try:
    from day03 import (compute_gradient_simple, compute_chain_rule, 
                       compute_multi_variable, gradient_accumulation,
                       control_gradient_flow, tensor_gradients)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_gradient_simple():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = compute_gradient_simple()
    
    assert result['gradient'] is not None, "gradient is None"
    
    expected = 6.0
    assert abs(result['gradient'].item() - expected) <= 1e-5, f"Expected {expected}, got {result['gradient'].item()}"


def test_chain_rule():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = compute_chain_rule()
    
    assert result['gradient'] is not None, "gradient is None"
    
    expected = 20.0
    assert abs(result['gradient'].item() - expected) <= 1e-5, f"Expected {expected}, got {result['gradient'].item()}"


def test_multi_variable():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = compute_multi_variable()
    
    assert result['grad_x'] is not None and result['grad_y'] is not None, "Gradients are None"
    
    expected_x = 12.0
    expected_y = 31.0
    
    assert abs(result['grad_x'].item() - expected_x) <= 1e-5, f"grad_x: expected {expected_x}, got {result['grad_x'].item()}"
    assert abs(result['grad_y'].item() - expected_y) <= 1e-5, f"grad_y: expected {expected_y}, got {result['grad_y'].item()}"


def test_gradient_accumulation():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = gradient_accumulation()
    
    assert result['grad_after_first'] == 4.0, f"After first: {result['grad_after_first']}, expected 4"
    assert result['grad_after_second'] is not None and result['grad_after_second'] == 8.0, f"After second: {result['grad_after_second']}, expected 8 (accumulated)"
    assert result['grad_after_zero'] is not None and result['grad_after_zero'] == 4.0, f"After zero: {result['grad_after_zero']}, expected 4"


def test_control_gradient_flow():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = control_gradient_flow()
    
    assert result['detached_requires_grad'] is not None, "detached_requires_grad is None"
    assert result['detached_requires_grad'] == False, "Detached should not require grad"
    assert result['no_grad_requires'] == False, "Inside no_grad should not require grad"
    assert result['after_no_grad_requires'] is not None and result['after_no_grad_requires'] == True, "After no_grad should require grad again"


def test_tensor_gradients():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = tensor_gradients()
    
    assert result['gradient'] is not None, "gradient is None"
    
    expected = torch.tensor([[2., 4.], [6., 8.]])
    assert torch.allclose(result['gradient'], expected), f"Expected {expected}, got {result['gradient']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
