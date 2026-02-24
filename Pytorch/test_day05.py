"""Test Suite for Day 5: Activation Functions"""

import torch
import torch.nn.functional as F
import pytest

try:
    from day05 import (relu_manual, leaky_relu_manual, gelu_manual,
                       sigmoid_manual, hard_sigmoid, softmax_manual,
                       log_softmax_manual, silu_manual, Mish)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_relu():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    x = torch.randn(4, 8)
    result = relu_manual(x)
    
    assert result is not None, "Not implemented"
    
    expected = F.relu(x)
    max_err = (result - expected).abs().max().item()
    
    assert max_err <= 1e-6, f"Error: {max_err:.6f}"

def test_leaky_relu():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    x = torch.randn(4, 8)
    result = leaky_relu_manual(x, 0.01)
    
    assert result is not None, "Not implemented"
    
    expected = F.leaky_relu(x, 0.01)
    max_err = (result - expected).abs().max().item()
    
    assert max_err <= 1e-6, f"Error: {max_err:.6f}"

def test_gelu():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    x = torch.randn(4, 8)
    result = gelu_manual(x)
    
    assert result is not None, "Not implemented"
    
    expected = F.gelu(x, approximate='tanh')
    max_err = (result - expected).abs().max().item()
    
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"

def test_sigmoid():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    x = torch.randn(4, 8)
    result = sigmoid_manual(x)
    
    assert result is not None, "Not implemented"
    
    expected = torch.sigmoid(x)
    max_err = (result - expected).abs().max().item()
    
    assert max_err <= 1e-6, f"Error: {max_err:.6f}"

def test_softmax():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    x = torch.randn(4, 8)
    result = softmax_manual(x, dim=-1)
    
    assert result is not None, "Not implemented"
    
    expected = F.softmax(x, dim=-1)
    max_err = (result - expected).abs().max().item()
    
    row_sums = result.sum(dim=-1)
    sum_err = (row_sums - 1.0).abs().max().item()
    
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"
    assert sum_err <= 1e-5, f"Sum error: {sum_err:.6f}"

def test_log_softmax():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    x = torch.randn(4, 8)
    result = log_softmax_manual(x, dim=-1)
    
    assert result is not None, "Not implemented"
    
    expected = F.log_softmax(x, dim=-1)
    max_err = (result - expected).abs().max().item()
    
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"

def test_silu():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    x = torch.randn(4, 8)
    result = silu_manual(x)
    
    assert result is not None, "Not implemented"
    
    expected = F.silu(x)
    max_err = (result - expected).abs().max().item()
    
    assert max_err <= 1e-6, f"Error: {max_err:.6f}"

def test_mish():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    x = torch.randn(4, 8)
    mish = Mish()
    result = mish(x)
    
    assert result is not None, "Not implemented"
    
    expected = F.mish(x)
    max_err = (result - expected).abs().max().item()
    
    assert max_err <= 1e-5, f"Error: {max_err:.6f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
