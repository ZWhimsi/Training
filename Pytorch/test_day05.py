"""Test Suite for Day 5: Activation Functions"""

import torch
import torch.nn.functional as F
from typing import Tuple

try:
    from day05 import (relu_manual, leaky_relu_manual, gelu_manual,
                       sigmoid_manual, hard_sigmoid, softmax_manual,
                       log_softmax_manual, silu_manual, Mish)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_relu() -> Tuple[bool, str]:
    try:
        x = torch.randn(4, 8)
        result = relu_manual(x)
        
        if result is None:
            return False, "Not implemented"
        
        expected = F.relu(x)
        max_err = (result - expected).abs().max().item()
        
        if max_err > 1e-6:
            return False, f"Error: {max_err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_leaky_relu() -> Tuple[bool, str]:
    try:
        x = torch.randn(4, 8)
        result = leaky_relu_manual(x, 0.01)
        
        if result is None:
            return False, "Not implemented"
        
        expected = F.leaky_relu(x, 0.01)
        max_err = (result - expected).abs().max().item()
        
        if max_err > 1e-6:
            return False, f"Error: {max_err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_gelu() -> Tuple[bool, str]:
    try:
        x = torch.randn(4, 8)
        result = gelu_manual(x)
        
        if result is None:
            return False, "Not implemented"
        
        expected = F.gelu(x, approximate='tanh')
        max_err = (result - expected).abs().max().item()
        
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_sigmoid() -> Tuple[bool, str]:
    try:
        x = torch.randn(4, 8)
        result = sigmoid_manual(x)
        
        if result is None:
            return False, "Not implemented"
        
        expected = torch.sigmoid(x)
        max_err = (result - expected).abs().max().item()
        
        if max_err > 1e-6:
            return False, f"Error: {max_err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_softmax() -> Tuple[bool, str]:
    try:
        x = torch.randn(4, 8)
        result = softmax_manual(x, dim=-1)
        
        if result is None:
            return False, "Not implemented"
        
        expected = F.softmax(x, dim=-1)
        max_err = (result - expected).abs().max().item()
        
        # Check sums to 1
        row_sums = result.sum(dim=-1)
        sum_err = (row_sums - 1.0).abs().max().item()
        
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        if sum_err > 1e-5:
            return False, f"Sum error: {sum_err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_log_softmax() -> Tuple[bool, str]:
    try:
        x = torch.randn(4, 8)
        result = log_softmax_manual(x, dim=-1)
        
        if result is None:
            return False, "Not implemented"
        
        expected = F.log_softmax(x, dim=-1)
        max_err = (result - expected).abs().max().item()
        
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_silu() -> Tuple[bool, str]:
    try:
        x = torch.randn(4, 8)
        result = silu_manual(x)
        
        if result is None:
            return False, "Not implemented"
        
        expected = F.silu(x)
        max_err = (result - expected).abs().max().item()
        
        if max_err > 1e-6:
            return False, f"Error: {max_err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_mish() -> Tuple[bool, str]:
    try:
        x = torch.randn(4, 8)
        mish = Mish()
        result = mish(x)
        
        if result is None:
            return False, "Not implemented"
        
        expected = F.mish(x)
        max_err = (result - expected).abs().max().item()
        
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("relu", test_relu),
        ("leaky_relu", test_leaky_relu),
        ("gelu", test_gelu),
        ("sigmoid", test_sigmoid),
        ("softmax", test_softmax),
        ("log_softmax", test_log_softmax),
        ("silu", test_silu),
        ("mish", test_mish),
    ]
    
    print(f"\n{'='*50}\nDay 5: Activation Functions - Tests\n{'='*50}")
    
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        return
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    run_all_tests()
