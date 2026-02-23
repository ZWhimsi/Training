"""Test Suite for Day 12: Activation Functions"""

import torch
import torch.nn.functional as F
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day12 import sigmoid, tanh, leaky_relu, elu, softplus, mish
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_sigmoid() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        result = sigmoid(x)
        expected = torch.sigmoid(x)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "sigmoid OK"
    except Exception as e:
        return False, str(e)


def test_tanh() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        result = tanh(x)
        expected = torch.tanh(x)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "tanh OK"
    except Exception as e:
        return False, str(e)


def test_leaky_relu() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        slope = 0.01
        result = leaky_relu(x, slope)
        expected = F.leaky_relu(x, slope)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "leaky_relu OK"
    except Exception as e:
        return False, str(e)


def test_elu() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        alpha = 1.0
        result = elu(x, alpha)
        expected = F.elu(x, alpha)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "elu OK"
    except Exception as e:
        return False, str(e)


def test_softplus() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        x = torch.clamp(x, -10, 10)  # Avoid overflow
        beta = 1.0
        result = softplus(x, beta)
        expected = F.softplus(x, beta)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "softplus OK"
    except Exception as e:
        return False, str(e)


def test_mish() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        x = torch.clamp(x, -10, 10)  # Avoid overflow
        result = mish(x)
        expected = F.mish(x)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "mish OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("sigmoid", test_sigmoid),
        ("tanh", test_tanh),
        ("leaky_relu", test_leaky_relu),
        ("elu", test_elu),
        ("softplus", test_softplus),
        ("mish", test_mish),
    ]
    
    print(f"\n{'='*50}\nDay 12: Activation Functions - Tests\n{'='*50}")
    
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
