"""Test Suite for Day 9: Kernel Fusion"""

import torch
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day09 import (fused_add_mul, fused_bias_relu, fused_scale_shift,
                           fused_residual_add, fused_linear_bias_relu)
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_fused_add_mul() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        y = torch.randn(1024, device='cuda')
        z = torch.randn(1024, device='cuda')
        
        result = fused_add_mul(x, y, z)
        expected = (x + y) * z
        
        if result is None or result.sum() == 0:
            return False, "Output is zero/None"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Max error {max_err:.6f}"
        
        return True, f"(x+y)*z OK, err={max_err:.2e}"
    except Exception as e:
        return False, str(e)


def test_fused_bias_relu() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        bias = torch.randn(1024, device='cuda')
        
        result = fused_bias_relu(x, bias)
        expected = torch.relu(x + bias)
        
        if result is None or result.abs().sum() == 0:
            return False, "Output is zero/None"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Max error {max_err:.6f}"
        
        return True, f"ReLU(x+bias) OK"
    except Exception as e:
        return False, str(e)


def test_fused_scale_shift() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        scale = torch.randn(1024, device='cuda')
        shift = torch.randn(1024, device='cuda')
        
        result = fused_scale_shift(x, scale, shift)
        expected = x * scale + shift
        
        if result is None or result.sum() == 0:
            return False, "Output is zero/None"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Max error {max_err:.6f}"
        
        return True, f"x*scale+shift OK"
    except Exception as e:
        return False, str(e)


def test_fused_residual() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        residual = torch.randn(1024, device='cuda')
        scale = 0.5
        
        result = fused_residual_add(x, residual, scale)
        expected = x * scale + residual
        
        if result is None or result.sum() == 0:
            return False, "Output is zero/None"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Max error {max_err:.6f}"
        
        return True, f"x*scale+residual OK"
    except Exception as e:
        return False, str(e)


def test_fused_linear_bias_relu() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(1024, device='cuda')
        weight = torch.randn(1024, device='cuda')
        bias = torch.randn(1024, device='cuda')
        
        result = fused_linear_bias_relu(x, weight, bias)
        expected = torch.relu(x * weight + bias)
        
        if result is None or result.abs().sum() == 0:
            return False, "Output is zero/None"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Max error {max_err:.6f}"
        
        return True, f"ReLU(x*w+b) OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("fused_add_mul", test_fused_add_mul),
        ("fused_bias_relu", test_fused_bias_relu),
        ("fused_scale_shift", test_fused_scale_shift),
        ("fused_residual", test_fused_residual),
        ("fused_linear_bias_relu", test_fused_linear_bias_relu),
    ]
    
    print(f"\n{'='*50}\nDay 9: Kernel Fusion - Tests\n{'='*50}")
    
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
