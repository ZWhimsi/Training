"""Test Suite for Day 15: Layer Normalization"""

import torch
import torch.nn.functional as F
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day15 import layer_norm, rms_norm, layer_norm_residual
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_layer_norm_basic() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, hidden = 32, 128
        x = torch.randn(batch, hidden, device='cuda')
        weight = torch.ones(hidden, device='cuda')
        bias = torch.zeros(hidden, device='cuda')
        
        result = layer_norm(x, weight, bias)
        expected = F.layer_norm(x, [hidden], weight, bias)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "LayerNorm basic OK"
    except Exception as e:
        return False, str(e)


def test_layer_norm_affine() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, hidden = 32, 128
        x = torch.randn(batch, hidden, device='cuda')
        weight = torch.randn(hidden, device='cuda')
        bias = torch.randn(hidden, device='cuda')
        
        result = layer_norm(x, weight, bias)
        expected = F.layer_norm(x, [hidden], weight, bias)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "LayerNorm with affine OK"
    except Exception as e:
        return False, str(e)


def test_rms_norm() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, hidden = 32, 128
        x = torch.randn(batch, hidden, device='cuda')
        weight = torch.ones(hidden, device='cuda')
        eps = 1e-5
        
        result = rms_norm(x, weight, eps)
        
        # Manual RMS norm reference
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        expected = x / rms * weight
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "RMSNorm OK"
    except Exception as e:
        return False, str(e)


def test_layer_norm_residual() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, hidden = 32, 128
        x = torch.randn(batch, hidden, device='cuda')
        residual = torch.randn(batch, hidden, device='cuda')
        weight = torch.ones(hidden, device='cuda')
        bias = torch.zeros(hidden, device='cuda')
        
        result = layer_norm_residual(x, residual, weight, bias)
        expected = F.layer_norm(x + residual, [hidden], weight, bias)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "LayerNorm+residual OK"
    except Exception as e:
        return False, str(e)


def test_different_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for batch, hidden in [(8, 64), (64, 256), (128, 512)]:
            x = torch.randn(batch, hidden, device='cuda')
            weight = torch.ones(hidden, device='cuda')
            bias = torch.zeros(hidden, device='cuda')
            
            result = layer_norm(x, weight, bias)
            expected = F.layer_norm(x, [hidden], weight, bias)
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-3:
                return False, f"Failed at {batch}x{hidden}"
        
        return True, "Various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("layer_norm_basic", test_layer_norm_basic),
        ("layer_norm_affine", test_layer_norm_affine),
        ("rms_norm", test_rms_norm),
        ("layer_norm_residual", test_layer_norm_residual),
        ("different_sizes", test_different_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 15: Layer Normalization - Tests\n{'='*50}")
    
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
