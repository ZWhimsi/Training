"""Test Suite for Day 14: Cross-Entropy Loss"""

import torch
import torch.nn.functional as F
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day14 import (log_softmax_rows, cross_entropy_loss, 
                           cross_entropy_mean, cross_entropy_smooth)
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_log_softmax() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(32, 64, device='cuda')
        result = log_softmax_rows(x)
        expected = F.log_softmax(x, dim=-1)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, f"log_softmax OK"
    except Exception as e:
        return False, str(e)


def test_cross_entropy_basic() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch_size, n_classes = 16, 10
        logits = torch.randn(batch_size, n_classes, device='cuda')
        targets = torch.randint(0, n_classes, (batch_size,), device='cuda')
        
        result = cross_entropy_loss(logits, targets)
        expected = F.cross_entropy(logits, targets, reduction='none')
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, f"cross_entropy OK"
    except Exception as e:
        return False, str(e)


def test_cross_entropy_mean() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch_size, n_classes = 16, 10
        logits = torch.randn(batch_size, n_classes, device='cuda')
        targets = torch.randint(0, n_classes, (batch_size,), device='cuda')
        
        result = cross_entropy_mean(logits, targets)
        
        if result is None:
            return False, "Returned None"
        
        expected = F.cross_entropy(logits, targets)
        
        err = abs(result.item() - expected.item())
        if err > 1e-4:
            return False, f"Error: {err:.6f}"
        return True, f"mean loss OK"
    except Exception as e:
        return False, str(e)


def test_label_smoothing() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch_size, n_classes = 16, 10
        logits = torch.randn(batch_size, n_classes, device='cuda')
        targets = torch.randint(0, n_classes, (batch_size,), device='cuda')
        smoothing = 0.1
        
        result = cross_entropy_smooth(logits, targets, smoothing)
        expected = F.cross_entropy(logits, targets, reduction='none', label_smoothing=smoothing)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, f"label_smoothing OK"
    except Exception as e:
        return False, str(e)


def test_different_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for batch, classes in [(8, 5), (32, 100), (64, 1000)]:
            logits = torch.randn(batch, classes, device='cuda')
            targets = torch.randint(0, classes, (batch,), device='cuda')
            
            result = cross_entropy_loss(logits, targets)
            expected = F.cross_entropy(logits, targets, reduction='none')
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-3:
                return False, f"Failed at {batch}x{classes}: err={max_err:.6f}"
        
        return True, "Various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("log_softmax", test_log_softmax),
        ("cross_entropy_basic", test_cross_entropy_basic),
        ("cross_entropy_mean", test_cross_entropy_mean),
        ("label_smoothing", test_label_smoothing),
        ("different_sizes", test_different_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 14: Cross-Entropy Loss - Tests\n{'='*50}")
    
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
