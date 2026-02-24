"""Test Suite for Day 6: Loss Functions"""

import torch
import torch.nn.functional as F
from typing import Tuple

try:
    from day06 import (mse_loss_manual, bce_loss_manual, bce_with_logits_manual,
                       cross_entropy_manual, cross_entropy_smooth,
                       huber_loss_manual, FocalLoss)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_mse_loss() -> Tuple[bool, str]:
    try:
        pred = torch.randn(8, 4)
        target = torch.randn(8, 4)
        
        result = mse_loss_manual(pred, target)
        if result is None:
            return False, "Not implemented"
        
        expected = F.mse_loss(pred, target)
        err = (result - expected).abs().item()
        
        if err > 1e-6:
            return False, f"Error: {err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_bce_loss() -> Tuple[bool, str]:
    try:
        pred = torch.sigmoid(torch.randn(8, 4))
        target = torch.randint(0, 2, (8, 4)).float()
        
        result = bce_loss_manual(pred, target)
        if result is None:
            return False, "Not implemented"
        
        expected = F.binary_cross_entropy(pred, target)
        err = (result - expected).abs().item()
        
        if err > 1e-5:
            return False, f"Error: {err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_bce_with_logits() -> Tuple[bool, str]:
    try:
        logits = torch.randn(8, 4)
        target = torch.randint(0, 2, (8, 4)).float()
        
        result = bce_with_logits_manual(logits, target)
        if result is None:
            return False, "Not implemented"
        
        expected = F.binary_cross_entropy_with_logits(logits, target)
        err = (result - expected).abs().item()
        
        if err > 1e-5:
            return False, f"Error: {err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_cross_entropy() -> Tuple[bool, str]:
    try:
        logits = torch.randn(8, 10)
        target = torch.randint(0, 10, (8,))
        
        result = cross_entropy_manual(logits, target)
        if result is None:
            return False, "Not implemented"
        
        expected = F.cross_entropy(logits, target)
        err = (result - expected).abs().item()
        
        if err > 1e-5:
            return False, f"Error: {err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_cross_entropy_smooth() -> Tuple[bool, str]:
    try:
        logits = torch.randn(8, 10)
        target = torch.randint(0, 10, (8,))
        
        result = cross_entropy_smooth(logits, target, smoothing=0.1)
        if result is None:
            return False, "Not implemented"
        
        expected = F.cross_entropy(logits, target, label_smoothing=0.1)
        err = (result - expected).abs().item()
        
        if err > 1e-4:
            return False, f"Error: {err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_huber_loss() -> Tuple[bool, str]:
    try:
        pred = torch.randn(8, 4)
        target = torch.randn(8, 4)
        
        result = huber_loss_manual(pred, target, delta=1.0)
        if result is None:
            return False, "Not implemented"
        
        expected = F.huber_loss(pred, target, delta=1.0)
        err = (result - expected).abs().item()
        
        if err > 1e-5:
            return False, f"Error: {err:.6f}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_focal_loss() -> Tuple[bool, str]:
    try:
        torch.manual_seed(42)
        logits = torch.randn(8, 10)
        target = torch.randint(0, 10, (8,))
        
        focal = FocalLoss(alpha=0.25, gamma=2.0)
        result = focal(logits, target)
        
        if result is None:
            return False, "Not implemented"
        
        # Should be a positive scalar
        if result.numel() != 1:
            return False, "Should return scalar"
        if result.item() < 0:
            return False, "Loss should be positive"
        
        # Compute expected focal loss manually
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        p = torch.exp(-ce_loss)
        focal_weight = 0.25 * (1 - p) ** 2.0
        expected = (focal_weight * ce_loss).mean()
        
        err = (result - expected).abs().item()
        if err > 1e-5:
            return False, f"Error: {err:.6f}, got {result.item():.4f}, expected {expected.item():.4f}"
        
        return True, f"OK (loss={result.item():.4f})"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("mse_loss", test_mse_loss),
        ("bce_loss", test_bce_loss),
        ("bce_with_logits", test_bce_with_logits),
        ("cross_entropy", test_cross_entropy),
        ("cross_entropy_smooth", test_cross_entropy_smooth),
        ("huber_loss", test_huber_loss),
        ("focal_loss", test_focal_loss),
    ]
    
    print(f"\n{'='*50}\nDay 6: Loss Functions - Tests\n{'='*50}")
    
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
