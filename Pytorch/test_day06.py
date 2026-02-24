"""Test Suite for Day 6: Loss Functions"""

import torch
import pytest
import torch.nn.functional as F
try:
    from day06 import (mse_loss_manual, bce_loss_manual, bce_with_logits_manual,
                       cross_entropy_manual, cross_entropy_smooth,
                       huber_loss_manual, FocalLoss)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_mse_loss():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    pred = torch.randn(8, 4)
    target = torch.randn(8, 4)
    
    result = mse_loss_manual(pred, target)
    assert result is not None, "Not implemented"
    
    expected = F.mse_loss(pred, target)
    err = (result - expected).abs().item()
    
    assert err <= 1e-6, f"Error: {err:.6f}"

def test_bce_loss():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    pred = torch.sigmoid(torch.randn(8, 4))
    target = torch.randint(0, 2, (8, 4)).float()
    
    result = bce_loss_manual(pred, target)
    assert result is not None, "Not implemented"
    
    expected = F.binary_cross_entropy(pred, target)
    err = (result - expected).abs().item()
    
    assert err <= 1e-5, f"Error: {err:.6f}"

def test_bce_with_logits():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    logits = torch.randn(8, 4)
    target = torch.randint(0, 2, (8, 4)).float()
    
    result = bce_with_logits_manual(logits, target)
    assert result is not None, "Not implemented"
    
    expected = F.binary_cross_entropy_with_logits(logits, target)
    err = (result - expected).abs().item()
    
    assert err <= 1e-5, f"Error: {err:.6f}"

def test_cross_entropy():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    logits = torch.randn(8, 10)
    target = torch.randint(0, 10, (8,))
    
    result = cross_entropy_manual(logits, target)
    assert result is not None, "Not implemented"
    
    expected = F.cross_entropy(logits, target)
    err = (result - expected).abs().item()
    
    assert err <= 1e-5, f"Error: {err:.6f}"

def test_cross_entropy_smooth():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    logits = torch.randn(8, 10)
    target = torch.randint(0, 10, (8,))
    
    result = cross_entropy_smooth(logits, target, smoothing=0.1)
    assert result is not None, "Not implemented"
    
    expected = F.cross_entropy(logits, target, label_smoothing=0.1)
    err = (result - expected).abs().item()
    
    assert err <= 1e-4, f"Error: {err:.6f}"

def test_huber_loss():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    pred = torch.randn(8, 4)
    target = torch.randn(8, 4)
    
    result = huber_loss_manual(pred, target, delta=1.0)
    assert result is not None, "Not implemented"
    
    expected = F.huber_loss(pred, target, delta=1.0)
    err = (result - expected).abs().item()
    
    assert err <= 1e-5, f"Error: {err:.6f}"

def test_focal_loss():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    logits = torch.randn(8, 10)
    target = torch.randint(0, 10, (8,))
    
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    result = focal(logits, target)
    
    assert result is not None, "Not implemented"
    
    assert result.numel() == 1, "Should return scalar"
    assert result.item() >= 0, "Loss should be positive"
    
    ce_loss = F.cross_entropy(logits, target, reduction='none')
    p = torch.exp(-ce_loss)
    focal_weight = 0.25 * (1 - p) ** 2.0
    expected = (focal_weight * ce_loss).mean()
    
    err = (result - expected).abs().item()
    assert err <= 1e-5, f"Error: {err:.6f}, got {result.item():.4f}, expected {expected.item():.4f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
