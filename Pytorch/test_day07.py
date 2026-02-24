"""Test Suite for Day 7: Optimizers"""

import torch
import pytest
try:
    from day07 import (SGD_Manual, SGD_Momentum, Adam_Manual, 
                       StepLRScheduler, train_simple_model)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_sgd():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    w = torch.tensor([1.0], requires_grad=True)
    opt = SGD_Manual([w], lr=0.1)
    
    loss = w ** 2
    loss.backward()
    
    opt.step()
    
    expected = 0.8
    assert abs(w.item() - expected) <= 1e-6, f"w={w.item()}, expected={expected}"

def test_sgd_momentum():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    w = torch.tensor([1.0], requires_grad=True)
    opt = SGD_Momentum([w], lr=0.1, momentum=0.9)
    
    loss = w ** 2
    loss.backward()
    opt.step()
    
    assert abs(w.item() - 0.8) <= 1e-6, f"Step 1: w={w.item()}"
    
    opt.zero_grad()
    loss = w ** 2
    loss.backward()
    opt.step()
    
    assert abs(w.item() - 0.46) <= 1e-5, f"Step 2: w={w.item()}"

def test_adam():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    w = torch.tensor([1.0], requires_grad=True)
    opt = Adam_Manual([w], lr=0.1, betas=(0.9, 0.999))
    
    loss = w ** 2
    loss.backward()
    opt.step()
    
    expected = 0.9
    assert abs(w.item() - expected) <= 1e-5, f"w={w.item():.6f}, expected={expected}"

def test_step_scheduler():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    w = torch.tensor([1.0], requires_grad=True)
    opt = SGD_Manual([w], lr=0.1)
    scheduler = StepLRScheduler(opt, step_size=2, gamma=0.5)
    
    assert abs(scheduler.get_lr() - 0.1) <= 1e-6, f"Initial lr: {scheduler.get_lr()}"
    
    scheduler.step()
    assert abs(scheduler.get_lr() - 0.1) <= 1e-6, f"After epoch 1: {scheduler.get_lr()}"
    
    scheduler.step()
    assert abs(scheduler.get_lr() - 0.05) <= 1e-6, f"After epoch 2: {scheduler.get_lr()}"

def test_sgd_convergence():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = train_simple_model('sgd', n_steps=200)
    assert result is not None, "Not implemented"
    
    loss, w, b = result
    
    assert loss <= 0.1, f"Loss too high: {loss:.4f}"
    assert abs(w - 2.0) <= 0.2, f"w={w:.3f}, expected ~2"
    assert abs(b - 1.0) <= 0.2, f"b={b:.3f}, expected ~1"

def test_adam_convergence():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = train_simple_model('adam', n_steps=100)
    assert result is not None, "Not implemented"
    
    loss, w, b = result
    
    assert loss <= 0.01, f"Loss too high: {loss:.4f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
