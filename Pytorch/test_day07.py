"""Test Suite for Day 7: Optimizers"""

import torch
from typing import Tuple

try:
    from day07 import (SGD_Manual, SGD_Momentum, Adam_Manual, 
                       StepLRScheduler, train_simple_model)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_sgd() -> Tuple[bool, str]:
    try:
        w = torch.tensor([1.0], requires_grad=True)
        opt = SGD_Manual([w], lr=0.1)
        
        # Manual gradient
        loss = w ** 2
        loss.backward()
        
        opt.step()
        
        # w should be 1.0 - 0.1 * 2.0 = 0.8
        expected = 0.8
        if abs(w.item() - expected) > 1e-6:
            return False, f"w={w.item()}, expected={expected}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_sgd_momentum() -> Tuple[bool, str]:
    try:
        w = torch.tensor([1.0], requires_grad=True)
        opt = SGD_Momentum([w], lr=0.1, momentum=0.9)
        
        # First step
        loss = w ** 2
        loss.backward()
        opt.step()
        
        # v = 0 * 0.9 + 2.0 = 2.0
        # w = 1.0 - 0.1 * 2.0 = 0.8
        if abs(w.item() - 0.8) > 1e-6:
            return False, f"Step 1: w={w.item()}"
        
        # Second step
        opt.zero_grad()
        loss = w ** 2
        loss.backward()
        opt.step()
        
        # v = 2.0 * 0.9 + 1.6 = 3.4
        # w = 0.8 - 0.1 * 3.4 = 0.46
        if abs(w.item() - 0.46) > 1e-5:
            return False, f"Step 2: w={w.item()}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_adam() -> Tuple[bool, str]:
    try:
        w = torch.tensor([1.0], requires_grad=True)
        opt = Adam_Manual([w], lr=0.1, betas=(0.9, 0.999))
        
        # Take a step
        loss = w ** 2
        loss.backward()
        opt.step()
        
        # Should decrease
        if w.item() >= 1.0:
            return False, f"w should decrease, got {w.item()}"
        
        return True, f"OK (w={w.item():.4f})"
    except Exception as e:
        return False, str(e)


def test_step_scheduler() -> Tuple[bool, str]:
    try:
        w = torch.tensor([1.0], requires_grad=True)
        opt = SGD_Manual([w], lr=0.1)
        scheduler = StepLRScheduler(opt, step_size=2, gamma=0.5)
        
        # Initial lr
        if abs(scheduler.get_lr() - 0.1) > 1e-6:
            return False, f"Initial lr: {scheduler.get_lr()}"
        
        scheduler.step()  # epoch 1
        if abs(scheduler.get_lr() - 0.1) > 1e-6:
            return False, f"After epoch 1: {scheduler.get_lr()}"
        
        scheduler.step()  # epoch 2 - should decay
        if abs(scheduler.get_lr() - 0.05) > 1e-6:
            return False, f"After epoch 2: {scheduler.get_lr()}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_sgd_convergence() -> Tuple[bool, str]:
    try:
        result = train_simple_model('sgd', n_steps=200)
        if result is None:
            return False, "Not implemented"
        
        loss, w, b = result
        
        if loss > 0.1:
            return False, f"Loss too high: {loss:.4f}"
        if abs(w - 2.0) > 0.2:
            return False, f"w={w:.3f}, expected ~2"
        if abs(b - 1.0) > 0.2:
            return False, f"b={b:.3f}, expected ~1"
        
        return True, f"converged (loss={loss:.4f})"
    except Exception as e:
        return False, str(e)


def test_adam_convergence() -> Tuple[bool, str]:
    try:
        result = train_simple_model('adam', n_steps=100)
        if result is None:
            return False, "Not implemented"
        
        loss, w, b = result
        
        if loss > 0.01:
            return False, f"Loss too high: {loss:.4f}"
        
        return True, f"converged (loss={loss:.6f})"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("sgd", test_sgd),
        ("sgd_momentum", test_sgd_momentum),
        ("adam", test_adam),
        ("step_scheduler", test_step_scheduler),
        ("sgd_convergence", test_sgd_convergence),
        ("adam_convergence", test_adam_convergence),
    ]
    
    print(f"\n{'='*50}\nDay 7: Optimizers - Tests\n{'='*50}")
    
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
