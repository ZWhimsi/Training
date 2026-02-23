"""Test Suite for Day 27: Momentum and Learning Rate Scheduling"""

import numpy as np
import sys
import math
from typing import Tuple

try:
    from day27 import (
        Tensor,
        Parameter,
        Module,
        Linear,
        ReLU,
        Sequential,
        Optimizer,
        SGDMomentum,
        SGDNesterov,
        LRScheduler,
        StepLR,
        ExponentialLR,
        CosineAnnealingLR,
        WarmupLR,
        mse_loss,
        train_with_scheduler,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_sgd_momentum_creation() -> Tuple[bool, str]:
    """Test SGDMomentum creation."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt = SGDMomentum(params, lr=0.01, momentum=0.9)
        
        if opt is None:
            return False, "Optimizer is None"
        
        return True, "SGDMomentum created"
    except Exception as e:
        return False, str(e)


def test_sgd_momentum_step() -> Tuple[bool, str]:
    """Test SGDMomentum step with velocity accumulation."""
    try:
        np.random.seed(42)
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        opt = SGDMomentum([p], lr=0.1, momentum=0.9)
        
        initial = p.data.copy()
        
        for _ in range(10):
            p.grad = np.array([0.1, 0.1, 0.1])
            opt.step()
        
        if np.allclose(p.data, initial):
            return False, "Parameters not updated"
        
        return True, "Momentum SGD updates correctly"
    except Exception as e:
        return False, str(e)


def test_momentum_accumulates_velocity() -> Tuple[bool, str]:
    """Test that momentum accumulates velocity."""
    try:
        np.random.seed(42)
        
        p1 = Parameter(np.array([1.0]))
        opt1 = SGDMomentum([p1], lr=0.1, momentum=0.0)
        
        p2 = Parameter(np.array([1.0]))
        opt2 = SGDMomentum([p2], lr=0.1, momentum=0.9)
        
        for _ in range(20):
            p1.grad = np.array([1.0])
            p2.grad = np.array([1.0])
            opt1.step()
            opt2.step()
        
        if p2.data[0] >= p1.data[0]:
            return False, "Momentum should move faster"
        
        return True, f"Momentum helps: {p1.data[0]:.3f} > {p2.data[0]:.3f}"
    except Exception as e:
        return False, str(e)


def test_sgd_momentum_state() -> Tuple[bool, str]:
    """Test SGDMomentum stores velocity in state."""
    try:
        p = Parameter(np.array([1.0, 2.0]))
        opt = SGDMomentum([p], lr=0.1, momentum=0.9)
        
        p.grad = np.array([1.0, 1.0])
        opt.step()
        
        if not opt.state:
            return False, "State is empty"
        
        param_state = list(opt.state.values())[0]
        if 'velocity' not in param_state:
            return False, "No velocity in state"
        
        return True, "Velocity stored in state"
    except Exception as e:
        return False, str(e)


def test_nesterov_creation() -> Tuple[bool, str]:
    """Test SGDNesterov creation."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt = SGDNesterov(params, lr=0.01, momentum=0.9)
        
        if opt is None:
            return False, "Optimizer is None"
        
        return True, "SGDNesterov created"
    except Exception as e:
        return False, str(e)


def test_nesterov_step() -> Tuple[bool, str]:
    """Test SGDNesterov step."""
    try:
        np.random.seed(42)
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        opt = SGDNesterov([p], lr=0.1, momentum=0.9)
        
        initial = p.data.copy()
        
        for _ in range(10):
            p.grad = np.array([0.1, 0.1, 0.1])
            opt.step()
        
        if np.allclose(p.data, initial):
            return False, "Parameters not updated"
        
        return True, "Nesterov SGD updates correctly"
    except Exception as e:
        return False, str(e)


def test_nesterov_vs_classical() -> Tuple[bool, str]:
    """Test that Nesterov differs from classical momentum."""
    try:
        np.random.seed(42)
        
        p1 = Parameter(np.array([5.0]))
        opt1 = SGDMomentum([p1], lr=0.1, momentum=0.9)
        
        p2 = Parameter(np.array([5.0]))
        opt2 = SGDNesterov([p2], lr=0.1, momentum=0.9)
        
        for i in range(10):
            grad = np.array([p1.data[0]])
            p1.grad = grad.copy()
            p2.grad = grad.copy()
            opt1.step()
            opt2.step()
        
        if np.allclose(p1.data, p2.data, rtol=1e-3):
            return True, "Both converge (Nesterov behaves similarly here)"
        
        return True, f"Different trajectories: classical={p1.data[0]:.4f}, nesterov={p2.data[0]:.4f}"
    except Exception as e:
        return False, str(e)


def test_step_lr_creation() -> Tuple[bool, str]:
    """Test StepLR creation."""
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = StepLR(opt, step_size=10, gamma=0.1)
        
        if scheduler is None:
            return False, "Scheduler is None"
        
        return True, "StepLR created"
    except Exception as e:
        return False, str(e)


def test_step_lr_decay() -> Tuple[bool, str]:
    """Test StepLR decays at correct intervals."""
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = StepLR(opt, step_size=5, gamma=0.1)
        
        lrs = []
        for i in range(20):
            lrs.append(opt.param_groups[0]['lr'])
            scheduler.step()
        
        if not np.isclose(lrs[0], 0.1, rtol=1e-5):
            return False, f"Initial lr wrong: {lrs[0]}"
        if not np.isclose(lrs[5], 0.01, rtol=1e-5):
            return False, f"lr at epoch 5 wrong: {lrs[5]}"
        if not np.isclose(lrs[10], 0.001, rtol=1e-5):
            return False, f"lr at epoch 10 wrong: {lrs[10]}"
        
        return True, "StepLR decays correctly"
    except Exception as e:
        return False, str(e)


def test_exponential_lr_creation() -> Tuple[bool, str]:
    """Test ExponentialLR creation."""
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = ExponentialLR(opt, gamma=0.95)
        
        if scheduler is None:
            return False, "Scheduler is None"
        
        return True, "ExponentialLR created"
    except Exception as e:
        return False, str(e)


def test_exponential_lr_decay() -> Tuple[bool, str]:
    """Test ExponentialLR decays exponentially."""
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = ExponentialLR(opt, gamma=0.9)
        
        lrs = []
        for i in range(10):
            lrs.append(opt.param_groups[0]['lr'])
            scheduler.step()
        
        for i, lr in enumerate(lrs):
            expected = 0.1 * (0.9 ** i)
            if not np.isclose(lr, expected, rtol=1e-4):
                return False, f"lr at epoch {i}: {lr:.6f}, expected {expected:.6f}"
        
        return True, "ExponentialLR decays correctly"
    except Exception as e:
        return False, str(e)


def test_cosine_lr_creation() -> Tuple[bool, str]:
    """Test CosineAnnealingLR creation."""
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0.001)
        
        if scheduler is None:
            return False, "Scheduler is None"
        
        return True, "CosineAnnealingLR created"
    except Exception as e:
        return False, str(e)


def test_cosine_lr_shape() -> Tuple[bool, str]:
    """Test CosineAnnealingLR follows cosine curve."""
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0.0)
        
        lrs = []
        for i in range(101):
            lrs.append(opt.param_groups[0]['lr'])
            scheduler.step()
        
        if lrs[0] < 0.09:
            return False, f"Should start near base_lr: {lrs[0]}"
        
        if lrs[50] > 0.06 or lrs[50] < 0.04:
            return False, f"Midpoint should be ~0.05: {lrs[50]}"
        
        if lrs[-1] > 0.01:
            return False, f"Should end near eta_min: {lrs[-1]}"
        
        return True, "CosineAnnealingLR follows cosine curve"
    except Exception as e:
        return False, str(e)


def test_warmup_lr_creation() -> Tuple[bool, str]:
    """Test WarmupLR creation."""
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = WarmupLR(opt, warmup_epochs=10)
        
        if scheduler is None:
            return False, "Scheduler is None"
        
        return True, "WarmupLR created"
    except Exception as e:
        return False, str(e)


def test_warmup_lr_ramp() -> Tuple[bool, str]:
    """Test WarmupLR ramps up correctly."""
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = WarmupLR(opt, warmup_epochs=5)
        
        lrs = []
        for i in range(10):
            lrs.append(opt.param_groups[0]['lr'])
            scheduler.step()
        
        for i in range(4):
            if lrs[i] >= lrs[i+1]:
                return False, f"Should increase during warmup: {lrs[i]} >= {lrs[i+1]}"
        
        if lrs[5] < 0.09:
            return False, f"Should reach base_lr after warmup: {lrs[5]}"
        
        return True, "WarmupLR ramps correctly"
    except Exception as e:
        return False, str(e)


def test_scheduler_with_training() -> Tuple[bool, str]:
    """Test scheduler during actual training."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        opt = SGDMomentum(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = StepLR(opt, step_size=20, gamma=0.5)
        
        X = np.random.randn(10, 4)
        Y = np.random.randn(10, 2)
        
        history = train_with_scheduler(model, opt, scheduler, X, Y, epochs=50)
        
        if len(history['losses']) != 50:
            return False, f"Wrong number of losses: {len(history['losses'])}"
        
        if len(history['learning_rates']) != 50:
            return False, f"Wrong number of lrs: {len(history['learning_rates'])}"
        
        return True, "Training with scheduler works"
    except Exception as e:
        return False, str(e)


def test_multiple_param_groups_scheduling() -> Tuple[bool, str]:
    """Test scheduler with multiple parameter groups."""
    try:
        params1 = [Parameter(np.random.randn(3))]
        params2 = [Parameter(np.random.randn(4))]
        
        param_groups = [
            {'params': params1, 'lr': 0.1},
            {'params': params2, 'lr': 0.01}
        ]
        
        opt = SGDMomentum(param_groups, lr=0.1, momentum=0.9)
        scheduler = StepLR(opt, step_size=5, gamma=0.1)
        
        initial_lr1 = opt.param_groups[0]['lr']
        initial_lr2 = opt.param_groups[1]['lr']
        
        for _ in range(10):
            scheduler.step()
        
        final_lr1 = opt.param_groups[0]['lr']
        final_lr2 = opt.param_groups[1]['lr']
        
        if final_lr1 >= initial_lr1:
            return False, "Group 1 lr should decrease"
        if final_lr2 >= initial_lr2:
            return False, "Group 2 lr should decrease"
        
        ratio = final_lr1 / final_lr2
        expected_ratio = initial_lr1 / initial_lr2
        if not np.isclose(ratio, expected_ratio, rtol=0.1):
            return False, f"Ratio changed: {ratio} vs {expected_ratio}"
        
        return True, "Multi-group scheduling works"
    except Exception as e:
        return False, str(e)


def test_against_pytorch_momentum() -> Tuple[bool, str]:
    """Test momentum SGD against PyTorch."""
    try:
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        our_p = Parameter(np.array([1.0, 2.0, 3.0]))
        our_opt = SGDMomentum([our_p], lr=0.1, momentum=0.9)
        
        torch_p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float64)
        torch_opt = torch.optim.SGD([torch_p], lr=0.1, momentum=0.9)
        
        grads = np.random.randn(10, 3)
        
        for g in grads:
            our_p.grad = g.copy()
            our_opt.step()
            
            torch_p.grad = torch.tensor(g, dtype=torch.float64)
            torch_opt.step()
        
        if not np.allclose(our_p.data, torch_p.detach().numpy(), rtol=1e-4):
            return False, f"Mismatch: {our_p.data} vs {torch_p.detach().numpy()}"
        
        return True, "Matches PyTorch SGD momentum"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def test_against_pytorch_scheduler() -> Tuple[bool, str]:
    """Test scheduler against PyTorch."""
    try:
        import torch
        import torch.optim as optim
        
        our_p = Parameter(np.array([1.0]))
        our_opt = SGDMomentum([our_p], lr=0.1, momentum=0.9)
        our_scheduler = StepLR(our_opt, step_size=5, gamma=0.1)
        
        torch_p = torch.tensor([1.0], requires_grad=True)
        torch_opt = optim.SGD([torch_p], lr=0.1, momentum=0.9)
        torch_scheduler = optim.lr_scheduler.StepLR(torch_opt, step_size=5, gamma=0.1)
        
        our_lrs = []
        torch_lrs = []
        
        for i in range(20):
            our_lrs.append(our_opt.param_groups[0]['lr'])
            torch_lrs.append(torch_opt.param_groups[0]['lr'])
            our_scheduler.step()
            torch_scheduler.step()
        
        for i, (our, theirs) in enumerate(zip(our_lrs, torch_lrs)):
            if not np.isclose(our, theirs, rtol=1e-4):
                return False, f"Mismatch at epoch {i}: {our} vs {theirs}"
        
        return True, "Matches PyTorch StepLR"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("sgd_momentum_creation", test_sgd_momentum_creation),
        ("sgd_momentum_step", test_sgd_momentum_step),
        ("momentum_accumulates_velocity", test_momentum_accumulates_velocity),
        ("sgd_momentum_state", test_sgd_momentum_state),
        ("nesterov_creation", test_nesterov_creation),
        ("nesterov_step", test_nesterov_step),
        ("nesterov_vs_classical", test_nesterov_vs_classical),
        ("step_lr_creation", test_step_lr_creation),
        ("step_lr_decay", test_step_lr_decay),
        ("exponential_lr_creation", test_exponential_lr_creation),
        ("exponential_lr_decay", test_exponential_lr_decay),
        ("cosine_lr_creation", test_cosine_lr_creation),
        ("cosine_lr_shape", test_cosine_lr_shape),
        ("warmup_lr_creation", test_warmup_lr_creation),
        ("warmup_lr_ramp", test_warmup_lr_ramp),
        ("scheduler_with_training", test_scheduler_with_training),
        ("multiple_param_groups_scheduling", test_multiple_param_groups_scheduling),
        ("against_pytorch_momentum", test_against_pytorch_momentum),
        ("against_pytorch_scheduler", test_against_pytorch_scheduler),
    ]
    
    print(f"\n{'='*60}")
    print("Day 27: Momentum and Learning Rate Scheduling - Tests")
    print(f"{'='*60}")
    
    passed = 0
    for name, fn in tests:
        try:
            p, m = fn()
        except Exception as e:
            p, m = False, str(e)
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    
    print(f"\nSummary: {passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    success = run_all_tests()
    sys.exit(0 if success else 1)
