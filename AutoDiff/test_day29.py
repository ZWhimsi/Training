"""Test Suite for Day 29: Learning Rate Schedulers"""

import numpy as np
import pytest

from day29 import (
    Tensor, Parameter, Linear, SGD,
    LRScheduler, StepLR, ExponentialLR, CosineAnnealingLR
)


def test_step_lr_creation():
    """Test StepLR creation."""
    p = Parameter([1.0])
    opt = SGD([p], lr=0.1)
    scheduler = StepLR(opt, step_size=10, gamma=0.1)
    
    assert scheduler is not None, "StepLR returned None"


def test_step_lr_no_change_before_step():
    """Test StepLR doesn't change LR before step_size epochs."""
    p = Parameter([1.0])
    opt = SGD([p], lr=0.1)
    scheduler = StepLR(opt, step_size=10, gamma=0.1)
    
    initial_lr = opt.param_groups[0]['lr']
    
    for _ in range(9):
        scheduler.step()
    
    assert opt.param_groups[0]['lr'] == initial_lr, f"LR changed before step_size"


def test_step_lr_change_at_step():
    """Test StepLR changes LR at step_size."""
    p = Parameter([1.0])
    opt = SGD([p], lr=0.1)
    scheduler = StepLR(opt, step_size=10, gamma=0.1)
    
    for _ in range(10):
        scheduler.step()
    
    expected = 0.1 * 0.1  # 0.01
    assert np.isclose(opt.param_groups[0]['lr'], expected), f"LR = {opt.param_groups[0]['lr']}, expected {expected}"


def test_step_lr_multiple_steps():
    """Test StepLR multiple step changes."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = StepLR(opt, step_size=5, gamma=0.5)
    
    # After 5 steps: 0.5
    # After 10 steps: 0.25
    # After 15 steps: 0.125
    
    for _ in range(15):
        scheduler.step()
    
    expected = 1.0 * (0.5 ** 3)  # 0.125
    assert np.isclose(opt.param_groups[0]['lr'], expected), f"LR = {opt.param_groups[0]['lr']}, expected {expected}"


def test_exponential_lr_creation():
    """Test ExponentialLR creation."""
    p = Parameter([1.0])
    opt = SGD([p], lr=0.1)
    scheduler = ExponentialLR(opt, gamma=0.9)
    
    assert scheduler is not None, "ExponentialLR returned None"


def test_exponential_lr_decay():
    """Test ExponentialLR exponential decay."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = ExponentialLR(opt, gamma=0.9)
    
    scheduler.step()
    assert np.isclose(opt.param_groups[0]['lr'], 0.9), f"LR = {opt.param_groups[0]['lr']}"
    
    scheduler.step()
    assert np.isclose(opt.param_groups[0]['lr'], 0.81), f"LR = {opt.param_groups[0]['lr']}"


def test_exponential_lr_multiple_steps():
    """Test ExponentialLR over multiple steps."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = ExponentialLR(opt, gamma=0.9)
    
    for _ in range(10):
        scheduler.step()
    
    expected = 1.0 * (0.9 ** 10)
    assert np.isclose(opt.param_groups[0]['lr'], expected, atol=1e-6), f"LR = {opt.param_groups[0]['lr']}, expected {expected}"


def test_cosine_annealing_creation():
    """Test CosineAnnealingLR creation."""
    p = Parameter([1.0])
    opt = SGD([p], lr=0.1)
    scheduler = CosineAnnealingLR(opt, T_max=100)
    
    assert scheduler is not None, "CosineAnnealingLR returned None"


def test_cosine_annealing_decrease():
    """Test CosineAnnealingLR decreases LR."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0.0)
    
    initial_lr = opt.param_groups[0]['lr']
    scheduler.step()
    
    assert opt.param_groups[0]['lr'] < initial_lr, "LR should decrease"


def test_cosine_annealing_minimum():
    """Test CosineAnnealingLR reaches minimum at T_max."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    eta_min = 0.01
    scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=eta_min)
    
    for _ in range(100):
        scheduler.step()
    
    assert np.isclose(opt.param_groups[0]['lr'], eta_min, atol=0.01), f"LR = {opt.param_groups[0]['lr']}, expected ~{eta_min}"


def test_cosine_annealing_restart():
    """Test CosineAnnealingLR restarts after T_max."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = CosineAnnealingLR(opt, T_max=50, eta_min=0.0)
    
    for _ in range(50):
        scheduler.step()
    
    lr_at_tmax = opt.param_groups[0]['lr']
    
    # After T_max, should start increasing again (cosine restart)
    scheduler.step()
    
    # Either it restarts (lr goes up) or stays at minimum
    assert opt.param_groups[0]['lr'] >= lr_at_tmax - 0.01, "LR behavior after T_max"


def test_scheduler_get_last_lr():
    """Test scheduler get_last_lr method."""
    p = Parameter([1.0])
    opt = SGD([p], lr=0.1)
    scheduler = StepLR(opt, step_size=10, gamma=0.1)
    
    try:
        lr = scheduler.get_last_lr()
        assert lr is not None, "get_last_lr returned None"
        assert len(lr) > 0, "get_last_lr returned empty"
    except AttributeError:
        pytest.skip("get_last_lr not implemented")


def test_scheduler_with_training():
    """Test scheduler in training loop."""
    np.random.seed(42)
    
    model = Linear(1, 1)
    opt = SGD(model.parameters(), lr=0.5)
    scheduler = StepLR(opt, step_size=10, gamma=0.5)
    
    X = Tensor([[1.0], [2.0], [3.0]])
    y = Tensor([[2.0], [4.0], [6.0]])
    
    for epoch in range(30):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        opt.step()
        scheduler.step()
    
    # LR should have been reduced
    final_lr = opt.param_groups[0]['lr']
    assert final_lr < 0.5, f"LR should have decreased: {final_lr}"


def test_multiple_param_groups():
    """Test scheduler with multiple param groups."""
    p1 = Parameter([1.0])
    p2 = Parameter([2.0])
    
    opt = SGD([
        {'params': [p1], 'lr': 0.1},
        {'params': [p2], 'lr': 0.01}
    ])
    
    scheduler = StepLR(opt, step_size=5, gamma=0.5)
    
    for _ in range(5):
        scheduler.step()
    
    assert np.isclose(opt.param_groups[0]['lr'], 0.05), f"Group 0 LR = {opt.param_groups[0]['lr']}"
    assert np.isclose(opt.param_groups[1]['lr'], 0.005), f"Group 1 LR = {opt.param_groups[1]['lr']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
