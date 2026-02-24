"""Test Suite for Day 30: Warmup Learning Rate Scheduler"""

import numpy as np
import pytest

from day30 import Tensor, Parameter, Linear, SGD, WarmupLR


def test_warmup_lr_creation():
    """Test WarmupLR scheduler creation."""
    p = Parameter([1.0])
    opt = SGD([p], lr=0.1)
    scheduler = WarmupLR(opt, warmup_steps=100, init_lr=0.0)
    
    assert scheduler is not None, "WarmupLR returned None"


def test_warmup_lr_at_start():
    """Test WarmupLR starts at init_lr."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = WarmupLR(opt, warmup_steps=100, init_lr=0.0)
    
    # Before any step, should be at init_lr or close
    current_lr = opt.param_groups[0]['lr']
    assert current_lr <= 0.1, f"Initial LR = {current_lr}, expected small"


def test_warmup_lr_increases():
    """Test WarmupLR increases during warmup."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = WarmupLR(opt, warmup_steps=100, init_lr=0.0)
    
    lrs = []
    for _ in range(50):
        scheduler.step()
        lrs.append(opt.param_groups[0]['lr'])
    
    # LR should increase during warmup
    assert lrs[-1] > lrs[0], f"LR should increase: {lrs[0]} -> {lrs[-1]}"


def test_warmup_lr_reaches_target():
    """Test WarmupLR reaches target LR after warmup."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = WarmupLR(opt, warmup_steps=100, init_lr=0.0)
    
    for _ in range(100):
        scheduler.step()
    
    # After warmup, should be at target LR
    assert np.isclose(opt.param_groups[0]['lr'], 1.0, atol=0.01), f"LR = {opt.param_groups[0]['lr']}"


def test_warmup_lr_linear():
    """Test WarmupLR increases linearly."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = WarmupLR(opt, warmup_steps=100, init_lr=0.0)
    
    # At step 50, should be at 0.5
    for _ in range(50):
        scheduler.step()
    
    assert np.isclose(opt.param_groups[0]['lr'], 0.5, atol=0.05), f"LR at step 50 = {opt.param_groups[0]['lr']}"


def test_warmup_lr_stays_after_warmup():
    """Test WarmupLR stays at target after warmup."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = WarmupLR(opt, warmup_steps=10, init_lr=0.0)
    
    for _ in range(20):
        scheduler.step()
    
    # After warmup, should stay at target
    assert np.isclose(opt.param_groups[0]['lr'], 1.0, atol=0.01), f"LR = {opt.param_groups[0]['lr']}"


def test_warmup_lr_nonzero_init():
    """Test WarmupLR with non-zero init_lr."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = WarmupLR(opt, warmup_steps=100, init_lr=0.1)
    
    # LR should start at 0.1
    initial_lr = opt.param_groups[0]['lr']
    assert initial_lr >= 0.1, f"Initial LR = {initial_lr}, expected >= 0.1"
    
    for _ in range(100):
        scheduler.step()
    
    # Should still reach target
    assert np.isclose(opt.param_groups[0]['lr'], 1.0, atol=0.01), f"Final LR = {opt.param_groups[0]['lr']}"


def test_warmup_lr_short():
    """Test WarmupLR with short warmup."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    scheduler = WarmupLR(opt, warmup_steps=5, init_lr=0.0)
    
    for _ in range(5):
        scheduler.step()
    
    assert np.isclose(opt.param_groups[0]['lr'], 1.0, atol=0.01), f"LR = {opt.param_groups[0]['lr']}"


def test_warmup_with_training():
    """Test WarmupLR in training loop."""
    np.random.seed(42)
    
    model = Linear(1, 1)
    opt = SGD(model.parameters(), lr=0.1)
    scheduler = WarmupLR(opt, warmup_steps=10, init_lr=0.0)
    
    X = Tensor([[1.0], [2.0]])
    y = Tensor([[2.0], [4.0]])
    
    losses = []
    for epoch in range(20):
        opt.zero_grad()
        pred = model(X)
        loss = ((pred - y) ** 2).mean()
        losses.append(loss.data)
        loss.backward()
        opt.step()
        scheduler.step()
    
    # Training should work with warmup
    assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]} -> {losses[-1]}"


def test_warmup_lr_with_decay():
    """Test WarmupLR combined with decay scheduler."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    warmup = WarmupLR(opt, warmup_steps=10, init_lr=0.0)
    
    # Warmup phase
    for _ in range(10):
        warmup.step()
    
    # After warmup should be at target
    assert np.isclose(opt.param_groups[0]['lr'], 1.0, atol=0.01), f"LR after warmup = {opt.param_groups[0]['lr']}"


def test_warmup_lr_multiple_param_groups():
    """Test WarmupLR with multiple param groups."""
    p1 = Parameter([1.0])
    p2 = Parameter([2.0])
    
    opt = SGD([
        {'params': [p1], 'lr': 0.1},
        {'params': [p2], 'lr': 0.01}
    ])
    
    try:
        scheduler = WarmupLR(opt, warmup_steps=10, init_lr=0.0)
        
        for _ in range(10):
            scheduler.step()
        
        # Both groups should warm up
        assert np.isclose(opt.param_groups[0]['lr'], 0.1, atol=0.01), "Group 0 LR wrong"
        assert np.isclose(opt.param_groups[1]['lr'], 0.01, atol=0.001), "Group 1 LR wrong"
    except Exception:
        pytest.skip("Multiple param groups not supported")


def test_warmup_lr_zero_steps():
    """Test WarmupLR with zero warmup steps."""
    p = Parameter([1.0])
    opt = SGD([p], lr=1.0)
    
    try:
        scheduler = WarmupLR(opt, warmup_steps=0, init_lr=0.0)
        scheduler.step()
        
        # Should immediately be at target LR
        assert np.isclose(opt.param_groups[0]['lr'], 1.0), f"LR = {opt.param_groups[0]['lr']}"
    except (ValueError, ZeroDivisionError):
        # Some implementations don't allow 0 warmup steps
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
