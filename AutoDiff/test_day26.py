"""Test Suite for Day 26: SGD Optimizer"""

import numpy as np
import sys
from typing import Tuple

try:
    from day26 import (
        Tensor,
        Parameter,
        Module,
        Linear,
        ReLU,
        Sequential,
        Optimizer,
        SGD,
        SGDWithState,
        create_optimizer_with_param_groups,
        mse_loss,
        train_step,
        train_loop,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_optimizer_init() -> Tuple[bool, str]:
    """Test Optimizer initialization."""
    try:
        params = [Parameter(np.random.randn(3, 4)) for _ in range(2)]
        opt = SGD(params, lr=0.01)
        
        if not hasattr(opt, 'param_groups'):
            return False, "No param_groups attribute"
        if not opt.param_groups:
            return False, "param_groups is empty"
        
        return True, "Optimizer initialized"
    except Exception as e:
        return False, str(e)


def test_optimizer_defaults() -> Tuple[bool, str]:
    """Test Optimizer stores defaults."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt = SGD(params, lr=0.01)
        
        if not hasattr(opt, 'defaults'):
            return False, "No defaults attribute"
        if 'lr' not in opt.defaults:
            return False, "lr not in defaults"
        
        return True, "Defaults stored"
    except Exception as e:
        return False, str(e)


def test_optimizer_param_groups() -> Tuple[bool, str]:
    """Test parameter groups structure."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt = SGD(params, lr=0.01)
        
        if not opt.param_groups:
            return False, "No param_groups"
        
        group = opt.param_groups[0]
        if 'params' not in group:
            return False, "No params in group"
        if 'lr' not in group:
            return False, "No lr in group"
        
        return True, "Param groups structured correctly"
    except Exception as e:
        return False, str(e)


def test_sgd_creation() -> Tuple[bool, str]:
    """Test SGD optimizer creation."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt = SGD(params, lr=0.01)
        
        if opt is None:
            return False, "SGD is None"
        
        return True, "SGD created"
    except Exception as e:
        return False, str(e)


def test_sgd_step_basic() -> Tuple[bool, str]:
    """Test basic SGD step."""
    try:
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        p.grad = np.array([1.0, 1.0, 1.0])
        
        opt = SGD([p], lr=0.1)
        old_data = p.data.copy()
        opt.step()
        
        expected = old_data - 0.1 * p.grad
        if not np.allclose(p.data, expected):
            return False, f"Got {p.data}, expected {expected}"
        
        return True, "SGD step works"
    except Exception as e:
        return False, str(e)


def test_sgd_multiple_params() -> Tuple[bool, str]:
    """Test SGD with multiple parameters."""
    try:
        p1 = Parameter(np.array([1.0, 2.0]))
        p2 = Parameter(np.array([3.0, 4.0]))
        p1.grad = np.array([0.1, 0.2])
        p2.grad = np.array([0.3, 0.4])
        
        opt = SGD([p1, p2], lr=1.0)
        
        old1 = p1.data.copy()
        old2 = p2.data.copy()
        
        opt.step()
        
        if not np.allclose(p1.data, old1 - p1.grad):
            return False, "p1 not updated correctly"
        if not np.allclose(p2.data, old2 - p2.grad):
            return False, "p2 not updated correctly"
        
        return True, "Multiple params updated"
    except Exception as e:
        return False, str(e)


def test_zero_grad_basic() -> Tuple[bool, str]:
    """Test zero_grad method."""
    try:
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        p.grad = np.array([0.1, 0.2, 0.3])
        
        opt = SGD([p], lr=0.1)
        opt.zero_grad()
        
        if not np.all(p.grad == 0):
            return False, f"Grad not zeroed: {p.grad}"
        
        return True, "zero_grad works"
    except Exception as e:
        return False, str(e)


def test_zero_grad_multiple() -> Tuple[bool, str]:
    """Test zero_grad with multiple parameters."""
    try:
        p1 = Parameter(np.random.randn(3))
        p2 = Parameter(np.random.randn(4))
        p1.grad = np.ones(3)
        p2.grad = np.ones(4)
        
        opt = SGD([p1, p2], lr=0.1)
        opt.zero_grad()
        
        if not np.all(p1.grad == 0):
            return False, "p1 grad not zeroed"
        if not np.all(p2.grad == 0):
            return False, "p2 grad not zeroed"
        
        return True, "Multiple params zeroed"
    except Exception as e:
        return False, str(e)


def test_mse_loss() -> Tuple[bool, str]:
    """Test MSE loss function."""
    try:
        pred = Tensor(np.array([1.0, 2.0, 3.0]))
        target = Tensor(np.array([1.1, 2.1, 3.1]))
        
        loss = mse_loss(pred, target)
        expected = 0.01
        
        if not np.isclose(loss.data, expected, rtol=1e-5):
            return False, f"Loss = {loss.data}, expected {expected}"
        
        return True, "MSE loss correct"
    except Exception as e:
        return False, str(e)


def test_train_step_runs() -> Tuple[bool, str]:
    """Test that train_step runs without error."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        opt = SGD(model.parameters(), lr=0.01)
        
        x = Tensor(np.random.randn(4, 4))
        y = Tensor(np.random.randn(4, 2))
        
        loss = train_step(model, opt, x, y)
        
        if loss <= 0:
            return False, f"Invalid loss: {loss}"
        
        return True, f"train_step returned loss={loss:.4f}"
    except Exception as e:
        return False, str(e)


def test_train_step_updates_params() -> Tuple[bool, str]:
    """Test that train_step updates parameters."""
    try:
        np.random.seed(42)
        model = Linear(4, 2)
        opt = SGD(model.parameters(), lr=0.1)
        
        old_weight = model.weight.data.copy()
        
        x = Tensor(np.random.randn(4, 4))
        y = Tensor(np.random.randn(4, 2))
        
        train_step(model, opt, x, y)
        
        if np.allclose(model.weight.data, old_weight):
            return False, "Weights not updated"
        
        return True, "Parameters updated"
    except Exception as e:
        return False, str(e)


def test_train_loop_returns_losses() -> Tuple[bool, str]:
    """Test that train_loop returns loss history."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        opt = SGD(model.parameters(), lr=0.01)
        
        X = np.random.randn(10, 4)
        Y = np.random.randn(10, 2)
        
        losses = train_loop(model, opt, X, Y, epochs=20)
        
        if len(losses) != 20:
            return False, f"Expected 20 losses, got {len(losses)}"
        
        return True, "Loss history returned"
    except Exception as e:
        return False, str(e)


def test_train_loop_convergence() -> Tuple[bool, str]:
    """Test that training actually reduces loss."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        opt = SGD(model.parameters(), lr=0.01)
        
        X = np.random.randn(10, 4)
        Y = np.random.randn(10, 2)
        
        losses = train_loop(model, opt, X, Y, epochs=100)
        
        if not losses:
            return False, "No losses returned"
        
        if losses[-1] >= losses[0]:
            return False, f"Loss didn't decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        
        return True, f"Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}"
    except Exception as e:
        return False, str(e)


def test_param_groups_creation() -> Tuple[bool, str]:
    """Test creating optimizer with parameter groups."""
    try:
        class TwoLayerNet(Module):
            def __init__(self):
                super().__init__()
                self.layer1 = Linear(4, 8)
                self.layer2 = Linear(8, 2)
            
            def forward(self, x):
                return self.layer2(ReLU()(self.layer1(x)))
        
        model = TwoLayerNet()
        opt = create_optimizer_with_param_groups(model, base_lr=0.1)
        
        if opt is None:
            return False, "Function returned None"
        
        if len(opt.param_groups) != 2:
            return False, f"Expected 2 groups, got {len(opt.param_groups)}"
        
        return True, "Parameter groups created"
    except Exception as e:
        return False, str(e)


def test_param_groups_different_lr() -> Tuple[bool, str]:
    """Test parameter groups have different learning rates."""
    try:
        class TwoLayerNet(Module):
            def __init__(self):
                super().__init__()
                self.layer1 = Linear(4, 8)
                self.layer2 = Linear(8, 2)
            
            def forward(self, x):
                return self.layer2(ReLU()(self.layer1(x)))
        
        model = TwoLayerNet()
        opt = create_optimizer_with_param_groups(model, base_lr=0.1)
        
        if opt is None:
            return False, "Function returned None"
        
        lr1 = opt.param_groups[0]['lr']
        lr2 = opt.param_groups[1]['lr']
        
        if lr1 == lr2:
            return False, f"Same lr for both groups: {lr1}"
        
        return True, f"Different lrs: {lr1}, {lr2}"
    except Exception as e:
        return False, str(e)


def test_sgd_with_state() -> Tuple[bool, str]:
    """Test SGDWithState."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt = SGDWithState(params, lr=0.01)
        
        if not hasattr(opt, 'state'):
            return False, "No state attribute"
        if not hasattr(opt, '_step_count'):
            return False, "No _step_count attribute"
        
        return True, "SGDWithState created"
    except Exception as e:
        return False, str(e)


def test_state_dict_save() -> Tuple[bool, str]:
    """Test state_dict generation."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt = SGDWithState(params, lr=0.01)
        
        for _ in range(5):
            for p in params:
                p.grad = np.random.randn(*p.shape)
            opt.step()
        
        state = opt.state_dict()
        
        if not state:
            return False, "state_dict is empty"
        
        return True, "state_dict created"
    except Exception as e:
        return False, str(e)


def test_state_dict_load() -> Tuple[bool, str]:
    """Test state_dict loading."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt1 = SGDWithState(params, lr=0.01)
        
        for _ in range(5):
            for p in params:
                p.grad = np.random.randn(*p.shape)
            opt1.step()
        
        state = opt1.state_dict()
        
        params2 = [Parameter(np.random.randn(3, 4))]
        opt2 = SGDWithState(params2, lr=0.05)
        opt2.load_state_dict(state)
        
        if opt2.param_groups[0]['lr'] != 0.01:
            return False, f"lr not restored: {opt2.param_groups[0]['lr']}"
        
        return True, "state_dict loaded"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test SGD against PyTorch implementation."""
    try:
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        our_model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        torch_model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        our_params = list(our_model.parameters())
        torch_model[0].weight.data = torch.tensor(our_params[0].data.copy())
        torch_model[0].bias.data = torch.tensor(our_params[1].data.copy())
        torch_model[2].weight.data = torch.tensor(our_params[2].data.copy())
        torch_model[2].bias.data = torch.tensor(our_params[3].data.copy())
        
        our_opt = SGD(our_model.parameters(), lr=0.01)
        torch_opt = torch.optim.SGD(torch_model.parameters(), lr=0.01)
        
        X = np.random.randn(4, 4)
        Y = np.random.randn(4, 2)
        
        our_opt.zero_grad()
        our_x = Tensor(X.copy())
        our_y = Tensor(Y.copy())
        our_pred = our_model(our_x)
        our_loss = mse_loss(our_pred, our_y)
        our_loss.backward()
        our_opt.step()
        
        torch_opt.zero_grad()
        torch_x = torch.tensor(X, dtype=torch.float64, requires_grad=True)
        torch_y = torch.tensor(Y, dtype=torch.float64)
        torch_pred = torch_model.double()(torch_x)
        torch_loss = nn.MSELoss()(torch_pred, torch_y)
        torch_loss.backward()
        torch_opt.step()
        
        our_weight = our_params[0].data
        torch_weight = torch_model[0].weight.data.numpy()
        
        if not np.allclose(our_weight, torch_weight, rtol=1e-4, atol=1e-4):
            return False, "Weights diverge from PyTorch"
        
        return True, "Matches PyTorch SGD"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("optimizer_init", test_optimizer_init),
        ("optimizer_defaults", test_optimizer_defaults),
        ("optimizer_param_groups", test_optimizer_param_groups),
        ("sgd_creation", test_sgd_creation),
        ("sgd_step_basic", test_sgd_step_basic),
        ("sgd_multiple_params", test_sgd_multiple_params),
        ("zero_grad_basic", test_zero_grad_basic),
        ("zero_grad_multiple", test_zero_grad_multiple),
        ("mse_loss", test_mse_loss),
        ("train_step_runs", test_train_step_runs),
        ("train_step_updates_params", test_train_step_updates_params),
        ("train_loop_returns_losses", test_train_loop_returns_losses),
        ("train_loop_convergence", test_train_loop_convergence),
        ("param_groups_creation", test_param_groups_creation),
        ("param_groups_different_lr", test_param_groups_different_lr),
        ("sgd_with_state", test_sgd_with_state),
        ("state_dict_save", test_state_dict_save),
        ("state_dict_load", test_state_dict_load),
        ("against_pytorch", test_against_pytorch),
    ]
    
    print(f"\n{'='*60}")
    print("Day 26: SGD Optimizer - Tests")
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
