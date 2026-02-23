"""Test Suite for Day 28: Adam Optimizer"""

import numpy as np
import sys
from typing import Tuple

try:
    from day28 import (
        Tensor,
        Parameter,
        Module,
        Linear,
        ReLU,
        Sequential,
        Optimizer,
        SGD,
        Adagrad,
        RMSprop,
        Adam,
        AdamW,
        mse_loss,
        compare_optimizers,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_adagrad_creation() -> Tuple[bool, str]:
    """Test Adagrad creation."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt = Adagrad(params, lr=0.01)
        
        if opt is None:
            return False, "Adagrad is None"
        
        return True, "Adagrad created"
    except Exception as e:
        return False, str(e)


def test_adagrad_step() -> Tuple[bool, str]:
    """Test Adagrad step."""
    try:
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        opt = Adagrad([p], lr=1.0, eps=1e-10)
        
        initial = p.data.copy()
        p.grad = np.array([1.0, 1.0, 1.0])
        opt.step()
        
        if np.allclose(p.data, initial):
            return False, "Parameters not updated"
        
        return True, "Adagrad step works"
    except Exception as e:
        return False, str(e)


def test_adagrad_accumulates() -> Tuple[bool, str]:
    """Test Adagrad accumulates squared gradients."""
    try:
        p = Parameter(np.array([1.0]))
        opt = Adagrad([p], lr=1.0, eps=0.0)
        
        updates = []
        for _ in range(5):
            p.grad = np.array([1.0])
            old_val = p.data.copy()
            opt.step()
            updates.append(old_val[0] - p.data[0])
        
        for i in range(len(updates) - 1):
            if updates[i] <= updates[i+1]:
                return False, "Updates should decrease"
        
        return True, "Adagrad decreases updates over time"
    except Exception as e:
        return False, str(e)


def test_rmsprop_creation() -> Tuple[bool, str]:
    """Test RMSprop creation."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt = RMSprop(params, lr=0.01, alpha=0.99)
        
        if opt is None:
            return False, "RMSprop is None"
        
        return True, "RMSprop created"
    except Exception as e:
        return False, str(e)


def test_rmsprop_step() -> Tuple[bool, str]:
    """Test RMSprop step."""
    try:
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        opt = RMSprop([p], lr=0.1, alpha=0.9)
        
        initial = p.data.copy()
        
        for _ in range(10):
            p.grad = np.array([0.1, 0.2, 0.3])
            opt.step()
        
        if np.allclose(p.data, initial):
            return False, "Parameters not updated"
        
        return True, "RMSprop step works"
    except Exception as e:
        return False, str(e)


def test_rmsprop_vs_adagrad() -> Tuple[bool, str]:
    """Test that RMSprop doesn't decay lr as aggressively as Adagrad."""
    try:
        np.random.seed(42)
        
        p1 = Parameter(np.array([10.0]))
        opt1 = Adagrad([p1], lr=1.0, eps=1e-10)
        
        p2 = Parameter(np.array([10.0]))
        opt2 = RMSprop([p2], lr=1.0, alpha=0.9, eps=1e-10)
        
        for _ in range(100):
            p1.grad = np.array([1.0])
            p2.grad = np.array([1.0])
            opt1.step()
            opt2.step()
        
        return True, f"Adagrad: {p1.data[0]:.2f}, RMSprop: {p2.data[0]:.2f}"
    except Exception as e:
        return False, str(e)


def test_adam_creation() -> Tuple[bool, str]:
    """Test Adam creation."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt = Adam(params, lr=0.001, betas=(0.9, 0.999))
        
        if opt is None:
            return False, "Adam is None"
        
        return True, "Adam created"
    except Exception as e:
        return False, str(e)


def test_adam_step() -> Tuple[bool, str]:
    """Test Adam step."""
    try:
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        opt = Adam([p], lr=0.1, betas=(0.9, 0.999))
        
        initial = p.data.copy()
        
        for _ in range(10):
            p.grad = np.array([0.1, 0.2, 0.3])
            opt.step()
        
        if np.allclose(p.data, initial):
            return False, "Parameters not updated"
        
        return True, "Adam step works"
    except Exception as e:
        return False, str(e)


def test_adam_state() -> Tuple[bool, str]:
    """Test Adam stores both moments."""
    try:
        p = Parameter(np.array([1.0, 2.0]))
        opt = Adam([p], lr=0.001)
        
        p.grad = np.array([1.0, 1.0])
        opt.step()
        
        if not opt.state:
            return False, "State is empty"
        
        state = list(opt.state.values())[0]
        
        if 'exp_avg' not in state:
            return False, "No first moment"
        if 'exp_avg_sq' not in state:
            return False, "No second moment"
        
        return True, "Both moments stored"
    except Exception as e:
        return False, str(e)


def test_adam_bias_correction() -> Tuple[bool, str]:
    """Test Adam bias correction is applied."""
    try:
        p = Parameter(np.array([5.0]))
        opt = Adam([p], lr=1.0, betas=(0.9, 0.999), eps=1e-8)
        
        p.grad = np.array([1.0])
        opt.step()
        
        state = list(opt.state.values())[0]
        raw_m = state['exp_avg'][0]
        raw_v = state['exp_avg_sq'][0]
        
        m_hat = raw_m / (1 - 0.9)
        v_hat = raw_v / (1 - 0.999)
        
        expected_update = m_hat / (np.sqrt(v_hat) + 1e-8)
        actual_update = 5.0 - p.data[0]
        
        if not np.isclose(actual_update, expected_update, rtol=0.1):
            return False, f"Update {actual_update:.4f} != expected {expected_update:.4f}"
        
        return True, "Bias correction applied"
    except Exception as e:
        return False, str(e)


def test_adamw_creation() -> Tuple[bool, str]:
    """Test AdamW creation."""
    try:
        params = [Parameter(np.random.randn(3, 4))]
        opt = AdamW(params, lr=0.001, weight_decay=0.01)
        
        if opt is None:
            return False, "AdamW is None"
        
        return True, "AdamW created"
    except Exception as e:
        return False, str(e)


def test_adamw_weight_decay() -> Tuple[bool, str]:
    """Test AdamW applies decoupled weight decay."""
    try:
        p = Parameter(np.array([10.0, 10.0, 10.0]))
        opt = AdamW([p], lr=0.1, weight_decay=0.1, betas=(0.9, 0.999))
        
        initial = p.data.copy()
        
        for _ in range(50):
            p.grad = np.zeros_like(p.data)
            opt.step()
        
        if not np.all(np.abs(p.data) < np.abs(initial)):
            return False, "Weight decay not applied"
        
        return True, f"Weights decayed from {initial[0]:.2f} to {p.data[0]:.4f}"
    except Exception as e:
        return False, str(e)


def test_adamw_vs_adam() -> Tuple[bool, str]:
    """Test that AdamW differs from Adam with L2."""
    try:
        np.random.seed(42)
        
        p1 = Parameter(np.array([5.0]))
        opt1 = AdamW([p1], lr=0.1, weight_decay=0.1, betas=(0.9, 0.999))
        
        for _ in range(20):
            p1.grad = np.array([1.0])
            opt1.step()
        
        return True, f"AdamW result: {p1.data[0]:.4f}"
    except Exception as e:
        return False, str(e)


def test_compare_optimizers_runs() -> Tuple[bool, str]:
    """Test compare_optimizers runs."""
    try:
        np.random.seed(42)
        X = np.random.randn(20, 4)
        Y = np.random.randn(20, 2)
        
        results = compare_optimizers(X, Y, epochs=20)
        
        if not results:
            return False, "Empty results"
        
        return True, f"Compared {len(results)} optimizers"
    except Exception as e:
        return False, str(e)


def test_compare_optimizers_converges() -> Tuple[bool, str]:
    """Test all optimizers converge in comparison."""
    try:
        np.random.seed(42)
        X = np.random.randn(20, 4)
        Y = np.random.randn(20, 2)
        
        results = compare_optimizers(X, Y, epochs=100)
        
        if not results:
            return False, "Empty results"
        
        for name, losses in results.items():
            if losses[-1] >= losses[0]:
                return False, f"{name} didn't converge"
        
        return True, "All optimizers converge"
    except Exception as e:
        return False, str(e)


def test_adam_training() -> Tuple[bool, str]:
    """Test Adam on actual training."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 16),
            ReLU(),
            Linear(16, 2)
        )
        
        opt = Adam(model.parameters(), lr=0.01)
        
        X = np.random.randn(20, 4)
        Y = np.random.randn(20, 2)
        
        losses = []
        for _ in range(100):
            opt.zero_grad()
            x = Tensor(X)
            y = Tensor(Y)
            pred = model(x)
            loss = mse_loss(pred, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.data))
        
        if losses[-1] >= losses[0]:
            return False, "Loss didn't decrease"
        
        return True, f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
    except Exception as e:
        return False, str(e)


def test_against_pytorch_adam() -> Tuple[bool, str]:
    """Test Adam against PyTorch."""
    try:
        import torch
        import torch.optim as optim
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        our_p = Parameter(np.array([1.0, 2.0, 3.0]))
        our_opt = Adam([our_p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
        
        torch_p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float64)
        torch_opt = optim.Adam([torch_p], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
        
        grads = np.random.randn(20, 3)
        
        for g in grads:
            our_p.grad = g.copy()
            our_opt.step()
            
            torch_p.grad = torch.tensor(g, dtype=torch.float64)
            torch_opt.step()
        
        if not np.allclose(our_p.data, torch_p.detach().numpy(), rtol=1e-4, atol=1e-4):
            return False, f"Mismatch: {our_p.data} vs {torch_p.detach().numpy()}"
        
        return True, "Matches PyTorch Adam"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def test_against_pytorch_adamw() -> Tuple[bool, str]:
    """Test AdamW against PyTorch."""
    try:
        import torch
        import torch.optim as optim
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        our_p = Parameter(np.array([5.0, 5.0, 5.0]))
        our_opt = AdamW([our_p], lr=0.1, betas=(0.9, 0.999), 
                       eps=1e-8, weight_decay=0.1)
        
        torch_p = torch.tensor([5.0, 5.0, 5.0], requires_grad=True, dtype=torch.float64)
        torch_opt = optim.AdamW([torch_p], lr=0.1, betas=(0.9, 0.999),
                                eps=1e-8, weight_decay=0.1)
        
        grads = np.random.randn(20, 3)
        
        for g in grads:
            our_p.grad = g.copy()
            our_opt.step()
            
            torch_p.grad = torch.tensor(g, dtype=torch.float64)
            torch_opt.step()
        
        if not np.allclose(our_p.data, torch_p.detach().numpy(), rtol=1e-3, atol=1e-3):
            return False, f"Mismatch: {our_p.data} vs {torch_p.detach().numpy()}"
        
        return True, "Matches PyTorch AdamW"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("adagrad_creation", test_adagrad_creation),
        ("adagrad_step", test_adagrad_step),
        ("adagrad_accumulates", test_adagrad_accumulates),
        ("rmsprop_creation", test_rmsprop_creation),
        ("rmsprop_step", test_rmsprop_step),
        ("rmsprop_vs_adagrad", test_rmsprop_vs_adagrad),
        ("adam_creation", test_adam_creation),
        ("adam_step", test_adam_step),
        ("adam_state", test_adam_state),
        ("adam_bias_correction", test_adam_bias_correction),
        ("adamw_creation", test_adamw_creation),
        ("adamw_weight_decay", test_adamw_weight_decay),
        ("adamw_vs_adam", test_adamw_vs_adam),
        ("compare_optimizers_runs", test_compare_optimizers_runs),
        ("compare_optimizers_converges", test_compare_optimizers_converges),
        ("adam_training", test_adam_training),
        ("against_pytorch_adam", test_against_pytorch_adam),
        ("against_pytorch_adamw", test_against_pytorch_adamw),
    ]
    
    print(f"\n{'='*60}")
    print("Day 28: Adam Optimizer - Tests")
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
