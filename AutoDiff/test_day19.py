"""Test Suite for Day 19: Exp and Log Operations"""

import numpy as np
import sys
from typing import Tuple

try:
    from day19 import (
        Tensor,
        test_exp,
        test_log,
        test_exp_log_inverse,
        test_logsumexp,
        test_logsumexp_stability,
        test_sigmoid,
        test_sigmoid_extreme,
        test_tanh,
        test_softplus,
        test_chain_rule
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_exp_forward() -> Tuple[bool, str]:
    """Test exp forward pass."""
    try:
        x = Tensor([0.0, 1.0, 2.0])
        y = x.exp()
        
        if y is None or y.data is None:
            return False, "Returned None"
        expected = np.array([1, np.e, np.e**2])
        if not np.allclose(y.data, expected):
            return False, f"values = {y.data}"
        return True, "exp([0,1,2]) correct"
    except Exception as e:
        return False, str(e)


def test_exp_backward() -> Tuple[bool, str]:
    """Test exp backward pass."""
    try:
        x = Tensor([0.0, 1.0, 2.0])
        y = x.exp()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        expected = np.array([1, np.e, np.e**2])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "d(exp)/dx = exp"
    except Exception as e:
        return False, str(e)


def test_exp_chain() -> Tuple[bool, str]:
    """Test exp in chain rule."""
    try:
        x = Tensor([1.0, 2.0])
        y = (x * 2).exp().sum()  # sum(exp(2x))
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        # d/dx = 2 * exp(2x)
        expected = 2 * np.exp(2 * x.data)
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "Chain rule OK"
    except Exception as e:
        return False, str(e)


def test_log_forward() -> Tuple[bool, str]:
    """Test log forward pass."""
    try:
        x = Tensor([1.0, np.e, np.e**2])
        y = x.log()
        
        if y is None or y.data is None:
            return False, "Returned None"
        expected = np.array([0, 1, 2])
        if not np.allclose(y.data, expected):
            return False, f"values = {y.data}"
        return True, "log([1,e,e²]) = [0,1,2]"
    except Exception as e:
        return False, str(e)


def test_log_backward() -> Tuple[bool, str]:
    """Test log backward pass."""
    try:
        x = Tensor([1.0, 2.0, 4.0])
        y = x.log()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        expected = np.array([1, 0.5, 0.25])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "d(log)/dx = 1/x"
    except Exception as e:
        return False, str(e)


def test_exp_log_identity() -> Tuple[bool, str]:
    """Test exp(log(x)) = x."""
    try:
        x = Tensor([1.0, 2.0, 3.0])
        y = x.log().exp()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, x.data):
            return False, f"exp(log(x)) = {y.data}"
        return True, "exp(log(x)) = x"
    except Exception as e:
        return False, str(e)


def test_log_exp_identity() -> Tuple[bool, str]:
    """Test log(exp(x)) = x."""
    try:
        x = Tensor([1.0, 2.0, 3.0])
        y = x.exp().log()
        y.backward()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, x.data):
            return False, f"log(exp(x)) = {y.data}"
        if not np.allclose(x.grad, np.ones(3)):
            return False, f"grad = {x.grad}"
        return True, "log(exp(x)) = x, grad=1"
    except Exception as e:
        return False, str(e)


def test_logsumexp_forward() -> Tuple[bool, str]:
    """Test logsumexp forward."""
    try:
        x = Tensor([1.0, 2.0, 3.0])
        y = x.logsumexp()
        
        if y is None or y.data is None:
            return False, "Returned None"
        expected = np.log(np.sum(np.exp(x.data)))
        if not np.allclose(y.data, expected):
            return False, f"logsumexp = {y.data}"
        return True, "logsumexp correct"
    except Exception as e:
        return False, str(e)


def test_logsumexp_backward() -> Tuple[bool, str]:
    """Test logsumexp backward is softmax."""
    try:
        x = Tensor([1.0, 2.0, 3.0])
        y = x.logsumexp()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        softmax = np.exp(x.data) / np.sum(np.exp(x.data))
        if not np.allclose(x.grad, softmax):
            return False, f"grad = {x.grad}"
        return True, "d(logsumexp)/dx = softmax"
    except Exception as e:
        return False, str(e)


def test_logsumexp_large_values() -> Tuple[bool, str]:
    """Test logsumexp doesn't overflow."""
    try:
        x = Tensor([1000.0, 1000.0, 1000.0])
        y = x.logsumexp()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.isfinite(y.data):
            return False, "Result is not finite"
        # Should be 1000 + log(3)
        expected = 1000 + np.log(3)
        if not np.allclose(y.data, expected):
            return False, f"logsumexp = {y.data}"
        return True, "No overflow"
    except Exception as e:
        return False, str(e)


def test_logsumexp_small_values() -> Tuple[bool, str]:
    """Test logsumexp doesn't underflow."""
    try:
        x = Tensor([-1000.0, -1000.0, -1000.0])
        y = x.logsumexp()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.isfinite(y.data):
            return False, "Result is not finite"
        # Should be -1000 + log(3)
        expected = -1000 + np.log(3)
        if not np.allclose(y.data, expected):
            return False, f"logsumexp = {y.data}"
        return True, "No underflow"
    except Exception as e:
        return False, str(e)


def test_logsumexp_axis() -> Tuple[bool, str]:
    """Test logsumexp along axis."""
    try:
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = x.logsumexp(axis=1)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (2,):
            return False, f"shape = {y.shape}"
        return True, "logsumexp axis works"
    except Exception as e:
        return False, str(e)


def test_sigmoid_forward() -> Tuple[bool, str]:
    """Test sigmoid forward."""
    try:
        x = Tensor([-1.0, 0.0, 1.0])
        y = x.sigmoid()
        
        if y is None or y.data is None:
            return False, "Returned None"
        expected = 1 / (1 + np.exp(-x.data))
        if not np.allclose(y.data, expected):
            return False, f"sigmoid = {y.data}"
        return True, "sigmoid correct"
    except Exception as e:
        return False, str(e)


def test_sigmoid_backward() -> Tuple[bool, str]:
    """Test sigmoid backward."""
    try:
        x = Tensor([-1.0, 0.0, 1.0])
        y = x.sigmoid()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        sig = 1 / (1 + np.exp(-x.data))
        expected = sig * (1 - sig)
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "d(sigmoid)/dx = σ(1-σ)"
    except Exception as e:
        return False, str(e)


def test_sigmoid_stability_positive() -> Tuple[bool, str]:
    """Test sigmoid stable for large positive."""
    try:
        x = Tensor([100.0, 500.0, 700.0])
        y = x.sigmoid()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.all(np.isfinite(y.data)):
            return False, "Not finite"
        if not np.allclose(y.data, 1.0, atol=1e-10):
            return False, f"sigmoid = {y.data}"
        return True, "sigmoid(large) ≈ 1"
    except Exception as e:
        return False, str(e)


def test_sigmoid_stability_negative() -> Tuple[bool, str]:
    """Test sigmoid stable for large negative."""
    try:
        x = Tensor([-100.0, -500.0, -700.0])
        y = x.sigmoid()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.all(np.isfinite(y.data)):
            return False, "Not finite"
        if not np.allclose(y.data, 0.0, atol=1e-10):
            return False, f"sigmoid = {y.data}"
        return True, "sigmoid(-large) ≈ 0"
    except Exception as e:
        return False, str(e)


def test_tanh_forward() -> Tuple[bool, str]:
    """Test tanh forward."""
    try:
        x = Tensor([-1.0, 0.0, 1.0])
        y = x.tanh()
        
        if y is None or y.data is None:
            return False, "Returned None"
        expected = np.tanh(x.data)
        if not np.allclose(y.data, expected):
            return False, f"tanh = {y.data}"
        return True, "tanh correct"
    except Exception as e:
        return False, str(e)


def test_tanh_backward() -> Tuple[bool, str]:
    """Test tanh backward."""
    try:
        x = Tensor([-1.0, 0.0, 1.0])
        y = x.tanh()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        tanh_val = np.tanh(x.data)
        expected = 1 - tanh_val ** 2
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "d(tanh)/dx = 1-tanh²"
    except Exception as e:
        return False, str(e)


def test_tanh_range() -> Tuple[bool, str]:
    """Test tanh output in (-1, 1)."""
    try:
        x = Tensor(np.linspace(-10, 10, 100))
        y = x.tanh()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.all((y.data > -1) & (y.data < 1)):
            return False, "Out of range"
        return True, "tanh ∈ (-1, 1)"
    except Exception as e:
        return False, str(e)


def test_softplus_forward() -> Tuple[bool, str]:
    """Test softplus forward."""
    try:
        x = Tensor([-2.0, 0.0, 2.0])
        y = x.softplus()
        
        if y is None or y.data is None:
            return False, "Returned None"
        expected = np.log(1 + np.exp(x.data))
        if not np.allclose(y.data, expected):
            return False, f"softplus = {y.data}"
        return True, "softplus correct"
    except Exception as e:
        return False, str(e)


def test_softplus_backward() -> Tuple[bool, str]:
    """Test softplus backward is sigmoid."""
    try:
        x = Tensor([-2.0, 0.0, 2.0])
        y = x.softplus()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        expected = 1 / (1 + np.exp(-x.data))
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "d(softplus)/dx = sigmoid"
    except Exception as e:
        return False, str(e)


def test_softplus_relu_approx() -> Tuple[bool, str]:
    """Test softplus approximates ReLU."""
    try:
        x = Tensor([5.0, 10.0, 20.0])
        y = x.softplus()
        
        if y is None or y.data is None:
            return False, "Returned None"
        # For large x, softplus(x) ≈ x
        if not np.allclose(y.data, x.data, rtol=0.01):
            return False, f"softplus = {y.data}"
        return True, "softplus(x) ≈ x for large x"
    except Exception as e:
        return False, str(e)


def test_log_sigmoid_forward() -> Tuple[bool, str]:
    """Test log_sigmoid forward."""
    try:
        x = Tensor([-1.0, 0.0, 1.0])
        if not hasattr(Tensor, 'log_sigmoid'):
            return True, "log_sigmoid not implemented (optional)"
        y = x.log_sigmoid()
        
        if y is None or y.data is None:
            return False, "Returned None"
        expected = np.log(1 / (1 + np.exp(-x.data)))
        if not np.allclose(y.data, expected):
            return False, f"log_sigmoid = {y.data}"
        return True, "log_sigmoid correct"
    except Exception as e:
        return False, str(e)


def test_log_sigmoid_stability() -> Tuple[bool, str]:
    """Test log_sigmoid stability."""
    try:
        if not hasattr(Tensor, 'log_sigmoid'):
            return True, "log_sigmoid not implemented (optional)"
        x = Tensor([-100.0, 0.0, 100.0])
        y = x.log_sigmoid()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.all(np.isfinite(y.data)):
            return False, "Not finite"
        return True, "log_sigmoid stable"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test against PyTorch reference."""
    try:
        import torch
        
        np_x = np.array([0.5, 1.0, 2.0])
        
        # Our exp
        x = Tensor(np_x)
        y = x.exp()
        y.backward()
        
        # PyTorch exp
        tx = torch.tensor(np_x, requires_grad=True)
        ty = tx.exp()
        ty.backward(torch.ones_like(ty))
        
        if not np.allclose(y.data, ty.detach().numpy()):
            return False, "Forward mismatch"
        if not np.allclose(x.grad, tx.grad.numpy()):
            return False, "Gradient mismatch"
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def test_sigmoid_tanh_relation() -> Tuple[bool, str]:
    """Test tanh(x) = 2*sigmoid(2x) - 1."""
    try:
        x = Tensor(np.linspace(-3, 3, 10))
        tanh_val = x.tanh()
        
        # 2*sigmoid(2x) - 1
        x2 = Tensor(2 * x.data)
        sig_val = x2.sigmoid()
        
        if tanh_val is None or sig_val is None:
            return False, "Returned None"
        
        reconstructed = 2 * sig_val.data - 1
        if not np.allclose(tanh_val.data, reconstructed):
            return False, "Relation doesn't hold"
        return True, "tanh = 2σ(2x) - 1"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("exp_forward", test_exp_forward),
        ("exp_backward", test_exp_backward),
        ("exp_chain", test_exp_chain),
        ("log_forward", test_log_forward),
        ("log_backward", test_log_backward),
        ("exp_log_identity", test_exp_log_identity),
        ("log_exp_identity", test_log_exp_identity),
        ("logsumexp_forward", test_logsumexp_forward),
        ("logsumexp_backward", test_logsumexp_backward),
        ("logsumexp_large_values", test_logsumexp_large_values),
        ("logsumexp_small_values", test_logsumexp_small_values),
        ("logsumexp_axis", test_logsumexp_axis),
        ("sigmoid_forward", test_sigmoid_forward),
        ("sigmoid_backward", test_sigmoid_backward),
        ("sigmoid_stability_positive", test_sigmoid_stability_positive),
        ("sigmoid_stability_negative", test_sigmoid_stability_negative),
        ("tanh_forward", test_tanh_forward),
        ("tanh_backward", test_tanh_backward),
        ("tanh_range", test_tanh_range),
        ("softplus_forward", test_softplus_forward),
        ("softplus_backward", test_softplus_backward),
        ("softplus_relu_approx", test_softplus_relu_approx),
        ("log_sigmoid_forward", test_log_sigmoid_forward),
        ("log_sigmoid_stability", test_log_sigmoid_stability),
        ("against_pytorch", test_against_pytorch),
        ("sigmoid_tanh_relation", test_sigmoid_tanh_relation),
    ]
    
    print(f"\n{'='*50}\nDay 19: Exp and Log Operations - Tests\n{'='*50}")
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    run_all_tests()
