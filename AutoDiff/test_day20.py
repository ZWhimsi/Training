"""Test Suite for Day 20: Softmax Implementation"""

import numpy as np
import sys
from typing import Tuple

try:
    from day20 import (
        Tensor,
        test_softmax_basic,
        test_softmax_stability,
        test_softmax_gradient,
        test_softmax_batch,
        test_log_softmax,
        test_log_softmax_stability,
        test_cross_entropy,
        test_cross_entropy_gradient,
        test_cross_entropy_onehot,
        test_softmax_temperature
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_softmax_sum_one() -> Tuple[bool, str]:
    """Test softmax sums to 1."""
    try:
        x = Tensor([[1.0, 2.0, 3.0, 4.0]])
        y = x.softmax()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(np.sum(y.data), 1.0):
            return False, f"sum = {np.sum(y.data)}"
        return True, "sum(softmax) = 1"
    except Exception as e:
        return False, str(e)


def test_softmax_positive() -> Tuple[bool, str]:
    """Test softmax outputs are positive."""
    try:
        x = Tensor([[-10.0, 0.0, 10.0]])
        y = x.softmax()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.all(y.data > 0):
            return False, "Not all positive"
        return True, "All outputs > 0"
    except Exception as e:
        return False, str(e)


def test_softmax_invariance() -> Tuple[bool, str]:
    """Test softmax shift invariance."""
    try:
        x1 = Tensor([[1.0, 2.0, 3.0]])
        x2 = Tensor([[101.0, 102.0, 103.0]])  # Shifted by 100
        
        y1 = x1.softmax()
        y2 = x2.softmax()
        
        if y1 is None or y2 is None:
            return False, "Returned None"
        if not np.allclose(y1.data, y2.data):
            return False, "Not shift invariant"
        return True, "softmax(x+c) = softmax(x)"
    except Exception as e:
        return False, str(e)


def test_softmax_numerical_large() -> Tuple[bool, str]:
    """Test softmax with large values."""
    try:
        x = Tensor([[1000.0, 1000.0, 1000.0]])
        y = x.softmax()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.all(np.isfinite(y.data)):
            return False, "Not finite"
        if not np.allclose(y.data, [[1/3, 1/3, 1/3]]):
            return False, f"values = {y.data}"
        return True, "Large values OK"
    except Exception as e:
        return False, str(e)


def test_softmax_numerical_small() -> Tuple[bool, str]:
    """Test softmax with very negative values."""
    try:
        x = Tensor([[-1000.0, -1000.0, -1000.0]])
        y = x.softmax()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.all(np.isfinite(y.data)):
            return False, "Not finite"
        if not np.allclose(y.data, [[1/3, 1/3, 1/3]]):
            return False, f"values = {y.data}"
        return True, "Small values OK"
    except Exception as e:
        return False, str(e)


def test_softmax_gradient_jvp() -> Tuple[bool, str]:
    """Test softmax Jacobian-vector product."""
    try:
        x = Tensor([[1.0, 2.0, 3.0]])
        y = x.softmax()
        loss = (y * Tensor([[1.0, 0.0, 0.0]])).sum()  # Focus on first element
        loss.backward()
        
        if x.grad is None:
            return False, "grad is None"
        
        # Numerical gradient check
        eps = 1e-5
        numerical_grad = np.zeros_like(x.data)
        for i in range(3):
            x_plus = x.data.copy()
            x_plus[0, i] += eps
            x_minus = x.data.copy()
            x_minus[0, i] -= eps
            
            y_plus = np.exp(x_plus) / np.sum(np.exp(x_plus))
            y_minus = np.exp(x_minus) / np.sum(np.exp(x_minus))
            numerical_grad[0, i] = (y_plus[0, 0] - y_minus[0, 0]) / (2 * eps)
        
        if not np.allclose(x.grad, numerical_grad, rtol=1e-4):
            return False, f"grad = {x.grad}"
        return True, "JVP matches numerical"
    except Exception as e:
        return False, str(e)


def test_softmax_batch() -> Tuple[bool, str]:
    """Test softmax on batched input."""
    try:
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        y = x.softmax(axis=-1)
        
        if y is None or y.data is None:
            return False, "Returned None"
        
        row_sums = np.sum(y.data, axis=1)
        if not np.allclose(row_sums, [1, 1, 1]):
            return False, f"row sums = {row_sums}"
        return True, "Batch softmax OK"
    except Exception as e:
        return False, str(e)


def test_log_softmax_values() -> Tuple[bool, str]:
    """Test log_softmax values."""
    try:
        x = Tensor([[1.0, 2.0, 3.0]])
        y = x.log_softmax()
        
        if y is None or y.data is None:
            return False, "Returned None"
        
        # exp(log_softmax) should be softmax
        expected_softmax = np.exp(x.data) / np.sum(np.exp(x.data))
        if not np.allclose(np.exp(y.data), expected_softmax):
            return False, "exp(log_softmax) != softmax"
        return True, "log_softmax correct"
    except Exception as e:
        return False, str(e)


def test_log_softmax_gradient() -> Tuple[bool, str]:
    """Test log_softmax gradient."""
    try:
        x = Tensor([[1.0, 2.0, 3.0]])
        y = x.log_softmax()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        
        # Gradient: I - softmax (for each row)
        softmax = np.exp(y.data)
        # With upstream = ones, grad = 1 - softmax * sum(1) = 1 - softmax * 3
        expected = 1 - softmax * 3
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "log_softmax gradient OK"
    except Exception as e:
        return False, str(e)


def test_log_softmax_extreme() -> Tuple[bool, str]:
    """Test log_softmax with extreme values."""
    try:
        x = Tensor([[0.0, 0.0, -1000.0]])  # Third class very unlikely
        y = x.log_softmax()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.all(np.isfinite(y.data)):
            return False, "Not finite"
        # Third value should be very negative
        if y.data[0, 2] > -500:
            return False, f"Third value = {y.data[0, 2]}"
        return True, "Extreme values handled"
    except Exception as e:
        return False, str(e)


def test_cross_entropy_forward() -> Tuple[bool, str]:
    """Test cross_entropy forward."""
    try:
        logits = Tensor([[1.0, 2.0, 3.0]])
        targets = np.array([2])
        
        loss = logits.cross_entropy(targets)
        
        if loss is None or loss.data is None:
            return False, "Returned None"
        
        # Manual calculation
        log_sm = logits.data - np.log(np.sum(np.exp(logits.data)))
        expected = -log_sm[0, 2]
        
        if not np.allclose(loss.data, expected):
            return False, f"loss = {loss.data}"
        return True, "CE forward OK"
    except Exception as e:
        return False, str(e)


def test_cross_entropy_gradient_simple() -> Tuple[bool, str]:
    """Test cross_entropy gradient (softmax - one_hot)."""
    try:
        logits = Tensor([[1.0, 2.0, 3.0]])
        targets = np.array([1])  # Target is class 1
        
        loss = logits.cross_entropy(targets)
        loss.backward()
        
        if logits.grad is None:
            return False, "grad is None"
        
        softmax = np.exp(logits.data) / np.sum(np.exp(logits.data))
        one_hot = np.array([[0, 1, 0]])
        expected = softmax - one_hot
        
        if not np.allclose(logits.grad, expected):
            return False, f"grad = {logits.grad}"
        return True, "CE gradient = softmax - one_hot"
    except Exception as e:
        return False, str(e)


def test_cross_entropy_batch() -> Tuple[bool, str]:
    """Test cross_entropy on batch."""
    try:
        logits = Tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])  # (2, 3)
        targets = np.array([2, 0])
        
        loss = logits.cross_entropy(targets)
        
        if loss is None or loss.data is None:
            return False, "Returned None"
        if loss.shape != (2,):
            return False, f"shape = {loss.shape}"
        
        # First sample: target is 2, which has highest logit
        # Second sample: target is 0, which also has highest logit
        # So losses should be relatively small
        if not np.all(loss.data < 2):
            return False, f"loss = {loss.data}"
        return True, "Batch CE OK"
    except Exception as e:
        return False, str(e)


def test_cross_entropy_perfect() -> Tuple[bool, str]:
    """Test CE with perfect predictions."""
    try:
        # One-hot like logits (very confident)
        logits = Tensor([[0.0, 0.0, 100.0]])
        targets = np.array([2])
        
        loss = logits.cross_entropy(targets)
        
        if loss is None or loss.data is None:
            return False, "Returned None"
        # Loss should be very small
        if not loss.data < 1e-10:
            return False, f"loss = {loss.data}"
        return True, "Perfect prediction, loss ≈ 0"
    except Exception as e:
        return False, str(e)


def test_cross_entropy_wrong() -> Tuple[bool, str]:
    """Test CE with wrong predictions."""
    try:
        # Confident but wrong
        logits = Tensor([[100.0, 0.0, 0.0]])
        targets = np.array([2])  # Target is class 2, but we predict class 0
        
        loss = logits.cross_entropy(targets)
        
        if loss is None or loss.data is None:
            return False, "Returned None"
        # Loss should be large (approximately 100)
        if not loss.data > 50:
            return False, f"loss = {loss.data}"
        return True, "Wrong prediction, loss is large"
    except Exception as e:
        return False, str(e)


def test_cross_entropy_onehot() -> Tuple[bool, str]:
    """Test CE with one-hot targets."""
    try:
        logits = Tensor([[1.0, 2.0, 3.0]])
        targets_idx = np.array([2])
        targets_onehot = np.array([[0, 0, 1]])
        
        loss_idx = logits.cross_entropy(targets_idx)
        loss_onehot = logits.cross_entropy_onehot(targets_onehot)
        
        if loss_idx is None or loss_onehot is None:
            return False, "Returned None"
        if not np.allclose(loss_idx.data, loss_onehot.data):
            return False, f"idx={loss_idx.data}, onehot={loss_onehot.data}"
        return True, "One-hot matches index"
    except Exception as e:
        return False, str(e)


def test_softmax_temperature_t1() -> Tuple[bool, str]:
    """Test temperature=1 equals standard softmax."""
    try:
        x = Tensor([[1.0, 2.0, 3.0]])
        y1 = x.softmax()
        y_temp = x.softmax_temperature(temperature=1.0)
        
        if y1 is None or y_temp is None:
            return False, "Returned None"
        if not np.allclose(y1.data, y_temp.data):
            return False, "T=1 doesn't match softmax"
        return True, "T=1 matches softmax"
    except Exception as e:
        return False, str(e)


def test_softmax_temperature_high() -> Tuple[bool, str]:
    """Test high temperature gives more uniform."""
    try:
        x = Tensor([[1.0, 2.0, 3.0]])
        y_low = x.softmax_temperature(temperature=0.5)
        y_high = x.softmax_temperature(temperature=5.0)
        
        if y_low is None or y_high is None:
            return False, "Returned None"
        
        # High temp should be more uniform (smaller max)
        if not np.max(y_high.data) < np.max(y_low.data):
            return False, "High T not more uniform"
        return True, "High T more uniform"
    except Exception as e:
        return False, str(e)


def test_softmax_temperature_gradient() -> Tuple[bool, str]:
    """Test temperature softmax gradient."""
    try:
        x = Tensor([[1.0, 2.0, 3.0]])
        y = x.softmax_temperature(temperature=2.0)
        loss = y.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "grad is None"
        # Gradient should sum to 0
        if not np.allclose(np.sum(x.grad), 0, atol=1e-10):
            return False, f"grad sum = {np.sum(x.grad)}"
        return True, "Temperature gradient OK"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test against PyTorch reference."""
    try:
        import torch
        import torch.nn.functional as F
        
        np_x = np.array([[1.0, 2.0, 3.0]])
        
        # Our softmax
        x = Tensor(np_x)
        y = x.softmax()
        
        # PyTorch softmax
        tx = torch.tensor(np_x, requires_grad=True)
        ty = F.softmax(tx, dim=-1)
        
        if not np.allclose(y.data, ty.detach().numpy()):
            return False, "Forward mismatch"
        
        # Test gradient
        ty.backward(torch.ones_like(ty))
        y.backward()
        
        if not np.allclose(x.grad, tx.grad.numpy()):
            return False, "Gradient mismatch"
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def test_ce_pytorch() -> Tuple[bool, str]:
    """Test cross_entropy against PyTorch."""
    try:
        import torch
        import torch.nn.functional as F
        
        np_x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        targets = np.array([2, 0])
        
        # Our CE
        x = Tensor(np_x)
        loss = x.cross_entropy(targets)
        
        # PyTorch CE
        tx = torch.tensor(np_x, requires_grad=True)
        t_targets = torch.tensor(targets)
        t_loss = F.cross_entropy(tx, t_targets, reduction='none')
        
        if not np.allclose(loss.data, t_loss.detach().numpy()):
            return False, f"Forward: ours={loss.data}, torch={t_loss.detach().numpy()}"
        return True, "CE matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("softmax_sum_one", test_softmax_sum_one),
        ("softmax_positive", test_softmax_positive),
        ("softmax_invariance", test_softmax_invariance),
        ("softmax_numerical_large", test_softmax_numerical_large),
        ("softmax_numerical_small", test_softmax_numerical_small),
        ("softmax_gradient_jvp", test_softmax_gradient_jvp),
        ("softmax_batch", test_softmax_batch),
        ("log_softmax_values", test_log_softmax_values),
        ("log_softmax_gradient", test_log_softmax_gradient),
        ("log_softmax_extreme", test_log_softmax_extreme),
        ("cross_entropy_forward", test_cross_entropy_forward),
        ("cross_entropy_gradient_simple", test_cross_entropy_gradient_simple),
        ("cross_entropy_batch", test_cross_entropy_batch),
        ("cross_entropy_perfect", test_cross_entropy_perfect),
        ("cross_entropy_wrong", test_cross_entropy_wrong),
        ("cross_entropy_onehot", test_cross_entropy_onehot),
        ("softmax_temperature_t1", test_softmax_temperature_t1),
        ("softmax_temperature_high", test_softmax_temperature_high),
        ("softmax_temperature_gradient", test_softmax_temperature_gradient),
        ("against_pytorch", test_against_pytorch),
        ("ce_pytorch", test_ce_pytorch),
    ]
    
    print(f"\n{'='*50}\nDay 20: Softmax Implementation - Tests\n{'='*50}")
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
