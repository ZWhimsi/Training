"""Test Suite for Day 21: Cross-Entropy Loss"""

import numpy as np
import sys
from typing import Tuple

try:
    from day21 import (
        Tensor,
        test_log_softmax,
        test_softmax,
        test_binary_cross_entropy,
        test_cross_entropy_loss,
        test_nll_loss,
        test_cross_entropy_gradient
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_log_softmax_forward() -> Tuple[bool, str]:
    """Test log-softmax forward pass."""
    try:
        x = Tensor([[1.0, 2.0, 3.0]])
        log_sm = x.log_softmax()
        
        if log_sm is None or log_sm.data is None:
            return False, "Returned None"
        
        probs = np.exp(log_sm.data)
        if not np.allclose(probs.sum(), 1.0):
            return False, f"exp(log_softmax) doesn't sum to 1: {probs.sum()}"
        
        return True, "log_softmax produces valid log-probabilities"
    except Exception as e:
        return False, str(e)


def test_log_softmax_backward() -> Tuple[bool, str]:
    """Test log-softmax backward pass."""
    try:
        x = Tensor([[1.0, 2.0, 3.0]])
        log_sm = x.log_softmax()
        log_sm.backward()
        
        if x.grad is None:
            return False, "grad is None"
        
        if np.all(x.grad == 0):
            return False, "grad is all zeros"
        
        return True, "log_softmax backward works"
    except Exception as e:
        return False, str(e)


def test_log_softmax_stability() -> Tuple[bool, str]:
    """Test log-softmax numerical stability."""
    try:
        x = Tensor([[1000.0, 1001.0, 1002.0]])
        log_sm = x.log_softmax()
        
        if log_sm is None or log_sm.data is None:
            return False, "Returned None"
        
        if np.any(np.isnan(log_sm.data)) or np.any(np.isinf(log_sm.data)):
            return False, "NaN or Inf in output"
        
        probs = np.exp(log_sm.data)
        if not np.allclose(probs.sum(), 1.0, rtol=1e-5):
            return False, f"Probabilities don't sum to 1: {probs.sum()}"
        
        return True, "Handles large values without overflow"
    except Exception as e:
        return False, str(e)


def test_softmax_forward() -> Tuple[bool, str]:
    """Test softmax forward pass."""
    try:
        x = Tensor([[1.0, 2.0, 3.0]])
        sm = x.softmax()
        
        if sm is None or sm.data is None:
            return False, "Returned None"
        
        if not np.allclose(sm.data.sum(), 1.0):
            return False, f"Doesn't sum to 1: {sm.data.sum()}"
        
        if not np.all(sm.data > 0):
            return False, "Contains non-positive values"
        
        return True, "softmax produces valid probabilities"
    except Exception as e:
        return False, str(e)


def test_softmax_backward() -> Tuple[bool, str]:
    """Test softmax backward pass."""
    try:
        x = Tensor([[1.0, 2.0, 3.0]])
        sm = x.softmax()
        sm.backward()
        
        if x.grad is None:
            return False, "grad is None"
        
        return True, "softmax backward works"
    except Exception as e:
        return False, str(e)


def test_bce_perfect_prediction() -> Tuple[bool, str]:
    """Test BCE with perfect predictions."""
    try:
        pred = Tensor([0.99, 0.01])
        target = Tensor([1.0, 0.0])
        loss = pred.binary_cross_entropy(target)
        
        if loss is None or loss.data is None:
            return False, "Returned None"
        
        if not loss.data < 0.1:
            return False, f"Loss too high for good prediction: {loss.data}"
        
        return True, "Low loss for correct predictions"
    except Exception as e:
        return False, str(e)


def test_bce_wrong_prediction() -> Tuple[bool, str]:
    """Test BCE with wrong predictions."""
    try:
        pred = Tensor([0.01, 0.99])
        target = Tensor([1.0, 0.0])
        loss = pred.binary_cross_entropy(target)
        
        if loss is None or loss.data is None:
            return False, "Returned None"
        
        if not loss.data > 2.0:
            return False, f"Loss too low for wrong prediction: {loss.data}"
        
        return True, "High loss for wrong predictions"
    except Exception as e:
        return False, str(e)


def test_bce_backward() -> Tuple[bool, str]:
    """Test BCE backward pass."""
    try:
        pred = Tensor([0.5, 0.5])
        target = Tensor([1.0, 0.0])
        loss = pred.binary_cross_entropy(target)
        loss.backward()
        
        if pred.grad is None:
            return False, "grad is None"
        
        if pred.grad[0] >= 0:
            return False, f"Gradient should be negative to increase pred[0]: {pred.grad[0]}"
        if pred.grad[1] <= 0:
            return False, f"Gradient should be positive to decrease pred[1]: {pred.grad[1]}"
        
        return True, "BCE gradient direction correct"
    except Exception as e:
        return False, str(e)


def test_ce_forward() -> Tuple[bool, str]:
    """Test cross-entropy forward pass."""
    try:
        logits = Tensor([[2.0, 1.0, 0.1]])
        target = np.array([0])
        loss = logits.cross_entropy_loss(target)
        
        if loss is None or loss.data is None:
            return False, "Returned None"
        
        if not loss.data > 0:
            return False, f"Loss should be positive: {loss.data}"
        
        return True, "CE loss is positive"
    except Exception as e:
        return False, str(e)


def test_ce_correct_class_low_loss() -> Tuple[bool, str]:
    """Test that correct confident predictions have low loss."""
    try:
        logits = Tensor([[10.0, 0.0, 0.0]])  # Very confident class 0
        target = np.array([0])
        loss = logits.cross_entropy_loss(target)
        
        if loss is None or loss.data is None:
            return False, "Returned None"
        
        if not loss.data < 0.01:
            return False, f"Loss should be near 0 for confident correct: {loss.data}"
        
        return True, "Low loss for confident correct prediction"
    except Exception as e:
        return False, str(e)


def test_ce_wrong_class_high_loss() -> Tuple[bool, str]:
    """Test that wrong confident predictions have high loss."""
    try:
        logits = Tensor([[0.0, 0.0, 10.0]])  # Very confident class 2
        target = np.array([0])  # But true class is 0
        loss = logits.cross_entropy_loss(target)
        
        if loss is None or loss.data is None:
            return False, "Returned None"
        
        if not loss.data > 5.0:
            return False, f"Loss should be high for confident wrong: {loss.data}"
        
        return True, "High loss for confident wrong prediction"
    except Exception as e:
        return False, str(e)


def test_ce_backward() -> Tuple[bool, str]:
    """Test cross-entropy backward pass."""
    try:
        logits = Tensor([[1.0, 2.0, 3.0]])
        target = np.array([1])
        loss = logits.cross_entropy_loss(target)
        loss.backward()
        
        if logits.grad is None:
            return False, "grad is None"
        
        if not np.allclose(logits.grad.sum(), 0, atol=1e-6):
            return False, f"Gradient should sum to 0: {logits.grad.sum()}"
        
        return True, "CE gradient sums to 0 (softmax - one_hot property)"
    except Exception as e:
        return False, str(e)


def test_ce_gradient_formula() -> Tuple[bool, str]:
    """Test the gradient = softmax - one_hot formula."""
    try:
        logits = Tensor([[1.0, 2.0, 3.0]])
        target = np.array([2])
        loss = logits.cross_entropy_loss(target)
        loss.backward()
        
        exp_x = np.exp(logits.data - logits.data.max())
        softmax = exp_x / exp_x.sum()
        one_hot = np.array([[0, 0, 1]])
        expected = softmax - one_hot
        
        if not np.allclose(logits.grad, expected, rtol=1e-5):
            return False, f"Gradient doesn't match softmax - one_hot"
        
        return True, "Gradient = softmax - one_hot verified"
    except Exception as e:
        return False, str(e)


def test_ce_batch() -> Tuple[bool, str]:
    """Test cross-entropy with batch."""
    try:
        logits = Tensor([[2.0, 1.0, 0.1],
                         [0.1, 0.2, 3.0],
                         [1.0, 2.0, 1.0]])
        target = np.array([0, 2, 1])
        loss = logits.cross_entropy_loss(target)
        
        if loss is None or loss.data is None:
            return False, "Returned None"
        
        loss.backward()
        if logits.grad is None:
            return False, "grad is None"
        
        return True, "Batch CE works"
    except Exception as e:
        return False, str(e)


def test_nll_forward() -> Tuple[bool, str]:
    """Test NLL loss forward pass."""
    try:
        log_probs = Tensor([[-0.5, -1.5, -2.5]])
        target = np.array([0])
        loss = log_probs.nll_loss(target)
        
        if loss is None or loss.data is None:
            return False, "Returned None"
        
        if not np.allclose(loss.data, 0.5):
            return False, f"Loss = {loss.data}, expected 0.5"
        
        return True, "NLL forward correct"
    except Exception as e:
        return False, str(e)


def test_nll_backward() -> Tuple[bool, str]:
    """Test NLL loss backward pass."""
    try:
        log_probs = Tensor([[-0.5, -1.5, -2.5],
                            [-1.0, -0.5, -2.0]])
        target = np.array([0, 1])
        loss = log_probs.nll_loss(target)
        loss.backward()
        
        if log_probs.grad is None:
            return False, "grad is None"
        
        expected = np.zeros_like(log_probs.data)
        expected[0, 0] = -0.5
        expected[1, 1] = -0.5
        
        if not np.allclose(log_probs.grad, expected):
            return False, f"Gradient mismatch"
        
        return True, "NLL backward correct"
    except Exception as e:
        return False, str(e)


def test_log_softmax_nll_equals_ce() -> Tuple[bool, str]:
    """Test that log_softmax + NLL equals cross-entropy."""
    try:
        np.random.seed(42)
        logits_data = np.random.randn(4, 5)
        target = np.array([0, 2, 4, 1])
        
        logits1 = Tensor(logits_data.copy())
        loss1 = logits1.cross_entropy_loss(target)
        
        if loss1 is None:
            return False, "CE returned None"
        
        loss1.backward()
        
        logits2 = Tensor(logits_data.copy())
        log_probs = logits2.log_softmax()
        if log_probs is None:
            return False, "log_softmax returned None"
        
        loss2 = log_probs.nll_loss(target)
        if loss2 is None:
            return False, "NLL returned None"
        
        if not np.allclose(loss1.data, loss2.data):
            return False, f"CE={loss1.data}, log_softmax+NLL={loss2.data}"
        
        loss2.backward()
        
        if not np.allclose(logits1.grad, logits2.grad, rtol=1e-5):
            return False, "Gradients don't match"
        
        return True, "CE = log_softmax + NLL"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test against PyTorch reference."""
    try:
        import torch
        import torch.nn.functional as F
        
        np.random.seed(42)
        logits_np = np.random.randn(4, 5)
        target_np = np.array([0, 2, 4, 1])
        
        ours = Tensor(logits_np.copy())
        our_loss = ours.cross_entropy_loss(target_np)
        
        if our_loss is None:
            return False, "CE returned None"
        
        our_loss.backward()
        
        torch_logits = torch.tensor(logits_np, requires_grad=True)
        torch_target = torch.tensor(target_np)
        torch_loss = F.cross_entropy(torch_logits, torch_target)
        torch_loss.backward()
        
        if not np.allclose(our_loss.data, torch_loss.item(), rtol=1e-5):
            return False, f"Loss mismatch: ours={our_loss.data}, torch={torch_loss.item()}"
        
        if not np.allclose(ours.grad, torch_logits.grad.numpy(), rtol=1e-5):
            return False, "Gradient mismatch"
        
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def test_log_softmax_axis() -> Tuple[bool, str]:
    """Test log-softmax along different axes."""
    try:
        x = Tensor(np.random.randn(3, 4, 5))
        
        log_sm0 = x.log_softmax(axis=0)
        if log_sm0 is None:
            return False, "axis=0 returned None"
        
        probs0 = np.exp(log_sm0.data)
        if not np.allclose(probs0.sum(axis=0), 1.0):
            return False, "axis=0 doesn't sum to 1"
        
        log_sm1 = Tensor(x.data.copy()).log_softmax(axis=1)
        if log_sm1 is None:
            return False, "axis=1 returned None"
        
        probs1 = np.exp(log_sm1.data)
        if not np.allclose(probs1.sum(axis=1), 1.0):
            return False, "axis=1 doesn't sum to 1"
        
        return True, "Multi-axis log-softmax works"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("log_softmax_forward", test_log_softmax_forward),
        ("log_softmax_backward", test_log_softmax_backward),
        ("log_softmax_stability", test_log_softmax_stability),
        ("softmax_forward", test_softmax_forward),
        ("softmax_backward", test_softmax_backward),
        ("bce_perfect_prediction", test_bce_perfect_prediction),
        ("bce_wrong_prediction", test_bce_wrong_prediction),
        ("bce_backward", test_bce_backward),
        ("ce_forward", test_ce_forward),
        ("ce_correct_class_low_loss", test_ce_correct_class_low_loss),
        ("ce_wrong_class_high_loss", test_ce_wrong_class_high_loss),
        ("ce_backward", test_ce_backward),
        ("ce_gradient_formula", test_ce_gradient_formula),
        ("ce_batch", test_ce_batch),
        ("nll_forward", test_nll_forward),
        ("nll_backward", test_nll_backward),
        ("log_softmax_nll_equals_ce", test_log_softmax_nll_equals_ce),
        ("against_pytorch", test_against_pytorch),
        ("log_softmax_axis", test_log_softmax_axis),
    ]
    
    print(f"\n{'='*60}")
    print("Day 21: Cross-Entropy Loss - Tests")
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
