"""Test Suite for Day 29: Gradient Clipping and Regularization"""

import numpy as np
import sys
from typing import Tuple

try:
    from day29 import (
        Tensor,
        Parameter,
        Module,
        Linear,
        ReLU,
        Sequential,
        Optimizer,
        SGD,
        Dropout,
        clip_grad_value_,
        clip_grad_norm_,
        l1_regularization,
        l2_regularization,
        apply_l1_gradient_,
        apply_l2_gradient_,
        mse_loss,
        train_with_regularization,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_clip_grad_value_clips_large() -> Tuple[bool, str]:
    """Test that large gradients are clipped."""
    try:
        p = Parameter(np.array([1.0]))
        p.grad = np.array([100.0])
        
        clip_grad_value_([p], clip_value=5.0)
        
        if p.grad[0] != 5.0:
            return False, f"Expected 5.0, got {p.grad[0]}"
        
        return True, "Large positive gradient clipped"
    except Exception as e:
        return False, str(e)


def test_clip_grad_value_clips_negative() -> Tuple[bool, str]:
    """Test that large negative gradients are clipped."""
    try:
        p = Parameter(np.array([1.0]))
        p.grad = np.array([-100.0])
        
        clip_grad_value_([p], clip_value=5.0)
        
        if p.grad[0] != -5.0:
            return False, f"Expected -5.0, got {p.grad[0]}"
        
        return True, "Large negative gradient clipped"
    except Exception as e:
        return False, str(e)


def test_clip_grad_value_preserves_small() -> Tuple[bool, str]:
    """Test that small gradients are preserved."""
    try:
        p = Parameter(np.array([1.0]))
        p.grad = np.array([2.0])
        
        clip_grad_value_([p], clip_value=5.0)
        
        if not np.isclose(p.grad[0], 2.0):
            return False, f"Expected 2.0, got {p.grad[0]}"
        
        return True, "Small gradient preserved"
    except Exception as e:
        return False, str(e)


def test_clip_grad_norm_basic() -> Tuple[bool, str]:
    """Test basic gradient norm clipping."""
    try:
        p = Parameter(np.array([3.0, 4.0]))  # norm = 5
        p.grad = np.array([3.0, 4.0])
        
        original_norm = clip_grad_norm_([p], max_norm=2.5)
        
        if not np.isclose(original_norm, 5.0, rtol=1e-4):
            return False, f"Original norm should be 5.0, got {original_norm}"
        
        new_norm = np.sqrt(np.sum(p.grad ** 2))
        if not np.isclose(new_norm, 2.5, rtol=1e-4):
            return False, f"Clipped norm should be 2.5, got {new_norm}"
        
        return True, "Norm clipped correctly"
    except Exception as e:
        return False, str(e)


def test_clip_grad_norm_direction() -> Tuple[bool, str]:
    """Test that clipping preserves gradient direction."""
    try:
        p = Parameter(np.array([3.0, 4.0]))
        original_direction = p.data / np.linalg.norm(p.data)
        p.grad = np.array([3.0, 4.0])
        
        clip_grad_norm_([p], max_norm=1.0)
        
        clipped_direction = p.grad / np.linalg.norm(p.grad)
        
        if not np.allclose(original_direction, clipped_direction, rtol=1e-4):
            return False, "Direction changed"
        
        return True, "Direction preserved"
    except Exception as e:
        return False, str(e)


def test_clip_grad_norm_no_clip_needed() -> Tuple[bool, str]:
    """Test that small gradients aren't modified."""
    try:
        p = Parameter(np.array([0.3, 0.4]))  # norm = 0.5
        p.grad = np.array([0.3, 0.4])
        original = p.grad.copy()
        
        clip_grad_norm_([p], max_norm=10.0)
        
        if not np.allclose(p.grad, original):
            return False, "Small gradient was modified"
        
        return True, "Small gradient preserved"
    except Exception as e:
        return False, str(e)


def test_clip_grad_norm_multiple_params() -> Tuple[bool, str]:
    """Test norm clipping with multiple parameters."""
    try:
        p1 = Parameter(np.array([3.0, 0.0]))
        p2 = Parameter(np.array([0.0, 4.0]))
        p1.grad = np.array([3.0, 0.0])
        p2.grad = np.array([0.0, 4.0])
        
        total_norm = clip_grad_norm_([p1, p2], max_norm=2.5)
        
        if not np.isclose(total_norm, 5.0, rtol=1e-4):
            return False, f"Total norm should be 5.0, got {total_norm}"
        
        new_norm = np.sqrt(np.sum(p1.grad ** 2) + np.sum(p2.grad ** 2))
        if not np.isclose(new_norm, 2.5, rtol=1e-4):
            return False, f"New norm should be 2.5, got {new_norm}"
        
        return True, "Multiple params clipped correctly"
    except Exception as e:
        return False, str(e)


def test_l1_regularization_basic() -> Tuple[bool, str]:
    """Test L1 regularization computation."""
    try:
        p = Parameter(np.array([1.0, -2.0, 3.0]))
        
        reg = l1_regularization([p], lambda_l1=0.1)
        expected = 0.1 * 6.0  # |1| + |-2| + |3| = 6
        
        if not np.isclose(reg.data, expected):
            return False, f"Expected {expected}, got {reg.data}"
        
        return True, "L1 regularization correct"
    except Exception as e:
        return False, str(e)


def test_l1_gradient() -> Tuple[bool, str]:
    """Test L1 gradient application."""
    try:
        p = Parameter(np.array([1.0, -2.0, 0.0]))
        p.grad = np.array([0.0, 0.0, 0.0])
        
        apply_l1_gradient_([p], lambda_l1=0.1)
        
        expected = np.array([0.1, -0.1, 0.0])
        if not np.allclose(p.grad, expected, atol=1e-5):
            return False, f"Expected {expected}, got {p.grad}"
        
        return True, "L1 gradient applied"
    except Exception as e:
        return False, str(e)


def test_l2_regularization_basic() -> Tuple[bool, str]:
    """Test L2 regularization computation."""
    try:
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        
        reg = l2_regularization([p], lambda_l2=0.1)
        expected = 0.05 * 14.0  # (1² + 2² + 3²) = 14, * λ/2 = 0.05
        
        if not np.isclose(reg.data, expected):
            return False, f"Expected {expected}, got {reg.data}"
        
        return True, "L2 regularization correct"
    except Exception as e:
        return False, str(e)


def test_l2_gradient() -> Tuple[bool, str]:
    """Test L2 gradient application."""
    try:
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        p.grad = np.array([0.0, 0.0, 0.0])
        
        apply_l2_gradient_([p], lambda_l2=0.1)
        
        expected = np.array([0.1, 0.2, 0.3])
        if not np.allclose(p.grad, expected):
            return False, f"Expected {expected}, got {p.grad}"
        
        return True, "L2 gradient applied"
    except Exception as e:
        return False, str(e)


def test_sgd_with_weight_decay() -> Tuple[bool, str]:
    """Test SGD with weight decay."""
    try:
        p = Parameter(np.array([10.0, 10.0]))
        opt = SGD([p], lr=0.1, weight_decay=0.1)
        
        initial = p.data.copy()
        
        for _ in range(20):
            p.grad = np.zeros_like(p.data)
            opt.step()
        
        if not np.all(np.abs(p.data) < np.abs(initial)):
            return False, "Weight decay not applied"
        
        return True, f"Decayed from {initial[0]:.2f} to {p.data[0]:.4f}"
    except Exception as e:
        return False, str(e)


def test_sgd_with_momentum_and_wd() -> Tuple[bool, str]:
    """Test SGD with both momentum and weight decay."""
    try:
        np.random.seed(42)
        p = Parameter(np.array([5.0, 5.0]))
        opt = SGD([p], lr=0.1, momentum=0.9, weight_decay=0.01)
        
        for _ in range(50):
            p.grad = np.random.randn(*p.shape) * 0.1
            opt.step()
        
        return True, "SGD with momentum and weight decay works"
    except Exception as e:
        return False, str(e)


def test_dropout_creation() -> Tuple[bool, str]:
    """Test Dropout layer creation."""
    try:
        dropout = Dropout(p=0.5)
        
        if dropout.p != 0.5:
            return False, f"Wrong p: {dropout.p}"
        
        return True, "Dropout created"
    except Exception as e:
        return False, str(e)


def test_dropout_train_mode() -> Tuple[bool, str]:
    """Test Dropout in training mode."""
    try:
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        dropout.train()
        
        x = Tensor(np.ones((100, 100)))
        out = dropout(x)
        
        num_zeros = np.sum(out.data == 0)
        total = out.data.size
        
        zero_ratio = num_zeros / total
        if not (0.4 < zero_ratio < 0.6):
            return False, f"Expected ~50% zeros, got {zero_ratio*100:.1f}%"
        
        return True, f"{zero_ratio*100:.1f}% dropped"
    except Exception as e:
        return False, str(e)


def test_dropout_scaling() -> Tuple[bool, str]:
    """Test Dropout scaling during training."""
    try:
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        dropout.train()
        
        x = Tensor(np.ones((1000, 1000)))
        out = dropout(x)
        
        nonzero_values = out.data[out.data != 0]
        
        if not np.allclose(nonzero_values, 2.0, rtol=0.01):
            return False, f"Non-zero values should be 2.0, got mean={np.mean(nonzero_values):.3f}"
        
        return True, "Scaling by 1/(1-p) correct"
    except Exception as e:
        return False, str(e)


def test_dropout_eval_mode() -> Tuple[bool, str]:
    """Test Dropout in evaluation mode."""
    try:
        dropout = Dropout(p=0.5)
        dropout.eval()
        
        x = Tensor(np.ones((10, 10)))
        out = dropout(x)
        
        if not np.allclose(out.data, x.data):
            return False, "Eval mode should pass through unchanged"
        
        return True, "Eval mode passes through"
    except Exception as e:
        return False, str(e)


def test_dropout_backward() -> Tuple[bool, str]:
    """Test Dropout backward pass."""
    try:
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        dropout.train()
        
        x = Tensor(np.ones((10, 10)))
        out = dropout(x)
        out.sum().backward()
        
        dropped_mask = (out.data == 0)
        if not np.all(x.grad[dropped_mask] == 0):
            return False, "Gradients should be zero where dropped"
        
        return True, "Backward pass correct"
    except Exception as e:
        return False, str(e)


def test_training_with_reg_runs() -> Tuple[bool, str]:
    """Test training with regularization runs."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        opt = SGD(list(model.parameters()), lr=0.01)
        
        X = np.random.randn(20, 4)
        Y = np.random.randn(20, 2)
        
        history = train_with_regularization(
            model, opt, X, Y,
            epochs=20,
            lambda_l2=0.01
        )
        
        if 'losses' not in history or len(history['losses']) == 0:
            return False, "No losses in history"
        
        return True, "Training with regularization runs"
    except Exception as e:
        return False, str(e)


def test_training_with_clipping() -> Tuple[bool, str]:
    """Test training with gradient clipping."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        opt = SGD(list(model.parameters()), lr=0.01)
        
        X = np.random.randn(20, 4)
        Y = np.random.randn(20, 2)
        
        history = train_with_regularization(
            model, opt, X, Y,
            epochs=20,
            clip_norm=1.0
        )
        
        if 'grad_norms' in history:
            clipped_norms = [n for n in history['grad_norms'] if n > 0]
            if clipped_norms and max(clipped_norms) > 1.1:
                return False, f"Gradient norm exceeded: {max(clipped_norms)}"
        
        return True, "Gradient clipping works"
    except Exception as e:
        return False, str(e)


def test_against_pytorch_clip_norm() -> Tuple[bool, str]:
    """Test gradient clipping against PyTorch."""
    try:
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        our_p = Parameter(np.array([3.0, 4.0]))
        our_p.grad = np.array([3.0, 4.0])
        
        torch_p = torch.tensor([3.0, 4.0], requires_grad=True, dtype=torch.float64)
        torch_p.grad = torch.tensor([3.0, 4.0], dtype=torch.float64)
        
        our_norm = clip_grad_norm_([our_p], max_norm=2.5)
        torch_norm = nn.utils.clip_grad_norm_([torch_p], max_norm=2.5).item()
        
        if not np.isclose(our_norm, torch_norm, rtol=1e-4):
            return False, f"Norm mismatch: {our_norm} vs {torch_norm}"
        
        if not np.allclose(our_p.grad, torch_p.grad.numpy(), rtol=1e-4):
            return False, f"Gradient mismatch"
        
        return True, "Matches PyTorch clip_grad_norm_"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("clip_grad_value_clips_large", test_clip_grad_value_clips_large),
        ("clip_grad_value_clips_negative", test_clip_grad_value_clips_negative),
        ("clip_grad_value_preserves_small", test_clip_grad_value_preserves_small),
        ("clip_grad_norm_basic", test_clip_grad_norm_basic),
        ("clip_grad_norm_direction", test_clip_grad_norm_direction),
        ("clip_grad_norm_no_clip_needed", test_clip_grad_norm_no_clip_needed),
        ("clip_grad_norm_multiple_params", test_clip_grad_norm_multiple_params),
        ("l1_regularization_basic", test_l1_regularization_basic),
        ("l1_gradient", test_l1_gradient),
        ("l2_regularization_basic", test_l2_regularization_basic),
        ("l2_gradient", test_l2_gradient),
        ("sgd_with_weight_decay", test_sgd_with_weight_decay),
        ("sgd_with_momentum_and_wd", test_sgd_with_momentum_and_wd),
        ("dropout_creation", test_dropout_creation),
        ("dropout_train_mode", test_dropout_train_mode),
        ("dropout_scaling", test_dropout_scaling),
        ("dropout_eval_mode", test_dropout_eval_mode),
        ("dropout_backward", test_dropout_backward),
        ("training_with_reg_runs", test_training_with_reg_runs),
        ("training_with_clipping", test_training_with_clipping),
        ("against_pytorch_clip_norm", test_against_pytorch_clip_norm),
    ]
    
    print(f"\n{'='*60}")
    print("Day 29: Gradient Clipping and Regularization - Tests")
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
