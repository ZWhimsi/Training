"""Test Suite for Day 12: Dropout and Regularization"""

import torch
import torch.nn as nn
from typing import Tuple

try:
    from day12 import (manual_dropout, ManualDropout, dropout2d,
                       compute_l2_regularization, train_step_with_l2,
                       compute_l1_regularization, compute_elastic_net_penalty,
                       RegularizedMLP, compare_dropout_behavior)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_manual_dropout_training() -> Tuple[bool, str]:
    """Test manual dropout in training mode."""
    try:
        torch.manual_seed(42)
        x = torch.randn(100, 50)
        
        out = manual_dropout(x, p=0.5, training=True)
        
        if torch.allclose(out, x):
            return False, "Not implemented"
        
        # Check that approximately 50% are zero
        zero_frac = (out == 0).float().mean().item()
        if zero_frac < 0.3 or zero_frac > 0.7:
            return False, f"Expected ~50% zeros, got {zero_frac:.1%}"
        
        # Check scaling: non-zero elements should be scaled by 1/(1-p) = 2
        mask = out != 0
        if mask.sum() > 0:
            # The non-zero values should be approximately 2x the original
            ratio = out[mask] / x[mask]
            expected_ratio = 2.0
            if not torch.allclose(ratio, torch.full_like(ratio, expected_ratio), atol=0.01):
                return False, f"Scaling incorrect, expected {expected_ratio}x"
        
        return True, f"OK ({zero_frac:.1%} zeros)"
    except Exception as e:
        return False, str(e)


def test_manual_dropout_eval() -> Tuple[bool, str]:
    """Test manual dropout in eval mode."""
    try:
        torch.manual_seed(42)
        x = torch.randn(100, 50)
        
        out = manual_dropout(x, p=0.5, training=False)
        
        # In eval mode, should be identity
        if not torch.allclose(out, x):
            return False, "Eval mode should be identity"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_dropout_layer() -> Tuple[bool, str]:
    """Test ManualDropout layer."""
    try:
        torch.manual_seed(42)
        
        dropout = ManualDropout(p=0.3)
        x = torch.randn(100, 50)
        
        # Training mode
        dropout.train()
        out_train = dropout(x)
        
        if torch.allclose(out_train, x):
            return False, "Not implemented"
        
        # Eval mode
        dropout.eval()
        out_eval = dropout(x)
        
        if not torch.allclose(out_eval, x):
            return False, "Eval mode should be identity"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_dropout2d() -> Tuple[bool, str]:
    """Test 2D dropout (channel-wise)."""
    try:
        torch.manual_seed(42)
        x = torch.randn(8, 64, 14, 14)  # (N, C, H, W)
        
        out = dropout2d(x, p=0.5, training=True)
        
        if torch.allclose(out, x):
            return False, "Not implemented"
        
        # Check that entire channels are dropped (all spatial locations same)
        # For each channel, either all zeros or all non-zeros
        for n in range(out.shape[0]):
            for c in range(out.shape[1]):
                channel = out[n, c]
                is_zero = (channel == 0).all()
                is_nonzero = (channel != 0).all()
                if not (is_zero or is_nonzero):
                    return False, "Channels should be fully dropped or kept"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_l2_regularization() -> Tuple[bool, str]:
    """Test L2 regularization computation."""
    try:
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        weight_decay = 0.01
        l2 = compute_l2_regularization(model, weight_decay)
        
        if l2.item() == 0.0:
            return False, "Not implemented"
        
        # Compute expected
        expected = torch.tensor(0.0)
        for param in model.parameters():
            expected = expected + (param ** 2).sum()
        expected = (weight_decay / 2) * expected
        
        if not torch.allclose(l2, expected, rtol=1e-4):
            return False, f"Expected {expected.item():.6f}, got {l2.item():.6f}"
        
        return True, f"OK (penalty={l2.item():.6f})"
    except Exception as e:
        return False, str(e)


def test_l1_regularization() -> Tuple[bool, str]:
    """Test L1 regularization computation."""
    try:
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        l1_lambda = 0.01
        l1 = compute_l1_regularization(model, l1_lambda)
        
        if l1.item() == 0.0:
            return False, "Not implemented"
        
        # Compute expected
        expected = torch.tensor(0.0)
        for param in model.parameters():
            expected = expected + param.abs().sum()
        expected = l1_lambda * expected
        
        if not torch.allclose(l1, expected, rtol=1e-4):
            return False, f"Expected {expected.item():.6f}, got {l1.item():.6f}"
        
        return True, f"OK (penalty={l1.item():.6f})"
    except Exception as e:
        return False, str(e)


def test_elastic_net() -> Tuple[bool, str]:
    """Test Elastic Net (L1 + L2) regularization."""
    try:
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        l1_lambda = 0.01
        l2_lambda = 0.001
        
        elastic = compute_elastic_net_penalty(model, l1_lambda, l2_lambda)
        
        if elastic.item() == 0.0:
            return False, "Not implemented"
        
        # Should be sum of L1 and L2
        l1 = compute_l1_regularization(model, l1_lambda)
        l2 = compute_l2_regularization(model, l2_lambda)
        expected = l1 + l2
        
        if not torch.allclose(elastic, expected, rtol=1e-4):
            return False, f"Expected {expected.item():.6f}, got {elastic.item():.6f}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_train_step_with_l2() -> Tuple[bool, str]:
    """Test training step with L2 regularization."""
    try:
        torch.manual_seed(42)
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 3)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        x = torch.randn(16, 10)
        y = torch.randint(0, 3, (16,))
        weight_decay = 0.1
        
        # Store initial weights
        initial_weight = model[0].weight.clone()
        
        # Compute expected loss manually
        with torch.no_grad():
            pred = model(x)
            base_loss = loss_fn(pred, y)
            l2_penalty = torch.tensor(0.0)
            for param in model.parameters():
                l2_penalty = l2_penalty + (param ** 2).sum()
            expected_loss = base_loss + (weight_decay / 2) * l2_penalty
        
        loss = train_step_with_l2(model, x, y, loss_fn, optimizer, weight_decay=weight_decay)
        
        if loss == 0.0:
            return False, "Not implemented"
        
        # Weights should have changed
        if torch.allclose(model[0].weight, initial_weight):
            return False, "Weights not updated"
        
        # Validate loss value includes L2 penalty
        if not abs(loss - expected_loss.item()) < 0.1:
            return False, f"Loss {loss:.4f} doesn't match expected {expected_loss.item():.4f} (including L2 penalty)"
        
        return True, f"OK (loss={loss:.4f})"
    except Exception as e:
        return False, str(e)


def test_regularized_mlp() -> Tuple[bool, str]:
    """Test RegularizedMLP with dropout."""
    try:
        model = RegularizedMLP(input_dim=64, hidden_dim=128, output_dim=10, dropout_p=0.5)
        
        if model.fc1 is None:
            return False, "Not implemented"
        
        x = torch.randn(32, 64)
        
        # Train mode - should have variation
        model.train()
        outputs = [model(x) for _ in range(5)]
        outputs = torch.stack(outputs)
        
        # Variance should be non-zero in train mode due to dropout
        variance = outputs.var(dim=0).mean()
        if variance < 1e-6:
            return False, "No variation in train mode (dropout not working)"
        
        # Eval mode - should be deterministic
        model.eval()
        out1 = model(x)
        out2 = model(x)
        
        if not torch.allclose(out1, out2):
            return False, "Eval mode should be deterministic"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_dropout_variance() -> Tuple[bool, str]:
    """Test comparing train vs eval behavior with dropout."""
    try:
        torch.manual_seed(42)
        
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )
        
        x = torch.randn(8, 32)
        
        train_var, eval_output = compare_dropout_behavior(model, x, num_runs=10)
        
        # Train variance should be non-zero
        if train_var.mean() < 1e-6:
            return False, "Train variance is zero (not implemented)"
        
        # Verify model is in eval mode after function
        if model.training:
            return False, "Model should be in eval mode after function"
        
        return True, f"OK (mean variance={train_var.mean():.4f})"
    except Exception as e:
        return False, str(e)


def test_dropout_preserves_expected_value() -> Tuple[bool, str]:
    """Test that dropout preserves expected value (when averaged)."""
    try:
        torch.manual_seed(42)
        x = torch.ones(1000, 500)  # Use ones for easy verification
        p = 0.5
        
        # Average over many runs
        outputs = []
        for _ in range(100):
            out = manual_dropout(x, p=p, training=True)
            outputs.append(out)
        
        avg_output = torch.stack(outputs).mean(dim=0)
        
        # Average should be close to original (expected value preserved)
        if not torch.allclose(avg_output, x, atol=0.1):
            return False, f"Expected value not preserved: mean={avg_output.mean():.3f}"
        
        return True, f"OK (mean={avg_output.mean():.3f})"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("manual_dropout_training", test_manual_dropout_training),
        ("manual_dropout_eval", test_manual_dropout_eval),
        ("dropout_layer", test_dropout_layer),
        ("dropout2d", test_dropout2d),
        ("l2_regularization", test_l2_regularization),
        ("l1_regularization", test_l1_regularization),
        ("elastic_net", test_elastic_net),
        ("train_step_with_l2", test_train_step_with_l2),
        ("regularized_mlp", test_regularized_mlp),
        ("dropout_variance", test_dropout_variance),
        ("dropout_expected_value", test_dropout_preserves_expected_value),
    ]
    
    print(f"\n{'='*50}\nDay 12: Dropout and Regularization - Tests\n{'='*50}")
    
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
