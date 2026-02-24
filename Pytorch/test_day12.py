"""Test Suite for Day 12: Dropout and Regularization"""

import torch
import pytest
import torch.nn as nn
try:
    from day12 import (manual_dropout, ManualDropout, dropout2d,
                       compute_l2_regularization, train_step_with_l2,
                       compute_l1_regularization, compute_elastic_net_penalty,
                       RegularizedMLP, compare_dropout_behavior)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_manual_dropout_training():
    """Test manual dropout in training mode."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    x = torch.randn(100, 50)
    
    out = manual_dropout(x, p=0.5, training=True)
    
    assert not torch.allclose(out, x), "Not implemented"
    
    zero_frac = (out == 0).float().mean().item()
    assert 0.3 <= zero_frac <= 0.7, f"Expected ~50% zeros, got {zero_frac:.1%}"
    
    mask = out != 0
    if mask.sum() > 0:
        ratio = out[mask] / x[mask]
        expected_ratio = 2.0
        assert torch.allclose(ratio, torch.full_like(ratio, expected_ratio), atol=0.01), f"Scaling incorrect, expected {expected_ratio}x"

def test_manual_dropout_eval():
    """Test manual dropout in eval mode."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    x = torch.randn(100, 50)
    
    out = manual_dropout(x, p=0.5, training=False)
    
    assert torch.allclose(out, x), "Eval mode should be identity"

def test_dropout_layer():
    """Test ManualDropout layer."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    dropout = ManualDropout(p=0.3)
    x = torch.randn(100, 50)
    
    dropout.train()
    out_train = dropout(x)
    
    assert not torch.allclose(out_train, x), "Not implemented"
    
    dropout.eval()
    out_eval = dropout(x)
    
    assert torch.allclose(out_eval, x), "Eval mode should be identity"

def test_dropout2d():
    """Test 2D dropout (channel-wise)."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    x = torch.randn(8, 64, 14, 14)
    
    out = dropout2d(x, p=0.5, training=True)
    
    assert not torch.allclose(out, x), "Not implemented"
    
    for n in range(out.shape[0]):
        for c in range(out.shape[1]):
            channel = out[n, c]
            is_zero = (channel == 0).all()
            is_nonzero = (channel != 0).all()
            assert is_zero or is_nonzero, "Channels should be fully dropped or kept"

def test_l2_regularization():
    """Test L2 regularization computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    weight_decay = 0.01
    l2 = compute_l2_regularization(model, weight_decay)
    
    assert l2.item() != 0.0, "Not implemented"
    
    expected = torch.tensor(0.0)
    for param in model.parameters():
        expected = expected + (param ** 2).sum()
    expected = (weight_decay / 2) * expected
    
    assert torch.allclose(l2, expected, rtol=1e-4), f"Expected {expected.item():.6f}, got {l2.item():.6f}"

def test_l1_regularization():
    """Test L1 regularization computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    l1_lambda = 0.01
    l1 = compute_l1_regularization(model, l1_lambda)
    
    assert l1.item() != 0.0, "Not implemented"
    
    expected = torch.tensor(0.0)
    for param in model.parameters():
        expected = expected + param.abs().sum()
    expected = l1_lambda * expected
    
    assert torch.allclose(l1, expected, rtol=1e-4), f"Expected {expected.item():.6f}, got {l1.item():.6f}"

def test_elastic_net():
    """Test Elastic Net (L1 + L2) regularization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    l1_lambda = 0.01
    l2_lambda = 0.001
    
    elastic = compute_elastic_net_penalty(model, l1_lambda, l2_lambda)
    
    assert elastic.item() != 0.0, "Not implemented"
    
    l1 = compute_l1_regularization(model, l1_lambda)
    l2 = compute_l2_regularization(model, l2_lambda)
    expected = l1 + l2
    
    assert torch.allclose(elastic, expected, rtol=1e-4), f"Expected {expected.item():.6f}, got {elastic.item():.6f}"

def test_train_step_with_l2():
    """Test training step with L2 regularization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
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
    
    initial_weight = model[0].weight.clone()
    
    with torch.no_grad():
        pred = model(x)
        base_loss = loss_fn(pred, y)
        l2_penalty = torch.tensor(0.0)
        for param in model.parameters():
            l2_penalty = l2_penalty + (param ** 2).sum()
        expected_loss = base_loss + (weight_decay / 2) * l2_penalty
    
    loss = train_step_with_l2(model, x, y, loss_fn, optimizer, weight_decay=weight_decay)
    
    assert loss != 0.0, "Not implemented"
    assert not torch.allclose(model[0].weight, initial_weight), "Weights not updated"
    assert abs(loss - expected_loss.item()) < 0.1, f"Loss {loss:.4f} doesn't match expected {expected_loss.item():.4f} (including L2 penalty)"

def test_regularized_mlp():
    """Test RegularizedMLP with dropout."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = RegularizedMLP(input_dim=64, hidden_dim=128, output_dim=10, dropout_p=0.5)
    
    assert model.fc1 is not None, "Not implemented"
    
    x = torch.randn(32, 64)
    
    model.train()
    outputs = [model(x) for _ in range(5)]
    outputs = torch.stack(outputs)
    
    variance = outputs.var(dim=0).mean()
    assert variance >= 1e-6, "No variation in train mode (dropout not working)"
    
    model.eval()
    out1 = model(x)
    out2 = model(x)
    
    assert torch.allclose(out1, out2), "Eval mode should be deterministic"

def test_dropout_variance():
    """Test comparing train vs eval behavior with dropout."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 10)
    )
    
    x = torch.randn(8, 32)
    
    train_var, eval_output = compare_dropout_behavior(model, x, num_runs=10)
    
    assert train_var.mean() >= 1e-6, "Train variance is zero (not implemented)"
    assert not model.training, "Model should be in eval mode after function"

def test_dropout_preserves_expected_value():
    """Test that dropout preserves expected value (when averaged)."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    x = torch.ones(1000, 500)
    p = 0.5
    
    outputs = []
    for _ in range(100):
        out = manual_dropout(x, p=p, training=True)
        outputs.append(out)
    
    avg_output = torch.stack(outputs).mean(dim=0)
    
    assert torch.allclose(avg_output, x, atol=0.1), f"Expected value not preserved: mean={avg_output.mean():.3f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
