"""Test Suite for Day 35: Complete Autodiff Library"""

import numpy as np
import pytest

try:
    from day35 import (
        Tensor,
        Parameter,
        Module,
        Linear,
        ReLU,
        Flatten,
        Dropout,
        Sequential,
        Conv2d,
        MaxPool2d,
        BatchNorm2d,
        CrossEntropyLoss,
        MSELoss,
        SGD,
        Adam,
        DataLoader,
        SimpleCNN,
        softmax,
        generate_synthetic_mnist,
        train_epoch,
        evaluate,
        im2col,
        col2im
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

# ============================================================================
# Tensor Tests
# ============================================================================

def test_tensor_creation():
    """Test tensor creation."""
    t = Tensor([1, 2, 3])
    assert not (t.shape != (3,)), f"shape {t.shape}"
        
    # Verify values are stored correctly
    assert np.allclose(t.data, [1, 2, 3]), f"values {t.data}"
        
def test_tensor_add():
    """Test tensor addition."""
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
        
    assert np.allclose(c.data, [5, 7, 9]), f"values {c.data}"
        
    c.sum().backward()
    assert np.allclose(a.grad, [1, 1, 1]), f"grad {a.grad}"
        
def test_tensor_mul():
    """Test tensor multiplication."""
    a = Tensor([2, 3, 4])
    b = Tensor([1, 2, 3])
    c = a * b
        
    assert np.allclose(c.data, [2, 6, 12]), f"values {c.data}"
        
    c.sum().backward()
    assert np.allclose(a.grad, [1, 2, 3]), f"grad {a.grad}"
        
def test_tensor_matmul():
    """Test matrix multiplication."""
    np.random.seed(42)
    a = Tensor(np.random.randn(3, 4))
    b = Tensor(np.random.randn(4, 5))
    c = a @ b
        
    assert not (c.shape != (3, 5)), f"shape {c.shape}"
        
    # Verify actual matmul values
    expected = a.data @ b.data
    assert np.allclose(c.data, expected), "matmul values incorrect"
        
    c.sum().backward()
    assert not (a.grad.shape != a.shape), "grad shape mismatch"
        
    # For sum loss, grad_a = ones @ b.T
    expected_grad_a = np.ones((3, 5)) @ b.data.T
    assert np.allclose(a.grad, expected_grad_a, rtol=1e-5), "grad values incorrect"
        
def test_tensor_sum():
    """Test tensor sum."""
    a = Tensor([[1, 2], [3, 4]])
    s = a.sum()
        
    assert not (s.data != 10), f"value {s.data}"
        
    s.backward()
    assert np.allclose(a.grad, [[1, 1], [1, 1]]), f"grad {a.grad}"
        
def test_tensor_mean():
    """Test tensor mean."""
    a = Tensor([[2, 4], [6, 8]])
    m = a.mean()
        
    assert not (m.data != 5), f"value {m.data}"
        
    m.backward()
    assert np.allclose(a.grad, [[0.25, 0.25], [0.25, 0.25]]), f"grad {a.grad}"
        
def test_tensor_relu():
    """Test ReLU activation."""
    a = Tensor([-2, -1, 0, 1, 2])
    r = a.relu()
        
    assert np.allclose(r.data, [0, 0, 0, 1, 2]), f"values {r.data}"
        
    r.sum().backward()
    assert np.allclose(a.grad, [0, 0, 0, 1, 1]), f"grad {a.grad}"
        
def test_tensor_reshape():
    """Test tensor reshape."""
    a = Tensor(np.arange(12))
    b = a.reshape(3, 4)
        
    assert not (b.shape != (3, 4)), f"shape {b.shape}"
        
    # Verify values are preserved
    assert np.allclose(b.data.flatten(), a.data), "values not preserved"
        
    b.sum().backward()
    assert not (a.grad.shape != (12,)), "grad shape mismatch"
        
    # For sum, gradient should be all ones
    assert np.allclose(a.grad, 1.0), f"grad {a.grad[0]}, expected 1.0"
        
# ============================================================================
# Module Tests
# ============================================================================

def test_parameter():
    """Test Parameter class."""
    np.random.seed(42)
    data = np.random.randn(3, 4)
    p = Parameter(data)
        
    assert isinstance(p, Tensor), "not a Tensor"
    assert not (p.shape != (3, 4)), f"shape {p.shape}"
        
    # Verify values are stored correctly
    assert np.allclose(p.data, data), "values not stored correctly"
        
def test_linear_layer():
    """Test Linear layer."""
    np.random.seed(42)
    layer = Linear(10, 5)
        
    x = Tensor(np.random.randn(4, 10))
    y = layer(x)
        
    assert not (y.shape != (4, 5)), f"shape {y.shape}"
        
    # Verify linear computation: y = x @ W^T + b
    expected = x.data @ layer.weight.data.T + layer.bias.data
    assert np.allclose(y.data, expected, rtol=1e-5), "output values incorrect"
        
    y.sum().backward()
    assert not (np.all(layer.weight.grad == 0)), "no gradient"
        
    # Bias gradient = batch_size = 4
    assert np.allclose(layer.bias.grad, 4.0), f"bias grad {layer.bias.grad[0]}, expected 4.0"
        
def test_sequential():
    """Test Sequential container."""
    np.random.seed(42)
    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    )
        
    x = Tensor(np.random.randn(4, 10))
    y = model(x)
        
    assert not (y.shape != (4, 5)), f"shape {y.shape}"
        
    params = list(model.parameters())
    assert not (len(params) != 4), f"param count {len(params)}"
        
    # Verify output is finite
    assert np.all(np.isfinite(y.data)), "output contains NaN or Inf"
        
    # Verify backward works
    y.sum().backward()
    for p in params:
        assert np.all(np.isfinite(p.grad)), "gradient contains NaN or Inf"
        
def test_conv2d():
    """Test Conv2d layer."""
    np.random.seed(42)
    conv = Conv2d(3, 16, kernel_size=3, padding=1)
        
    x = Tensor(np.random.randn(2, 3, 8, 8))
    y = conv(x)
        
    assert not (y.shape != (2, 16, 8, 8)), f"shape {y.shape}"
        
    # Verify output is finite and not all zeros
    assert np.all(np.isfinite(y.data)), "output contains NaN or Inf"
    assert not (np.all(y.data == 0)), "output is all zeros"
        
    y.sum().backward()
    assert not (np.all(conv.weight.grad == 0)), "no gradient"
        
    # Verify gradients are finite
    assert np.all(np.isfinite(conv.weight.grad)), "weight grad contains NaN or Inf"
        
def test_maxpool2d():
    """Test MaxPool2d layer."""
    np.random.seed(42)
    pool = MaxPool2d(kernel_size=2)
        
    x = Tensor(np.random.randn(2, 3, 8, 8))
    y = pool(x)
        
    assert not (y.shape != (2, 3, 4, 4)), f"shape {y.shape}"
        
    # Verify pooling is actually taking max
    expected_val = np.max(x.data[0, 0, 0:2, 0:2])
    assert np.isclose(y.data[0, 0, 0, 0], expected_val), f"pooled value {y.data[0,0,0,0]} vs expected {expected_val}"
        
    y.sum().backward()
    assert not (x.grad.shape != x.shape), "grad shape mismatch"
        
    # Gradient should be sparse (only max positions get gradient)
    non_zero_count = np.sum(x.grad != 0)
    assert not (non_zero_count != y.data.size), f"gradient sparsity: {non_zero_count} vs {y.data.size}"
        
def test_batchnorm2d():
    """Test BatchNorm2d layer."""
    np.random.seed(42)
    bn = BatchNorm2d(16)
    bn._training = True
        
    x = Tensor(np.random.randn(4, 16, 8, 8))
    y = bn(x)
        
    assert not (y.shape != (4, 16, 8, 8)), f"shape {y.shape}"
        
    mean = np.mean(y.data, axis=(0, 2, 3))
    assert np.allclose(mean, 0, atol=1e-5), "mean not normalized to 0"
        
    # Verify variance is approximately 1
    var = np.var(y.data, axis=(0, 2, 3))
    assert np.allclose(var, 1, atol=0.1), f"variance not normalized to 1: {var[0]}"
        
def test_dropout():
    """Test Dropout layer."""
    np.random.seed(42)
    dropout = Dropout(p=0.5)
    dropout._training = True
        
    x = Tensor(np.ones((100, 100)))
    y = dropout(x)
        
    zero_ratio = np.mean(y.data == 0)
    assert (0.3 < zero_ratio < 0.7), f"drop ratio {zero_ratio}"
        
    # Non-zero values should be scaled by 1/(1-p) = 2
    non_zero_vals = y.data[y.data != 0]
    assert np.allclose(non_zero_vals, 2.0, rtol=1e-5), f"scaling incorrect: {non_zero_vals[0]} vs 2.0"
        
    dropout._training = False
    y2 = dropout(x)
    assert np.allclose(y2.data, x.data), "eval not identity"
        
# ============================================================================
# Loss Function Tests
# ============================================================================

def test_cross_entropy():
    """Test CrossEntropyLoss."""
    loss_fn = CrossEntropyLoss()
        
    logits = Tensor(np.array([[2.0, 1.0, 0.1], [0.1, 2.0, 0.1]]))
    targets = np.array([0, 1])
        
    loss = loss_fn(logits, targets)
        
    assert not (loss.data.shape != ()), "not scalar"
    assert not (loss.data < 0), "negative loss"
        
    # Verify actual loss value
    probs = softmax(logits.data)
    expected_loss = -np.mean(np.log(probs[np.arange(2), targets]))
    assert np.isclose(loss.data, expected_loss, rtol=1e-5), f"loss {loss.data} vs expected {expected_loss}"
        
    loss.backward()
    assert not (np.all(logits.grad == 0)), "no gradient"
        
    # Gradient rows should sum to approximately 0
    assert np.allclose(logits.grad.sum(axis=1), 0, atol=1e-6), "grad rows should sum to ~0"
        
def test_mse_loss():
    """Test MSELoss."""
    loss_fn = MSELoss()
        
    pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
    target = Tensor([[1.5, 2.5], [3.5, 4.5]])
        
    loss = loss_fn(pred, target)
        
    assert not (loss.data.shape != ()), "not scalar"
        
    expected = 0.25
    assert np.allclose(loss.data, expected), f"value {loss.data}"
        
def test_softmax():
    """Test softmax function."""
    x = np.array([[1, 2, 3], [1, 2, 3]])
    probs = softmax(x)
        
    assert np.allclose(probs.sum(axis=1), 1), "doesn't sum to 1"
        
    # Verify actual softmax values
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    expected = exp_x / exp_x.sum(axis=1, keepdims=True)
    assert np.allclose(probs, expected), "values incorrect"
        
    # Verify ordering (larger inputs -> larger probs)
    assert (probs[0, 2] > probs[0, 1] > probs[0, 0]), "ordering incorrect"
        
    x_large = np.array([[1000, 1001, 1002]])
    probs_large = softmax(x_large)
    assert np.all(np.isfinite(probs_large)), "not numerically stable"
        
# ============================================================================
# Optimizer Tests
# ============================================================================

def test_sgd():
    """Test SGD optimizer."""
    w = Tensor(np.array([1.0, 2.0, 3.0]))
    optimizer = SGD([w], lr=0.1)
        
    w.grad = np.array([1.0, 1.0, 1.0])
    optimizer.step()
        
    assert np.allclose(w.data, [0.9, 1.9, 2.9]), f"values {w.data}"
        
    optimizer.zero_grad()
    assert np.allclose(w.grad, 0), "grad not zeroed"
        
def test_adam():
    """Test Adam optimizer."""
    w = Tensor(np.array([1.0, 2.0, 3.0]))
    original = w.data.copy()
    optimizer = Adam([w], lr=0.1)
        
    for _ in range(5):
        w.grad = np.array([1.0, 1.0, 1.0])
        optimizer.step()
        
    assert not (np.allclose(w.data, [1, 2, 3])), "no update"
        
    # Weights should have decreased (positive gradient = decrease)
    assert np.all(w.data < original), "weights should decrease with positive gradient"
        
    # Values should be finite
    assert np.all(np.isfinite(w.data)), "weights contain NaN or Inf"
        
# ============================================================================
# Data Loading Tests
# ============================================================================

def test_synthetic_data():
    """Test synthetic MNIST generation."""
    np.random.seed(42)
    data = generate_synthetic_mnist(n_samples=100, n_classes=10)
        
    assert not (data is None), "not implemented"
        
    X, y = data
        
    assert not (X.shape != (100, 1, 28, 28)), f"X shape {X.shape}"
    assert not (y.shape != (100,)), f"y shape {y.shape}"
        
    assert np.all((y >= 0) & (y < 10)), "invalid labels"
        
    # Verify data is finite
    assert np.all(np.isfinite(X)), "X contains NaN or Inf"
        
def test_data_loader():
    """Test DataLoader."""
    np.random.seed(42)
    X = np.arange(1000).reshape(100, 10).astype(float)
    y = np.arange(100)
        
    # Test without shuffle first
    loader_no_shuffle = DataLoader(X, y, batch_size=32, shuffle=False)
    batches_ns = list(loader_no_shuffle)
        
    # Verify first batch values
    assert np.allclose(batches_ns[0][0], X[:32]), "batch values incorrect"
    assert np.allclose(batches_ns[0][1], y[:32]), "batch labels incorrect"
        
    loader = DataLoader(X, y, batch_size=32, shuffle=True)
        
    batches = list(loader)
    assert not (len(batches) != 4), f"batch count {len(batches)}"
        
    total_samples = sum(len(batch[1]) for batch in batches)
    assert not (total_samples != 100), f"sample count {total_samples}"
        
# ============================================================================
# Complete Model Tests
# ============================================================================

def test_simple_cnn():
    """Test SimpleCNN model."""
    np.random.seed(42)
    model = SimpleCNN(in_channels=1, num_classes=10)
        
    assert not (model.conv1 is None), "not implemented"
        
    x = Tensor(np.random.randn(2, 1, 28, 28))
    y = model(x)
        
    assert not (y is None), "forward returned None"
    assert not (y.shape != (2, 10)), f"shape {y.shape}"
        
    # Verify output is finite
    assert np.all(np.isfinite(y.data)), "output contains NaN or Inf"
        
    # Different inputs should give different outputs
    x2 = Tensor(np.random.randn(2, 1, 28, 28))
    y2 = model(x2)
    assert not (np.allclose(y.data, y2.data)), "same output for different inputs"
        
def test_simple_cnn_backward():
    """Test SimpleCNN backward pass."""
    np.random.seed(42)
    model = SimpleCNN(in_channels=1, num_classes=10)
        
    assert not (model.conv1 is None), "not implemented"
        
    x = Tensor(np.random.randn(2, 1, 28, 28))
    y = model(x)
        
    assert not (y is None), "forward returned None"
        
    loss_fn = CrossEntropyLoss()
    targets = np.array([0, 1])
    loss = loss_fn(y, targets)
        
    assert not (loss is None), "loss is None"
        
    loss.backward()
        
    has_grad = any(np.any(p.grad != 0) for p in model.parameters())
    assert has_grad, "no gradients"
        
    # Verify all gradients are finite
    for p in model.parameters():
        assert np.all(np.isfinite(p.grad)), "gradient contains NaN or Inf"
        
def test_training_step():
    """Test a complete training step."""
    np.random.seed(42)
        
    model = Sequential(
        Linear(10, 32),
        ReLU(),
        Linear(32, 5)
    )
        
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1)
        
    x = Tensor(np.random.randn(8, 10))
    targets = np.random.randint(0, 5, 8)
        
    logits = model(x)
    initial_loss = loss_fn(logits, targets)
    initial_loss_value = initial_loss.data
        
    for _ in range(10):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        
    assert not (loss.data >= initial_loss_value), f"loss didn't decrease: {initial_loss_value:.4f} -> {loss.data:.4f}"
        
def test_full_training():
    """Test full training loop."""
    np.random.seed(42)
        
    data = generate_synthetic_mnist(n_samples=200, n_classes=10)
    assert not (data is None), "synthetic data not implemented"
        
    X, y = data
        
    train_loader = DataLoader(X[:160], y[:160], batch_size=32)
    val_loader = DataLoader(X[160:], y[160:], batch_size=32)
        
    model = SimpleCNN(in_channels=1, num_classes=10)
    assert not (model.conv1 is None), "SimpleCNN not implemented"
        
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
        
    initial_loss, _ = evaluate(model, val_loader, loss_fn)
        
    for _ in range(3):
        train_epoch(model, train_loader, loss_fn, optimizer)
        
    final_loss, accuracy = evaluate(model, val_loader, loss_fn)
        
    assert not (final_loss >= initial_loss), "loss didn't decrease"
        
def test_im2col_col2im():
    """Test im2col and col2im functions."""
    np.random.seed(42)
    x = np.random.randn(2, 3, 8, 8)
        
    col = im2col(x, 3, 3, stride=1, padding=1)
        
    assert not (col.shape[0] != 2 * 8 * 8), f"col shape {col.shape}"
        
    # Verify col contains finite values
    assert np.all(np.isfinite(col)), "im2col output contains NaN or Inf"
        
    x_reconstructed = col2im(col, x.shape, 3, 3, stride=1, padding=1)
        
    assert not (x_reconstructed.shape != x.shape), "reconstruction shape mismatch"
        
    # Verify reconstruction is finite
    assert np.all(np.isfinite(x_reconstructed)), "col2im output contains NaN or Inf"
        
def test_model_train_eval_mode():
    """Test train/eval mode switching."""
    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Dropout(0.5),
        Linear(20, 5)
    )
        
    model.train()
    for m in model._modules.values():
        assert not (hasattr(m, '_training') and not m._training), "not in train mode"
        
    model.eval()
    for m in model._modules.values():
        assert not (hasattr(m, '_training') and m._training), "not in eval mode"
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])