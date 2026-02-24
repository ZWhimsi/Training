"""Test Suite for Day 34: Complete CNN Module"""

import numpy as np
import pytest

try:
    from day34 import (
        Tensor,
        Module,
        Conv2d,
        MaxPool2d,
        BatchNorm2d,
        ReLU,
        Flatten,
        Linear,
        Dropout,
        Sequential,
        ConvBlock,
        ResidualBlock,
        LeNet,
        SimpleCNN,
        GlobalAvgPool,
        CrossEntropyLoss,
        SGD,
        softmax
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_relu_forward():
    """Test ReLU forward pass."""
    relu = ReLU()
    x = Tensor(np.array([-2, -1, 0, 1, 2]))
    y = relu(x)
        
    expected = np.array([0, 0, 0, 1, 2])
    assert np.allclose(y.data, expected), f"values {y.data}"
        
def test_relu_backward():
    """Test ReLU backward pass."""
    relu = ReLU()
    x = Tensor(np.array([-2.0, -1.0, 0.5, 1.0, 2.0]))
    y = relu(x)
    y.sum().backward()
        
    expected_grad = np.array([0, 0, 1, 1, 1])
    assert np.allclose(x.grad, expected_grad), f"grad {x.grad}"
        
def test_flatten():
    """Test Flatten layer."""
    np.random.seed(42)
    flatten = Flatten()
    x = Tensor(np.random.randn(2, 3, 4, 4))
    y = flatten(x)
        
    assert not (y.shape != (2, 48)), f"shape {y.shape}"
        
    # Verify values are preserved (just reshaped)
    expected = x.data.reshape(2, -1)
    assert np.allclose(y.data, expected), "values not preserved during flatten"
        
    y.sum().backward()
    assert not (x.grad.shape != x.shape), "grad shape mismatch"
        
    # For sum loss, gradient should be all ones
    assert np.allclose(x.grad, 1.0), f"gradient {x.grad[0,0,0,0]}, expected 1.0"
        
def test_linear_forward():
    """Test Linear layer forward."""
    np.random.seed(42)
    linear = Linear(10, 5)
        
    x = Tensor(np.random.randn(4, 10))
    y = linear(x)
        
    assert not (y.shape != (4, 5)), f"shape {y.shape}"
        
    # Verify linear computation: y = x @ W^T + b
    expected = x.data @ linear.weight.data.T + linear.bias.data
    assert np.allclose(y.data, expected, rtol=1e-5), f"output mismatch, max diff: {np.max(np.abs(y.data - expected))}"
        
def test_linear_backward():
    """Test Linear layer backward."""
    np.random.seed(42)
    linear = Linear(10, 5)
        
    x = Tensor(np.random.randn(4, 10))
    y = linear(x)
    y.sum().backward()
        
    assert not (np.all(linear.weight.grad == 0)), "weight grad is zero"
    assert not (np.all(linear.bias.grad == 0)), "bias grad is zero"
        
    # For sum loss, bias gradient = batch_size = 4
    expected_bias_grad = 4.0
    assert np.allclose(linear.bias.grad, expected_bias_grad), f"bias grad {linear.bias.grad[0]}, expected {expected_bias_grad}"
        
    # Weight gradient = sum of outer products = sum(x) for each output
    expected_weight_grad = np.ones((4, 5)).T @ x.data  # (5, 10)
    assert np.allclose(linear.weight.grad, expected_weight_grad, rtol=1e-5), "weight grad mismatch"
        
def test_dropout_training():
    """Test Dropout in training mode."""
    np.random.seed(42)
    dropout = Dropout(p=0.5)
    dropout._training = True
        
    x = Tensor(np.ones((100, 100)))
    y = dropout(x)
        
    zero_ratio = np.mean(y.data == 0)
    assert (0.3 < zero_ratio < 0.7), f"drop ratio {zero_ratio}"
        
    # Non-zero values should be scaled by 1/(1-p) = 2
    non_zero_vals = y.data[y.data != 0]
    assert np.allclose(non_zero_vals, 2.0, rtol=1e-5), f"non-zero values should be 2.0, got {non_zero_vals[0]}"
        
def test_dropout_eval():
    """Test Dropout in eval mode."""
    dropout = Dropout(p=0.5)
    dropout._training = False
        
    x = Tensor(np.ones((10, 10)) * 3.0)
    y = dropout(x)
        
    assert np.allclose(y.data, x.data), "should be identity in eval"
        
    # Explicitly verify value preservation
    assert np.allclose(y.data, 3.0), f"values not preserved: {y.data[0,0]} vs 3.0"
        
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
        
    params = model.parameters()
    assert not (len(params) != 4), f"params count {len(params)}"
        
    # Verify output is finite
    assert np.all(np.isfinite(y.data)), "output contains NaN or Inf"
        
    # Verify backward works
    y.sum().backward()
    for p in params:
        assert np.all(np.isfinite(p.grad)), "gradient contains NaN or Inf"
        
def test_conv_block_forward():
    """Test ConvBlock forward pass."""
    np.random.seed(42)
    block = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        
    assert not (block.conv is None), "conv is None (not implemented)"
        
    x = Tensor(np.random.randn(2, 3, 8, 8))
    y = block(x)
        
    assert not (y is None), "forward returned None"
        
    assert not (y.shape != (2, 32, 8, 8)), f"shape {y.shape}"
        
    # Verify ReLU is applied (no negative values)
    assert not (np.any(y.data < 0)), "ReLU not applied (found negative values)"
        
    # Verify output is finite
    assert np.all(np.isfinite(y.data)), "output contains NaN or Inf"
        
def test_conv_block_backward():
    """Test ConvBlock backward pass."""
    np.random.seed(42)
    block = ConvBlock(3, 16, kernel_size=3, stride=1, padding=1)
        
    assert not (block.conv is None), "conv is None"
        
    x = Tensor(np.random.randn(2, 3, 8, 8))
    y = block(x)
        
    assert not (y is None), "forward returned None"
        
    y.sum().backward()
        
    params = block.parameters()
    assert params, "no parameters"
        
    has_grad = any(np.any(p.grad != 0) for p in params)
    assert has_grad, "no gradients"
        
    # Verify gradients are finite
    for p in params:
        assert np.all(np.isfinite(p.grad)), "parameter gradient contains NaN or Inf"
    assert np.all(np.isfinite(x.grad)), "input gradient contains NaN or Inf"
        
def test_residual_block_same_dim():
    """Test ResidualBlock with same dimensions."""
    np.random.seed(42)
    block = ResidualBlock(32, 32, stride=1)
        
    assert not (block.conv1 is None), "conv1 is None"
        
    x = Tensor(np.random.randn(2, 32, 8, 8))
    y = block(x)
        
    assert not (y is None), "forward returned None"
        
    assert not (y.shape != (2, 32, 8, 8)), f"shape {y.shape}"
        
    # Verify output is finite and ReLU applied
    assert np.all(np.isfinite(y.data)), "output contains NaN or Inf"
    assert not (np.any(y.data < 0)), "ReLU not applied"
        
def test_residual_block_downsample():
    """Test ResidualBlock with downsampling."""
    np.random.seed(42)
    block = ResidualBlock(32, 64, stride=2)
        
    assert not (block.conv1 is None), "conv1 is None"
        
    x = Tensor(np.random.randn(2, 32, 8, 8))
    y = block(x)
        
    assert not (y is None), "forward returned None"
        
    assert not (y.shape != (2, 64, 4, 4)), f"shape {y.shape}"
        
    # Verify spatial dimensions reduced by stride=2
    assert not (y.shape[2] != x.shape[2] // 2), f"height not halved: {y.shape[2]} vs {x.shape[2]//2}"
    assert not (y.shape[3] != x.shape[3] // 2), f"width not halved: {y.shape[3]} vs {x.shape[3]//2}"
        
    # Verify output is finite
    assert np.all(np.isfinite(y.data)), "output contains NaN or Inf"
        
def test_lenet_forward():
    """Test LeNet forward pass."""
    np.random.seed(42)
    model = LeNet(in_channels=1, num_classes=10)
        
    assert not (model.conv1 is None), "conv1 is None"
        
    x = Tensor(np.random.randn(2, 1, 28, 28))
    y = model(x)
        
    assert not (y is None), "forward returned None"
        
    assert not (y.shape != (2, 10)), f"shape {y.shape}"
        
    # Verify output is finite
    assert np.all(np.isfinite(y.data)), "output contains NaN or Inf"
        
    # Output should be logits (unbounded)
    # Different inputs should produce different outputs
    x2 = Tensor(np.random.randn(2, 1, 28, 28))
    y2 = model(x2)
    assert not (np.allclose(y.data, y2.data)), "same output for different inputs"
        
def test_lenet_backward():
    """Test LeNet backward pass."""
    np.random.seed(42)
    model = LeNet(in_channels=1, num_classes=10)
        
    assert not (model.conv1 is None), "conv1 is None"
        
    x = Tensor(np.random.randn(2, 1, 28, 28))
    y = model(x)
        
    assert not (y is None), "forward returned None"
        
    y.sum().backward()
        
    params = model.parameters()
    has_grad = any(np.any(p.grad != 0) for p in params)
        
    assert has_grad, "no gradients"
        
    # Verify all gradients are finite
    for p in params:
        assert np.all(np.isfinite(p.grad)), "gradient contains NaN or Inf"
        
    # Verify input gradient exists and is finite
    assert np.all(np.isfinite(x.grad)), "input gradient contains NaN or Inf"
        
def test_simple_cnn_forward():
    """Test SimpleCNN forward pass."""
    np.random.seed(42)
    model = SimpleCNN(in_channels=3, num_classes=10)
        
    assert not (model.block1 is None), "block1 is None"
        
    x = Tensor(np.random.randn(2, 3, 32, 32))
    y = model(x)
        
    assert not (y is None), "forward returned None"
        
    assert not (y.shape != (2, 10)), f"shape {y.shape}"
        
    # Verify output is finite
    assert np.all(np.isfinite(y.data)), "output contains NaN or Inf"
        
    # Different inputs should give different outputs
    x2 = Tensor(np.random.randn(2, 3, 32, 32))
    y2 = model(x2)
    assert not (np.allclose(y.data, y2.data)), "same output for different inputs"
        
def test_global_avg_pool():
    """Test GlobalAvgPool layer."""
    np.random.seed(42)
    gap = GlobalAvgPool()
    x = Tensor(np.random.randn(2, 16, 4, 4))
    y = gap(x)
        
    assert not (y.shape != (2, 16)), f"shape {y.shape}"
        
    expected = np.mean(x.data, axis=(2, 3))
    assert np.allclose(y.data, expected), "values mismatch"
        
    y.sum().backward()
    assert not (x.grad.shape != x.shape), "grad shape mismatch"
        
    # Gradient should be uniform: 1 / (H * W) = 1 / 16
    expected_grad = 1.0 / (4 * 4)
    assert np.allclose(x.grad, expected_grad), f"grad {x.grad[0,0,0,0]} vs expected {expected_grad}"
        
def test_softmax():
    """Test softmax function."""
    x = np.array([[1, 2, 3], [1, 2, 3]])
    probs = softmax(x)
        
    assert np.allclose(probs.sum(axis=1), 1), "doesn't sum to 1"
        
    # Verify actual softmax values
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    expected = exp_x / exp_x.sum(axis=1, keepdims=True)
    assert np.allclose(probs, expected), f"values mismatch"
        
    # Third element should be largest
    assert np.all(probs[:, 2] > probs[:, 1]) or not np.all(probs[:, 1] > probs[:, 0]), "ordering incorrect"
        
    x_large = np.array([[1000, 1001, 1002]])
    probs_large = softmax(x_large)
    assert np.all(np.isfinite(probs_large)), "not numerically stable"
        
def test_cross_entropy_forward():
    """Test CrossEntropyLoss forward."""
    loss_fn = CrossEntropyLoss()
        
    logits = Tensor(np.array([[2.0, 1.0, 0.1],
                               [0.1, 2.0, 0.1]]))
    targets = np.array([0, 1])
        
    loss = loss_fn(logits, targets)
        
    assert not (loss is None), "returned None"
        
    assert not (loss.data.shape != ()), f"not scalar: {loss.data.shape}"
        
    assert not (loss.data < 0), "loss should be positive"
        
    # Manually compute expected loss
    probs = softmax(logits.data)
    expected_loss = -np.mean(np.log(probs[np.arange(2), targets]))
    assert np.isclose(loss.data, expected_loss, rtol=1e-5), f"loss {loss.data} vs expected {expected_loss}"
        
def test_cross_entropy_backward():
    """Test CrossEntropyLoss backward."""
    np.random.seed(42)
    loss_fn = CrossEntropyLoss()
        
    logits = Tensor(np.random.randn(4, 5))
    targets = np.array([0, 1, 2, 3])
        
    loss = loss_fn(logits, targets)
        
    assert not (loss is None), "forward returned None"
        
    loss.backward()
        
    assert not (np.all(logits.grad == 0)), "gradient is zero"
        
    assert np.allclose(logits.grad.sum(axis=1), 0, atol=1e-6), "grad rows should sum to ~0"
        
    # Verify gradient formula: grad = (softmax(logits) - one_hot(targets)) / N
    probs = softmax(logits.data)
    expected_grad = probs.copy()
    expected_grad[np.arange(4), targets] -= 1
    expected_grad /= 4
    assert np.allclose(logits.grad, expected_grad, rtol=1e-5), f"gradient mismatch, max diff: {np.max(np.abs(logits.grad - expected_grad))}"
        
def test_sgd_step():
    """Test SGD optimizer step."""
    w = Tensor(np.array([1.0, 2.0, 3.0]))
    optimizer = SGD([w], lr=0.1)
        
    w.grad = np.array([1.0, 1.0, 1.0])
    optimizer.step()
        
    expected = np.array([0.9, 1.9, 2.9])
    assert np.allclose(w.data, expected), f"data {w.data}"
        
def test_sgd_zero_grad():
    """Test SGD zero_grad."""
    w = Tensor(np.array([1.0, 2.0, 3.0]))
    w.grad = np.array([1.0, 1.0, 1.0])
        
    optimizer = SGD([w], lr=0.1)
    optimizer.zero_grad()
        
    assert np.allclose(w.grad, 0), "grad not zeroed"
        
def test_sgd_momentum():
    """Test SGD with momentum."""
    w = Tensor(np.array([1.0, 2.0, 3.0]))
    optimizer = SGD([w], lr=0.1, momentum=0.9)
        
    # First step: velocity = grad = [1,1,1], update = velocity
    w.grad = np.array([1.0, 1.0, 1.0])
    optimizer.step()
    # w = [1,2,3] - 0.1 * [1,1,1] = [0.9, 1.9, 2.9]
    expected_after_step1 = np.array([0.9, 1.9, 2.9])
    assert np.allclose(w.data, expected_after_step1), f"step 1: {w.data} vs {expected_after_step1}"
        
    # Second step: velocity = 0.9*[1,1,1] + [1,1,1] = [1.9,1.9,1.9]
    w.grad = np.array([1.0, 1.0, 1.0])
    optimizer.step()
    # w = [0.9,1.9,2.9] - 0.1 * [1.9,1.9,1.9] = [0.71, 1.71, 2.71]
    expected_after_step2 = np.array([0.71, 1.71, 2.71])
    assert np.allclose(w.data, expected_after_step2), f"step 2: {w.data} vs {expected_after_step2}"
        
def test_training_loop():
    """Test complete training loop."""
    np.random.seed(42)
        
    model = Sequential(
        Linear(10, 32),
        ReLU(),
        Linear(32, 5)
    )
        
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1)
        
    x = Tensor(np.random.randn(16, 10))
    targets = np.random.randint(0, 5, 16)
        
    initial_loss = None
    for i in range(20):
        optimizer.zero_grad()
            
        logits = model(x)
        loss = loss_fn(logits, targets)
            
        assert not (loss is None), "loss is None"
            
        if i == 0:
            initial_loss = loss.data
            
        loss.backward()
        optimizer.step()
        
    final_loss = loss.data
        
    assert not (final_loss >= initial_loss), f"loss didn't decrease: {initial_loss} -> {final_loss}"
        
def test_model_train_eval():
    """Test model train/eval mode switching."""
    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Dropout(0.5),
        Linear(20, 5)
    )
        
    model.train()
    for m in model._modules:
        assert not (hasattr(m, '_training') and not m._training), "not in train mode"
        
    model.eval()
    for m in model._modules:
        assert not (hasattr(m, '_training') and m._training), "not in eval mode"
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])