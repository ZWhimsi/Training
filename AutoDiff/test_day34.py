"""Test Suite for Day 34: Complete CNN Module"""

import numpy as np
import sys
from typing import Tuple

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


def test_relu_forward() -> Tuple[bool, str]:
    """Test ReLU forward pass."""
    try:
        relu = ReLU()
        x = Tensor(np.array([-2, -1, 0, 1, 2]))
        y = relu(x)
        
        expected = np.array([0, 0, 0, 1, 2])
        if not np.allclose(y.data, expected):
            return False, f"values {y.data}"
        
        return True, "ReLU forward works"
    except Exception as e:
        return False, str(e)


def test_relu_backward() -> Tuple[bool, str]:
    """Test ReLU backward pass."""
    try:
        relu = ReLU()
        x = Tensor(np.array([-2.0, -1.0, 0.5, 1.0, 2.0]))
        y = relu(x)
        y.sum().backward()
        
        expected_grad = np.array([0, 0, 1, 1, 1])
        if not np.allclose(x.grad, expected_grad):
            return False, f"grad {x.grad}"
        
        return True, "ReLU backward works"
    except Exception as e:
        return False, str(e)


def test_flatten() -> Tuple[bool, str]:
    """Test Flatten layer."""
    try:
        np.random.seed(42)
        flatten = Flatten()
        x = Tensor(np.random.randn(2, 3, 4, 4))
        y = flatten(x)
        
        if y.shape != (2, 48):
            return False, f"shape {y.shape}"
        
        # Verify values are preserved (just reshaped)
        expected = x.data.reshape(2, -1)
        if not np.allclose(y.data, expected):
            return False, "values not preserved during flatten"
        
        y.sum().backward()
        if x.grad.shape != x.shape:
            return False, "grad shape mismatch"
        
        # For sum loss, gradient should be all ones
        if not np.allclose(x.grad, 1.0):
            return False, f"gradient {x.grad[0,0,0,0]}, expected 1.0"
        
        return True, "Flatten works"
    except Exception as e:
        return False, str(e)


def test_linear_forward() -> Tuple[bool, str]:
    """Test Linear layer forward."""
    try:
        np.random.seed(42)
        linear = Linear(10, 5)
        
        x = Tensor(np.random.randn(4, 10))
        y = linear(x)
        
        if y.shape != (4, 5):
            return False, f"shape {y.shape}"
        
        # Verify linear computation: y = x @ W^T + b
        expected = x.data @ linear.weight.data.T + linear.bias.data
        if not np.allclose(y.data, expected, rtol=1e-5):
            return False, f"output mismatch, max diff: {np.max(np.abs(y.data - expected))}"
        
        return True, "Linear forward works"
    except Exception as e:
        return False, str(e)


def test_linear_backward() -> Tuple[bool, str]:
    """Test Linear layer backward."""
    try:
        np.random.seed(42)
        linear = Linear(10, 5)
        
        x = Tensor(np.random.randn(4, 10))
        y = linear(x)
        y.sum().backward()
        
        if np.all(linear.weight.grad == 0):
            return False, "weight grad is zero"
        if np.all(linear.bias.grad == 0):
            return False, "bias grad is zero"
        
        # For sum loss, bias gradient = batch_size = 4
        expected_bias_grad = 4.0
        if not np.allclose(linear.bias.grad, expected_bias_grad):
            return False, f"bias grad {linear.bias.grad[0]}, expected {expected_bias_grad}"
        
        # Weight gradient = sum of outer products = sum(x) for each output
        expected_weight_grad = np.ones((4, 5)).T @ x.data  # (5, 10)
        if not np.allclose(linear.weight.grad, expected_weight_grad, rtol=1e-5):
            return False, "weight grad mismatch"
        
        return True, "Linear backward works"
    except Exception as e:
        return False, str(e)


def test_dropout_training() -> Tuple[bool, str]:
    """Test Dropout in training mode."""
    try:
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        dropout._training = True
        
        x = Tensor(np.ones((100, 100)))
        y = dropout(x)
        
        zero_ratio = np.mean(y.data == 0)
        if not (0.3 < zero_ratio < 0.7):
            return False, f"drop ratio {zero_ratio}"
        
        # Non-zero values should be scaled by 1/(1-p) = 2
        non_zero_vals = y.data[y.data != 0]
        if not np.allclose(non_zero_vals, 2.0, rtol=1e-5):
            return False, f"non-zero values should be 2.0, got {non_zero_vals[0]}"
        
        return True, "Dropout training works"
    except Exception as e:
        return False, str(e)


def test_dropout_eval() -> Tuple[bool, str]:
    """Test Dropout in eval mode."""
    try:
        dropout = Dropout(p=0.5)
        dropout._training = False
        
        x = Tensor(np.ones((10, 10)) * 3.0)
        y = dropout(x)
        
        if not np.allclose(y.data, x.data):
            return False, "should be identity in eval"
        
        # Explicitly verify value preservation
        if not np.allclose(y.data, 3.0):
            return False, f"values not preserved: {y.data[0,0]} vs 3.0"
        
        return True, "Dropout eval works"
    except Exception as e:
        return False, str(e)


def test_sequential() -> Tuple[bool, str]:
    """Test Sequential container."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
        
        x = Tensor(np.random.randn(4, 10))
        y = model(x)
        
        if y.shape != (4, 5):
            return False, f"shape {y.shape}"
        
        params = model.parameters()
        if len(params) != 4:
            return False, f"params count {len(params)}"
        
        # Verify output is finite
        if not np.all(np.isfinite(y.data)):
            return False, "output contains NaN or Inf"
        
        # Verify backward works
        y.sum().backward()
        for p in params:
            if not np.all(np.isfinite(p.grad)):
                return False, "gradient contains NaN or Inf"
        
        return True, "Sequential works"
    except Exception as e:
        return False, str(e)


def test_conv_block_forward() -> Tuple[bool, str]:
    """Test ConvBlock forward pass."""
    try:
        np.random.seed(42)
        block = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        
        if block.conv is None:
            return False, "conv is None (not implemented)"
        
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = block(x)
        
        if y is None:
            return False, "forward returned None"
        
        if y.shape != (2, 32, 8, 8):
            return False, f"shape {y.shape}"
        
        # Verify ReLU is applied (no negative values)
        if np.any(y.data < 0):
            return False, "ReLU not applied (found negative values)"
        
        # Verify output is finite
        if not np.all(np.isfinite(y.data)):
            return False, "output contains NaN or Inf"
        
        return True, "ConvBlock forward works"
    except Exception as e:
        return False, str(e)


def test_conv_block_backward() -> Tuple[bool, str]:
    """Test ConvBlock backward pass."""
    try:
        np.random.seed(42)
        block = ConvBlock(3, 16, kernel_size=3, stride=1, padding=1)
        
        if block.conv is None:
            return False, "conv is None"
        
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = block(x)
        
        if y is None:
            return False, "forward returned None"
        
        y.sum().backward()
        
        params = block.parameters()
        if not params:
            return False, "no parameters"
        
        has_grad = any(np.any(p.grad != 0) for p in params)
        if not has_grad:
            return False, "no gradients"
        
        # Verify gradients are finite
        for p in params:
            if not np.all(np.isfinite(p.grad)):
                return False, "parameter gradient contains NaN or Inf"
        if not np.all(np.isfinite(x.grad)):
            return False, "input gradient contains NaN or Inf"
        
        return True, "ConvBlock backward works"
    except Exception as e:
        return False, str(e)


def test_residual_block_same_dim() -> Tuple[bool, str]:
    """Test ResidualBlock with same dimensions."""
    try:
        np.random.seed(42)
        block = ResidualBlock(32, 32, stride=1)
        
        if block.conv1 is None:
            return False, "conv1 is None"
        
        x = Tensor(np.random.randn(2, 32, 8, 8))
        y = block(x)
        
        if y is None:
            return False, "forward returned None"
        
        if y.shape != (2, 32, 8, 8):
            return False, f"shape {y.shape}"
        
        # Verify output is finite and ReLU applied
        if not np.all(np.isfinite(y.data)):
            return False, "output contains NaN or Inf"
        if np.any(y.data < 0):
            return False, "ReLU not applied"
        
        return True, "ResidualBlock same dim works"
    except Exception as e:
        return False, str(e)


def test_residual_block_downsample() -> Tuple[bool, str]:
    """Test ResidualBlock with downsampling."""
    try:
        np.random.seed(42)
        block = ResidualBlock(32, 64, stride=2)
        
        if block.conv1 is None:
            return False, "conv1 is None"
        
        x = Tensor(np.random.randn(2, 32, 8, 8))
        y = block(x)
        
        if y is None:
            return False, "forward returned None"
        
        if y.shape != (2, 64, 4, 4):
            return False, f"shape {y.shape}"
        
        # Verify spatial dimensions reduced by stride=2
        if y.shape[2] != x.shape[2] // 2:
            return False, f"height not halved: {y.shape[2]} vs {x.shape[2]//2}"
        if y.shape[3] != x.shape[3] // 2:
            return False, f"width not halved: {y.shape[3]} vs {x.shape[3]//2}"
        
        # Verify output is finite
        if not np.all(np.isfinite(y.data)):
            return False, "output contains NaN or Inf"
        
        return True, "ResidualBlock downsample works"
    except Exception as e:
        return False, str(e)


def test_lenet_forward() -> Tuple[bool, str]:
    """Test LeNet forward pass."""
    try:
        np.random.seed(42)
        model = LeNet(in_channels=1, num_classes=10)
        
        if model.conv1 is None:
            return False, "conv1 is None"
        
        x = Tensor(np.random.randn(2, 1, 28, 28))
        y = model(x)
        
        if y is None:
            return False, "forward returned None"
        
        if y.shape != (2, 10):
            return False, f"shape {y.shape}"
        
        # Verify output is finite
        if not np.all(np.isfinite(y.data)):
            return False, "output contains NaN or Inf"
        
        # Output should be logits (unbounded)
        # Different inputs should produce different outputs
        x2 = Tensor(np.random.randn(2, 1, 28, 28))
        y2 = model(x2)
        if np.allclose(y.data, y2.data):
            return False, "same output for different inputs"
        
        return True, "LeNet forward works"
    except Exception as e:
        return False, str(e)


def test_lenet_backward() -> Tuple[bool, str]:
    """Test LeNet backward pass."""
    try:
        np.random.seed(42)
        model = LeNet(in_channels=1, num_classes=10)
        
        if model.conv1 is None:
            return False, "conv1 is None"
        
        x = Tensor(np.random.randn(2, 1, 28, 28))
        y = model(x)
        
        if y is None:
            return False, "forward returned None"
        
        y.sum().backward()
        
        params = model.parameters()
        has_grad = any(np.any(p.grad != 0) for p in params)
        
        if not has_grad:
            return False, "no gradients"
        
        # Verify all gradients are finite
        for p in params:
            if not np.all(np.isfinite(p.grad)):
                return False, "gradient contains NaN or Inf"
        
        # Verify input gradient exists and is finite
        if not np.all(np.isfinite(x.grad)):
            return False, "input gradient contains NaN or Inf"
        
        return True, "LeNet backward works"
    except Exception as e:
        return False, str(e)


def test_simple_cnn_forward() -> Tuple[bool, str]:
    """Test SimpleCNN forward pass."""
    try:
        np.random.seed(42)
        model = SimpleCNN(in_channels=3, num_classes=10)
        
        if model.block1 is None:
            return False, "block1 is None"
        
        x = Tensor(np.random.randn(2, 3, 32, 32))
        y = model(x)
        
        if y is None:
            return False, "forward returned None"
        
        if y.shape != (2, 10):
            return False, f"shape {y.shape}"
        
        # Verify output is finite
        if not np.all(np.isfinite(y.data)):
            return False, "output contains NaN or Inf"
        
        # Different inputs should give different outputs
        x2 = Tensor(np.random.randn(2, 3, 32, 32))
        y2 = model(x2)
        if np.allclose(y.data, y2.data):
            return False, "same output for different inputs"
        
        return True, "SimpleCNN forward works"
    except Exception as e:
        return False, str(e)


def test_global_avg_pool() -> Tuple[bool, str]:
    """Test GlobalAvgPool layer."""
    try:
        np.random.seed(42)
        gap = GlobalAvgPool()
        x = Tensor(np.random.randn(2, 16, 4, 4))
        y = gap(x)
        
        if y.shape != (2, 16):
            return False, f"shape {y.shape}"
        
        expected = np.mean(x.data, axis=(2, 3))
        if not np.allclose(y.data, expected):
            return False, "values mismatch"
        
        y.sum().backward()
        if x.grad.shape != x.shape:
            return False, "grad shape mismatch"
        
        # Gradient should be uniform: 1 / (H * W) = 1 / 16
        expected_grad = 1.0 / (4 * 4)
        if not np.allclose(x.grad, expected_grad):
            return False, f"grad {x.grad[0,0,0,0]} vs expected {expected_grad}"
        
        return True, "GlobalAvgPool works"
    except Exception as e:
        return False, str(e)


def test_softmax() -> Tuple[bool, str]:
    """Test softmax function."""
    try:
        x = np.array([[1, 2, 3], [1, 2, 3]])
        probs = softmax(x)
        
        if not np.allclose(probs.sum(axis=1), 1):
            return False, "doesn't sum to 1"
        
        # Verify actual softmax values
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        expected = exp_x / exp_x.sum(axis=1, keepdims=True)
        if not np.allclose(probs, expected):
            return False, f"values mismatch"
        
        # Third element should be largest
        if not np.all(probs[:, 2] > probs[:, 1]) or not np.all(probs[:, 1] > probs[:, 0]):
            return False, "ordering incorrect"
        
        x_large = np.array([[1000, 1001, 1002]])
        probs_large = softmax(x_large)
        if not np.all(np.isfinite(probs_large)):
            return False, "not numerically stable"
        
        return True, "Softmax works"
    except Exception as e:
        return False, str(e)


def test_cross_entropy_forward() -> Tuple[bool, str]:
    """Test CrossEntropyLoss forward."""
    try:
        loss_fn = CrossEntropyLoss()
        
        logits = Tensor(np.array([[2.0, 1.0, 0.1],
                                   [0.1, 2.0, 0.1]]))
        targets = np.array([0, 1])
        
        loss = loss_fn(logits, targets)
        
        if loss is None:
            return False, "returned None"
        
        if loss.data.shape != ():
            return False, f"not scalar: {loss.data.shape}"
        
        if loss.data < 0:
            return False, "loss should be positive"
        
        # Manually compute expected loss
        probs = softmax(logits.data)
        expected_loss = -np.mean(np.log(probs[np.arange(2), targets]))
        if not np.isclose(loss.data, expected_loss, rtol=1e-5):
            return False, f"loss {loss.data} vs expected {expected_loss}"
        
        return True, "CrossEntropy forward works"
    except Exception as e:
        return False, str(e)


def test_cross_entropy_backward() -> Tuple[bool, str]:
    """Test CrossEntropyLoss backward."""
    try:
        np.random.seed(42)
        loss_fn = CrossEntropyLoss()
        
        logits = Tensor(np.random.randn(4, 5))
        targets = np.array([0, 1, 2, 3])
        
        loss = loss_fn(logits, targets)
        
        if loss is None:
            return False, "forward returned None"
        
        loss.backward()
        
        if np.all(logits.grad == 0):
            return False, "gradient is zero"
        
        if not np.allclose(logits.grad.sum(axis=1), 0, atol=1e-6):
            return False, "grad rows should sum to ~0"
        
        # Verify gradient formula: grad = (softmax(logits) - one_hot(targets)) / N
        probs = softmax(logits.data)
        expected_grad = probs.copy()
        expected_grad[np.arange(4), targets] -= 1
        expected_grad /= 4
        if not np.allclose(logits.grad, expected_grad, rtol=1e-5):
            return False, f"gradient mismatch, max diff: {np.max(np.abs(logits.grad - expected_grad))}"
        
        return True, "CrossEntropy backward works"
    except Exception as e:
        return False, str(e)


def test_sgd_step() -> Tuple[bool, str]:
    """Test SGD optimizer step."""
    try:
        w = Tensor(np.array([1.0, 2.0, 3.0]))
        optimizer = SGD([w], lr=0.1)
        
        w.grad = np.array([1.0, 1.0, 1.0])
        optimizer.step()
        
        expected = np.array([0.9, 1.9, 2.9])
        if not np.allclose(w.data, expected):
            return False, f"data {w.data}"
        
        return True, "SGD step works"
    except Exception as e:
        return False, str(e)


def test_sgd_zero_grad() -> Tuple[bool, str]:
    """Test SGD zero_grad."""
    try:
        w = Tensor(np.array([1.0, 2.0, 3.0]))
        w.grad = np.array([1.0, 1.0, 1.0])
        
        optimizer = SGD([w], lr=0.1)
        optimizer.zero_grad()
        
        if not np.allclose(w.grad, 0):
            return False, "grad not zeroed"
        
        return True, "SGD zero_grad works"
    except Exception as e:
        return False, str(e)


def test_sgd_momentum() -> Tuple[bool, str]:
    """Test SGD with momentum."""
    try:
        w = Tensor(np.array([1.0, 2.0, 3.0]))
        optimizer = SGD([w], lr=0.1, momentum=0.9)
        
        # First step: velocity = grad = [1,1,1], update = velocity
        w.grad = np.array([1.0, 1.0, 1.0])
        optimizer.step()
        # w = [1,2,3] - 0.1 * [1,1,1] = [0.9, 1.9, 2.9]
        expected_after_step1 = np.array([0.9, 1.9, 2.9])
        if not np.allclose(w.data, expected_after_step1):
            return False, f"step 1: {w.data} vs {expected_after_step1}"
        
        # Second step: velocity = 0.9*[1,1,1] + [1,1,1] = [1.9,1.9,1.9]
        w.grad = np.array([1.0, 1.0, 1.0])
        optimizer.step()
        # w = [0.9,1.9,2.9] - 0.1 * [1.9,1.9,1.9] = [0.71, 1.71, 2.71]
        expected_after_step2 = np.array([0.71, 1.71, 2.71])
        if not np.allclose(w.data, expected_after_step2):
            return False, f"step 2: {w.data} vs {expected_after_step2}"
        
        return True, "SGD momentum works"
    except Exception as e:
        return False, str(e)


def test_training_loop() -> Tuple[bool, str]:
    """Test complete training loop."""
    try:
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
            
            if loss is None:
                return False, "loss is None"
            
            if i == 0:
                initial_loss = loss.data
            
            loss.backward()
            optimizer.step()
        
        final_loss = loss.data
        
        if final_loss >= initial_loss:
            return False, f"loss didn't decrease: {initial_loss} -> {final_loss}"
        
        return True, f"Loss: {initial_loss:.4f} -> {final_loss:.4f}"
    except Exception as e:
        return False, str(e)


def test_model_train_eval() -> Tuple[bool, str]:
    """Test model train/eval mode switching."""
    try:
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Dropout(0.5),
            Linear(20, 5)
        )
        
        model.train()
        for m in model._modules:
            if hasattr(m, '_training') and not m._training:
                return False, "not in train mode"
        
        model.eval()
        for m in model._modules:
            if hasattr(m, '_training') and m._training:
                return False, "not in eval mode"
        
        return True, "Train/eval mode works"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("relu_forward", test_relu_forward),
        ("relu_backward", test_relu_backward),
        ("flatten", test_flatten),
        ("linear_forward", test_linear_forward),
        ("linear_backward", test_linear_backward),
        ("dropout_training", test_dropout_training),
        ("dropout_eval", test_dropout_eval),
        ("sequential", test_sequential),
        ("conv_block_forward", test_conv_block_forward),
        ("conv_block_backward", test_conv_block_backward),
        ("residual_block_same_dim", test_residual_block_same_dim),
        ("residual_block_downsample", test_residual_block_downsample),
        ("lenet_forward", test_lenet_forward),
        ("lenet_backward", test_lenet_backward),
        ("simple_cnn_forward", test_simple_cnn_forward),
        ("global_avg_pool", test_global_avg_pool),
        ("softmax", test_softmax),
        ("cross_entropy_forward", test_cross_entropy_forward),
        ("cross_entropy_backward", test_cross_entropy_backward),
        ("sgd_step", test_sgd_step),
        ("sgd_zero_grad", test_sgd_zero_grad),
        ("sgd_momentum", test_sgd_momentum),
        ("training_loop", test_training_loop),
        ("model_train_eval", test_model_train_eval),
    ]
    
    print(f"\n{'='*60}")
    print("Day 34: Complete CNN Module - Tests")
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
