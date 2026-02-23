"""Test Suite for Day 35: Complete Autodiff Library"""

import numpy as np
import sys
from typing import Tuple

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

def test_tensor_creation() -> Tuple[bool, str]:
    """Test tensor creation."""
    try:
        t = Tensor([1, 2, 3])
        if t.shape != (3,):
            return False, f"shape {t.shape}"
        return True, "Tensor creation works"
    except Exception as e:
        return False, str(e)


def test_tensor_add() -> Tuple[bool, str]:
    """Test tensor addition."""
    try:
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        
        if not np.allclose(c.data, [5, 7, 9]):
            return False, f"values {c.data}"
        
        c.sum().backward()
        if not np.allclose(a.grad, [1, 1, 1]):
            return False, f"grad {a.grad}"
        
        return True, "Addition works"
    except Exception as e:
        return False, str(e)


def test_tensor_mul() -> Tuple[bool, str]:
    """Test tensor multiplication."""
    try:
        a = Tensor([2, 3, 4])
        b = Tensor([1, 2, 3])
        c = a * b
        
        if not np.allclose(c.data, [2, 6, 12]):
            return False, f"values {c.data}"
        
        c.sum().backward()
        if not np.allclose(a.grad, [1, 2, 3]):
            return False, f"grad {a.grad}"
        
        return True, "Multiplication works"
    except Exception as e:
        return False, str(e)


def test_tensor_matmul() -> Tuple[bool, str]:
    """Test matrix multiplication."""
    try:
        np.random.seed(42)
        a = Tensor(np.random.randn(3, 4))
        b = Tensor(np.random.randn(4, 5))
        c = a @ b
        
        if c.shape != (3, 5):
            return False, f"shape {c.shape}"
        
        c.sum().backward()
        if a.grad.shape != a.shape:
            return False, "grad shape mismatch"
        
        return True, "Matmul works"
    except Exception as e:
        return False, str(e)


def test_tensor_sum() -> Tuple[bool, str]:
    """Test tensor sum."""
    try:
        a = Tensor([[1, 2], [3, 4]])
        s = a.sum()
        
        if s.data != 10:
            return False, f"value {s.data}"
        
        s.backward()
        if not np.allclose(a.grad, [[1, 1], [1, 1]]):
            return False, f"grad {a.grad}"
        
        return True, "Sum works"
    except Exception as e:
        return False, str(e)


def test_tensor_mean() -> Tuple[bool, str]:
    """Test tensor mean."""
    try:
        a = Tensor([[2, 4], [6, 8]])
        m = a.mean()
        
        if m.data != 5:
            return False, f"value {m.data}"
        
        m.backward()
        if not np.allclose(a.grad, [[0.25, 0.25], [0.25, 0.25]]):
            return False, f"grad {a.grad}"
        
        return True, "Mean works"
    except Exception as e:
        return False, str(e)


def test_tensor_relu() -> Tuple[bool, str]:
    """Test ReLU activation."""
    try:
        a = Tensor([-2, -1, 0, 1, 2])
        r = a.relu()
        
        if not np.allclose(r.data, [0, 0, 0, 1, 2]):
            return False, f"values {r.data}"
        
        r.sum().backward()
        if not np.allclose(a.grad, [0, 0, 0, 1, 1]):
            return False, f"grad {a.grad}"
        
        return True, "ReLU works"
    except Exception as e:
        return False, str(e)


def test_tensor_reshape() -> Tuple[bool, str]:
    """Test tensor reshape."""
    try:
        a = Tensor(np.arange(12))
        b = a.reshape(3, 4)
        
        if b.shape != (3, 4):
            return False, f"shape {b.shape}"
        
        b.sum().backward()
        if a.grad.shape != (12,):
            return False, "grad shape mismatch"
        
        return True, "Reshape works"
    except Exception as e:
        return False, str(e)


# ============================================================================
# Module Tests
# ============================================================================

def test_parameter() -> Tuple[bool, str]:
    """Test Parameter class."""
    try:
        p = Parameter(np.random.randn(3, 4))
        
        if not isinstance(p, Tensor):
            return False, "not a Tensor"
        if p.shape != (3, 4):
            return False, f"shape {p.shape}"
        
        return True, "Parameter works"
    except Exception as e:
        return False, str(e)


def test_linear_layer() -> Tuple[bool, str]:
    """Test Linear layer."""
    try:
        np.random.seed(42)
        layer = Linear(10, 5)
        
        x = Tensor(np.random.randn(4, 10))
        y = layer(x)
        
        if y.shape != (4, 5):
            return False, f"shape {y.shape}"
        
        y.sum().backward()
        if np.all(layer.weight.grad == 0):
            return False, "no gradient"
        
        return True, "Linear layer works"
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
        
        params = list(model.parameters())
        if len(params) != 4:
            return False, f"param count {len(params)}"
        
        return True, "Sequential works"
    except Exception as e:
        return False, str(e)


def test_conv2d() -> Tuple[bool, str]:
    """Test Conv2d layer."""
    try:
        np.random.seed(42)
        conv = Conv2d(3, 16, kernel_size=3, padding=1)
        
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = conv(x)
        
        if y.shape != (2, 16, 8, 8):
            return False, f"shape {y.shape}"
        
        y.sum().backward()
        if np.all(conv.weight.grad == 0):
            return False, "no gradient"
        
        return True, "Conv2d works"
    except Exception as e:
        return False, str(e)


def test_maxpool2d() -> Tuple[bool, str]:
    """Test MaxPool2d layer."""
    try:
        pool = MaxPool2d(kernel_size=2)
        
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = pool(x)
        
        if y.shape != (2, 3, 4, 4):
            return False, f"shape {y.shape}"
        
        y.sum().backward()
        if x.grad.shape != x.shape:
            return False, "grad shape mismatch"
        
        return True, "MaxPool2d works"
    except Exception as e:
        return False, str(e)


def test_batchnorm2d() -> Tuple[bool, str]:
    """Test BatchNorm2d layer."""
    try:
        np.random.seed(42)
        bn = BatchNorm2d(16)
        bn._training = True
        
        x = Tensor(np.random.randn(4, 16, 8, 8))
        y = bn(x)
        
        if y.shape != (4, 16, 8, 8):
            return False, f"shape {y.shape}"
        
        mean = np.mean(y.data, axis=(0, 2, 3))
        if not np.allclose(mean, 0, atol=1e-5):
            return False, "not normalized"
        
        return True, "BatchNorm2d works"
    except Exception as e:
        return False, str(e)


def test_dropout() -> Tuple[bool, str]:
    """Test Dropout layer."""
    try:
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        dropout._training = True
        
        x = Tensor(np.ones((100, 100)))
        y = dropout(x)
        
        zero_ratio = np.mean(y.data == 0)
        if not (0.3 < zero_ratio < 0.7):
            return False, f"drop ratio {zero_ratio}"
        
        dropout._training = False
        y2 = dropout(x)
        if not np.allclose(y2.data, x.data):
            return False, "eval not identity"
        
        return True, "Dropout works"
    except Exception as e:
        return False, str(e)


# ============================================================================
# Loss Function Tests
# ============================================================================

def test_cross_entropy() -> Tuple[bool, str]:
    """Test CrossEntropyLoss."""
    try:
        loss_fn = CrossEntropyLoss()
        
        logits = Tensor(np.array([[2.0, 1.0, 0.1], [0.1, 2.0, 0.1]]))
        targets = np.array([0, 1])
        
        loss = loss_fn(logits, targets)
        
        if loss.data.shape != ():
            return False, "not scalar"
        if loss.data < 0:
            return False, "negative loss"
        
        loss.backward()
        if np.all(logits.grad == 0):
            return False, "no gradient"
        
        return True, "CrossEntropyLoss works"
    except Exception as e:
        return False, str(e)


def test_mse_loss() -> Tuple[bool, str]:
    """Test MSELoss."""
    try:
        loss_fn = MSELoss()
        
        pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = Tensor([[1.5, 2.5], [3.5, 4.5]])
        
        loss = loss_fn(pred, target)
        
        if loss.data.shape != ():
            return False, "not scalar"
        
        expected = 0.25
        if not np.allclose(loss.data, expected):
            return False, f"value {loss.data}"
        
        return True, "MSELoss works"
    except Exception as e:
        return False, str(e)


def test_softmax() -> Tuple[bool, str]:
    """Test softmax function."""
    try:
        x = np.array([[1, 2, 3], [1, 2, 3]])
        probs = softmax(x)
        
        if not np.allclose(probs.sum(axis=1), 1):
            return False, "doesn't sum to 1"
        
        x_large = np.array([[1000, 1001, 1002]])
        probs_large = softmax(x_large)
        if not np.all(np.isfinite(probs_large)):
            return False, "not numerically stable"
        
        return True, "Softmax works"
    except Exception as e:
        return False, str(e)


# ============================================================================
# Optimizer Tests
# ============================================================================

def test_sgd() -> Tuple[bool, str]:
    """Test SGD optimizer."""
    try:
        w = Tensor(np.array([1.0, 2.0, 3.0]))
        optimizer = SGD([w], lr=0.1)
        
        w.grad = np.array([1.0, 1.0, 1.0])
        optimizer.step()
        
        if not np.allclose(w.data, [0.9, 1.9, 2.9]):
            return False, f"values {w.data}"
        
        optimizer.zero_grad()
        if not np.allclose(w.grad, 0):
            return False, "grad not zeroed"
        
        return True, "SGD works"
    except Exception as e:
        return False, str(e)


def test_adam() -> Tuple[bool, str]:
    """Test Adam optimizer."""
    try:
        w = Tensor(np.array([1.0, 2.0, 3.0]))
        optimizer = Adam([w], lr=0.1)
        
        for _ in range(5):
            w.grad = np.array([1.0, 1.0, 1.0])
            optimizer.step()
        
        if np.allclose(w.data, [1, 2, 3]):
            return False, "no update"
        
        return True, "Adam works"
    except Exception as e:
        return False, str(e)


# ============================================================================
# Data Loading Tests
# ============================================================================

def test_synthetic_data() -> Tuple[bool, str]:
    """Test synthetic MNIST generation."""
    try:
        data = generate_synthetic_mnist(n_samples=100, n_classes=10)
        
        if data is None:
            return False, "not implemented"
        
        X, y = data
        
        if X.shape != (100, 1, 28, 28):
            return False, f"X shape {X.shape}"
        if y.shape != (100,):
            return False, f"y shape {y.shape}"
        
        if not np.all((y >= 0) & (y < 10)):
            return False, "invalid labels"
        
        return True, "Synthetic data works"
    except Exception as e:
        return False, str(e)


def test_data_loader() -> Tuple[bool, str]:
    """Test DataLoader."""
    try:
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 5, 100)
        
        loader = DataLoader(X, y, batch_size=32, shuffle=True)
        
        batches = list(loader)
        if len(batches) != 4:
            return False, f"batch count {len(batches)}"
        
        total_samples = sum(len(batch[1]) for batch in batches)
        if total_samples != 100:
            return False, f"sample count {total_samples}"
        
        return True, "DataLoader works"
    except Exception as e:
        return False, str(e)


# ============================================================================
# Complete Model Tests
# ============================================================================

def test_simple_cnn() -> Tuple[bool, str]:
    """Test SimpleCNN model."""
    try:
        np.random.seed(42)
        model = SimpleCNN(in_channels=1, num_classes=10)
        
        if model.conv1 is None:
            return False, "not implemented"
        
        x = Tensor(np.random.randn(2, 1, 28, 28))
        y = model(x)
        
        if y is None:
            return False, "forward returned None"
        if y.shape != (2, 10):
            return False, f"shape {y.shape}"
        
        return True, "SimpleCNN forward works"
    except Exception as e:
        return False, str(e)


def test_simple_cnn_backward() -> Tuple[bool, str]:
    """Test SimpleCNN backward pass."""
    try:
        np.random.seed(42)
        model = SimpleCNN(in_channels=1, num_classes=10)
        
        if model.conv1 is None:
            return False, "not implemented"
        
        x = Tensor(np.random.randn(2, 1, 28, 28))
        y = model(x)
        
        if y is None:
            return False, "forward returned None"
        
        loss_fn = CrossEntropyLoss()
        targets = np.array([0, 1])
        loss = loss_fn(y, targets)
        
        if loss is None:
            return False, "loss is None"
        
        loss.backward()
        
        has_grad = any(np.any(p.grad != 0) for p in model.parameters())
        if not has_grad:
            return False, "no gradients"
        
        return True, "SimpleCNN backward works"
    except Exception as e:
        return False, str(e)


def test_training_step() -> Tuple[bool, str]:
    """Test a complete training step."""
    try:
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
        
        if loss.data >= initial_loss_value:
            return False, f"loss didn't decrease: {initial_loss_value:.4f} -> {loss.data:.4f}"
        
        return True, f"Loss: {initial_loss_value:.4f} -> {loss.data:.4f}"
    except Exception as e:
        return False, str(e)


def test_full_training() -> Tuple[bool, str]:
    """Test full training loop."""
    try:
        np.random.seed(42)
        
        data = generate_synthetic_mnist(n_samples=200, n_classes=10)
        if data is None:
            return False, "synthetic data not implemented"
        
        X, y = data
        
        train_loader = DataLoader(X[:160], y[:160], batch_size=32)
        val_loader = DataLoader(X[160:], y[160:], batch_size=32)
        
        model = SimpleCNN(in_channels=1, num_classes=10)
        if model.conv1 is None:
            return False, "SimpleCNN not implemented"
        
        loss_fn = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        
        initial_loss, _ = evaluate(model, val_loader, loss_fn)
        
        for _ in range(3):
            train_epoch(model, train_loader, loss_fn, optimizer)
        
        final_loss, accuracy = evaluate(model, val_loader, loss_fn)
        
        if final_loss >= initial_loss:
            return False, "loss didn't decrease"
        
        return True, f"Accuracy: {accuracy:.2%}"
    except Exception as e:
        return False, str(e)


def test_im2col_col2im() -> Tuple[bool, str]:
    """Test im2col and col2im functions."""
    try:
        x = np.random.randn(2, 3, 8, 8)
        
        col = im2col(x, 3, 3, stride=1, padding=1)
        
        if col.shape[0] != 2 * 8 * 8:
            return False, f"col shape {col.shape}"
        
        x_reconstructed = col2im(col, x.shape, 3, 3, stride=1, padding=1)
        
        if x_reconstructed.shape != x.shape:
            return False, "reconstruction shape mismatch"
        
        return True, "im2col/col2im work"
    except Exception as e:
        return False, str(e)


def test_model_train_eval_mode() -> Tuple[bool, str]:
    """Test train/eval mode switching."""
    try:
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Dropout(0.5),
            Linear(20, 5)
        )
        
        model.train()
        for m in model._modules.values():
            if hasattr(m, '_training') and not m._training:
                return False, "not in train mode"
        
        model.eval()
        for m in model._modules.values():
            if hasattr(m, '_training') and m._training:
                return False, "not in eval mode"
        
        return True, "Train/eval mode works"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("tensor_creation", test_tensor_creation),
        ("tensor_add", test_tensor_add),
        ("tensor_mul", test_tensor_mul),
        ("tensor_matmul", test_tensor_matmul),
        ("tensor_sum", test_tensor_sum),
        ("tensor_mean", test_tensor_mean),
        ("tensor_relu", test_tensor_relu),
        ("tensor_reshape", test_tensor_reshape),
        ("parameter", test_parameter),
        ("linear_layer", test_linear_layer),
        ("sequential", test_sequential),
        ("conv2d", test_conv2d),
        ("maxpool2d", test_maxpool2d),
        ("batchnorm2d", test_batchnorm2d),
        ("dropout", test_dropout),
        ("cross_entropy", test_cross_entropy),
        ("mse_loss", test_mse_loss),
        ("softmax", test_softmax),
        ("sgd", test_sgd),
        ("adam", test_adam),
        ("synthetic_data", test_synthetic_data),
        ("data_loader", test_data_loader),
        ("simple_cnn", test_simple_cnn),
        ("simple_cnn_backward", test_simple_cnn_backward),
        ("training_step", test_training_step),
        ("full_training", test_full_training),
        ("im2col_col2im", test_im2col_col2im),
        ("model_train_eval_mode", test_model_train_eval_mode),
    ]
    
    print(f"\n{'='*70}")
    print("Day 35: Complete Autodiff Library - Final Test Suite")
    print(f"{'='*70}")
    
    passed = 0
    for name, fn in tests:
        try:
            p, m = fn()
        except Exception as e:
            p, m = False, str(e)
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    
    print(f"\n{'='*70}")
    print(f"Summary: {passed}/{len(tests)} tests passed")
    print(f"{'='*70}")
    
    if passed == len(tests):
        print("\nCongratulations! All tests passed!")
        print("You have successfully built a complete autodiff library from scratch!")
    elif passed >= len(tests) * 0.8:
        print("\nGreat progress! Most tests passing.")
        print("Complete the remaining TODOs to finish the project.")
    else:
        print("\nKeep working on the implementations.")
        print("Review the TODO sections in day35.py")
    
    return passed == len(tests)


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    success = run_all_tests()
    sys.exit(0 if success else 1)
