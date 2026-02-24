"""Test Suite for Day 32: Pooling Operations"""

import numpy as np
import sys
from typing import Tuple

try:
    from day32 import (
        Tensor,
        maxpool2d_forward,
        maxpool2d_backward,
        avgpool2d_forward,
        avgpool2d_backward,
        MaxPool2d,
        AvgPool2d,
        GlobalAvgPool2d,
        AdaptiveAvgPool2d,
        Module
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_maxpool_forward_basic() -> Tuple[bool, str]:
    """Test max pooling forward basic case."""
    try:
        x = np.array([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]]).astype(np.float64)
        
        result = maxpool2d_forward(x, kernel_size=2, stride=2)
        
        if result is None:
            return False, "maxpool2d_forward returned None"
        
        out, mask = result
        expected = np.array([[[[6, 8], [14, 16]]]])
        
        if not np.allclose(out, expected):
            return False, f"Values mismatch: {out}"
        
        return True, "Max pooling forward works"
    except Exception as e:
        return False, str(e)


def test_maxpool_forward_shape() -> Tuple[bool, str]:
    """Test max pooling output shape."""
    try:
        np.random.seed(42)
        x = np.random.randn(4, 8, 16, 16)
        result = maxpool2d_forward(x, kernel_size=2, stride=2)
        
        if result is None:
            return False, "returned None"
        
        out, mask = result
        
        if out.shape != (4, 8, 8, 8):
            return False, f"shape {out.shape}, expected (4, 8, 8, 8)"
        
        # Verify max pooling property: output should be >= all values in window
        # Check a specific window
        window = x[0, 0, 0:2, 0:2]
        expected_max = np.max(window)
        if not np.isclose(out[0, 0, 0, 0], expected_max):
            return False, f"output {out[0,0,0,0]} != max {expected_max}"
        
        return True, "Output shape correct"
    except Exception as e:
        return False, str(e)


def test_maxpool_stride() -> Tuple[bool, str]:
    """Test max pooling with different stride."""
    try:
        np.random.seed(42)
        x = np.arange(36).reshape(1, 1, 6, 6).astype(np.float64)
        result = maxpool2d_forward(x, kernel_size=2, stride=1)
        
        if result is None:
            return False, "returned None"
        
        out, mask = result
        
        if out.shape != (1, 1, 5, 5):
            return False, f"shape {out.shape}, expected (1, 1, 5, 5)"
        
        # Verify values: with stride=1, windows overlap
        # First window [0,1,6,7] -> max=7
        if not np.isclose(out[0, 0, 0, 0], 7.0):
            return False, f"position (0,0) = {out[0,0,0,0]}, expected 7"
        # Window at (0,1) is [1,2,7,8] -> max=8
        if not np.isclose(out[0, 0, 0, 1], 8.0):
            return False, f"position (0,1) = {out[0,0,0,1]}, expected 8"
        
        return True, "Stride works correctly"
    except Exception as e:
        return False, str(e)


def test_maxpool_backward_gradient() -> Tuple[bool, str]:
    """Test max pooling backward passes gradient to max elements."""
    try:
        x = np.array([[[[1, 2], [3, 4]]]]).astype(np.float64)
        result = maxpool2d_forward(x, kernel_size=2, stride=2)
        
        if result is None:
            return False, "forward returned None"
        
        out, mask = result
        dy = np.array([[[[1.0]]]])
        
        dx = maxpool2d_backward(dy, mask, x.shape, kernel_size=2, stride=2)
        
        if dx is None:
            return False, "backward returned None"
        
        expected = np.array([[[[0, 0], [0, 1]]]])
        
        if not np.allclose(dx, expected):
            return False, f"gradient {dx}, expected {expected}"
        
        return True, "Gradient flows to max element"
    except Exception as e:
        return False, str(e)


def test_maxpool_backward_shape() -> Tuple[bool, str]:
    """Test max pooling backward output shape."""
    try:
        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 8)
        result = maxpool2d_forward(x, kernel_size=2, stride=2)
        
        if result is None:
            return False, "forward returned None"
        
        out, mask = result
        dy = np.random.randn(*out.shape)
        
        dx = maxpool2d_backward(dy, mask, x.shape, kernel_size=2, stride=2)
        
        if dx is None:
            return False, "backward returned None"
        
        if dx.shape != x.shape:
            return False, f"shape {dx.shape}, expected {x.shape}"
        
        # Verify sparsity: only max positions should have non-zero gradients
        non_zero_count = np.sum(dx != 0)
        expected_count = out.size  # One gradient per output position
        if non_zero_count != expected_count:
            return False, f"non-zero count {non_zero_count}, expected {expected_count}"
        
        return True, "Backward shape correct"
    except Exception as e:
        return False, str(e)


def test_maxpool2d_module() -> Tuple[bool, str]:
    """Test MaxPool2d module end-to-end."""
    try:
        np.random.seed(42)
        pool = MaxPool2d(kernel_size=2, stride=2)
        
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = pool(x)
        
        if y is None:
            return False, "forward returned None"
        
        if y.shape != (2, 3, 4, 4):
            return False, f"shape {y.shape}"
        
        # Verify max pooling values
        window = x.data[0, 0, 0:2, 0:2]
        expected_max = np.max(window)
        if not np.isclose(y.data[0, 0, 0, 0], expected_max):
            return False, f"max value mismatch: {y.data[0,0,0,0]} vs {expected_max}"
        
        loss = y.sum()
        loss.backward()
        
        if x.grad.shape != x.shape:
            return False, "gradient shape mismatch"
        
        # Verify gradient sparsity for max pooling
        non_zero_grads = np.sum(x.grad != 0)
        if non_zero_grads != y.data.size:
            return False, f"gradient sparsity: {non_zero_grads} non-zero, expected {y.data.size}"
        
        return True, "MaxPool2d module works"
    except Exception as e:
        return False, str(e)


def test_avgpool_forward_basic() -> Tuple[bool, str]:
    """Test average pooling forward basic case."""
    try:
        x = np.array([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]]).astype(np.float64)
        
        out = avgpool2d_forward(x, kernel_size=2, stride=2)
        
        if out is None:
            return False, "avgpool2d_forward returned None"
        
        expected = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])
        
        if not np.allclose(out, expected):
            return False, f"Values mismatch: {out}"
        
        return True, "Average pooling forward works"
    except Exception as e:
        return False, str(e)


def test_avgpool_forward_shape() -> Tuple[bool, str]:
    """Test average pooling output shape."""
    try:
        np.random.seed(42)
        x = np.random.randn(4, 8, 16, 16)
        out = avgpool2d_forward(x, kernel_size=2, stride=2)
        
        if out is None:
            return False, "returned None"
        
        if out.shape != (4, 8, 8, 8):
            return False, f"shape {out.shape}, expected (4, 8, 8, 8)"
        
        # Verify average pooling value
        window = x[0, 0, 0:2, 0:2]
        expected_avg = np.mean(window)
        if not np.isclose(out[0, 0, 0, 0], expected_avg):
            return False, f"output {out[0,0,0,0]} != mean {expected_avg}"
        
        return True, "Output shape correct"
    except Exception as e:
        return False, str(e)


def test_avgpool_backward_uniform() -> Tuple[bool, str]:
    """Test average pooling distributes gradient uniformly."""
    try:
        x_shape = (1, 1, 4, 4)
        dy = np.array([[[[4.0, 4.0], [4.0, 4.0]]]])
        
        dx = avgpool2d_backward(dy, x_shape, kernel_size=2, stride=2)
        
        if dx is None:
            return False, "backward returned None"
        
        if not np.allclose(dx, 1.0):
            return False, f"gradient not uniform: {dx}"
        
        return True, "Gradient distributed uniformly"
    except Exception as e:
        return False, str(e)


def test_avgpool_backward_shape() -> Tuple[bool, str]:
    """Test average pooling backward output shape."""
    try:
        x_shape = (2, 4, 8, 8)
        dy = np.ones((2, 4, 4, 4))  # Use ones for predictable gradient
        
        dx = avgpool2d_backward(dy, x_shape, kernel_size=2, stride=2)
        
        if dx is None:
            return False, "backward returned None"
        
        if dx.shape != x_shape:
            return False, f"shape {dx.shape}, expected {x_shape}"
        
        # With ones gradient and 2x2 pool, each input should get 1/4 = 0.25
        expected_grad = 0.25
        if not np.allclose(dx, expected_grad):
            return False, f"gradient {dx[0,0,0,0]}, expected {expected_grad}"
        
        return True, "Backward shape correct"
    except Exception as e:
        return False, str(e)


def test_avgpool2d_module() -> Tuple[bool, str]:
    """Test AvgPool2d module end-to-end."""
    try:
        np.random.seed(42)
        pool = AvgPool2d(kernel_size=2, stride=2)
        
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = pool(x)
        
        if y is None:
            return False, "forward returned None"
        
        if y.shape != (2, 3, 4, 4):
            return False, f"shape {y.shape}"
        
        # Verify average pooling value
        window = x.data[0, 0, 0:2, 0:2]
        expected_avg = np.mean(window)
        if not np.isclose(y.data[0, 0, 0, 0], expected_avg):
            return False, f"avg mismatch: {y.data[0,0,0,0]} vs {expected_avg}"
        
        loss = y.sum()
        loss.backward()
        
        if x.grad.shape != x.shape:
            return False, "gradient shape mismatch"
        
        # For sum loss with avg pooling, gradient = 1 / pool_size = 0.25
        expected_grad = 0.25
        if not np.allclose(x.grad, expected_grad):
            return False, f"gradient {x.grad[0,0,0,0]}, expected {expected_grad}"
        
        return True, "AvgPool2d module works"
    except Exception as e:
        return False, str(e)


def test_global_avgpool_flatten() -> Tuple[bool, str]:
    """Test GlobalAvgPool2d with flatten=True."""
    try:
        np.random.seed(42)
        gap = GlobalAvgPool2d(flatten=True)
        x = Tensor(np.random.randn(2, 16, 4, 4))
        y = gap(x)
        
        if y is None:
            return False, "forward returned None"
        
        if y.shape != (2, 16):
            return False, f"shape {y.shape}, expected (2, 16)"
        
        # Verify global average values
        expected = np.mean(x.data, axis=(2, 3))
        if not np.allclose(y.data, expected):
            return False, f"values mismatch, diff: {np.max(np.abs(y.data - expected))}"
        
        return True, "GlobalAvgPool flatten works"
    except Exception as e:
        return False, str(e)


def test_global_avgpool_no_flatten() -> Tuple[bool, str]:
    """Test GlobalAvgPool2d with flatten=False."""
    try:
        np.random.seed(42)
        gap = GlobalAvgPool2d(flatten=False)
        x = Tensor(np.random.randn(2, 16, 4, 4))
        y = gap(x)
        
        if y is None:
            return False, "forward returned None"
        
        if y.shape != (2, 16, 1, 1):
            return False, f"shape {y.shape}, expected (2, 16, 1, 1)"
        
        # Verify global average values
        expected = np.mean(x.data, axis=(2, 3), keepdims=True)
        if not np.allclose(y.data, expected):
            return False, f"values mismatch, diff: {np.max(np.abs(y.data - expected))}"
        
        return True, "GlobalAvgPool no flatten works"
    except Exception as e:
        return False, str(e)


def test_global_avgpool_values() -> Tuple[bool, str]:
    """Test GlobalAvgPool2d computes correct values."""
    try:
        np.random.seed(42)
        gap = GlobalAvgPool2d(flatten=True)
        x = Tensor(np.random.randn(2, 3, 4, 4))
        y = gap(x)
        
        if y is None:
            return False, "forward returned None"
        
        expected = np.mean(x.data, axis=(2, 3))
        
        if not np.allclose(y.data, expected):
            return False, "values mismatch"
        
        return True, "GlobalAvgPool values correct"
    except Exception as e:
        return False, str(e)


def test_global_avgpool_backward() -> Tuple[bool, str]:
    """Test GlobalAvgPool2d backward pass."""
    try:
        np.random.seed(42)
        gap = GlobalAvgPool2d(flatten=True)
        x = Tensor(np.random.randn(2, 3, 4, 4))
        y = gap(x)
        
        if y is None:
            return False, "forward returned None"
        
        loss = y.sum()
        loss.backward()
        
        expected_grad = 1.0 / (4 * 4)
        
        if not np.allclose(x.grad, expected_grad):
            return False, f"gradient mismatch"
        
        return True, "GlobalAvgPool backward works"
    except Exception as e:
        return False, str(e)


def test_adaptive_avgpool_shape() -> Tuple[bool, str]:
    """Test AdaptiveAvgPool2d output shape."""
    try:
        np.random.seed(42)
        pool = AdaptiveAvgPool2d(output_size=(2, 2))
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = pool(x)
        
        if y is None:
            return False, "forward returned None"
        
        if y.shape != (2, 3, 2, 2):
            return False, f"shape {y.shape}, expected (2, 3, 2, 2)"
        
        # Verify adaptive pooling produces correct averages
        # With 8x8 -> 2x2, each output covers 4x4 region
        expected_00 = np.mean(x.data[0, 0, 0:4, 0:4])
        if not np.isclose(y.data[0, 0, 0, 0], expected_00, rtol=1e-5):
            return False, f"value at (0,0): {y.data[0,0,0,0]} vs expected {expected_00}"
        
        return True, "AdaptiveAvgPool shape correct"
    except Exception as e:
        return False, str(e)


def test_adaptive_avgpool_backward() -> Tuple[bool, str]:
    """Test AdaptiveAvgPool2d backward pass."""
    try:
        np.random.seed(42)
        pool = AdaptiveAvgPool2d(output_size=(2, 2))
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = pool(x)
        
        if y is None:
            return False, "forward returned None"
        
        loss = y.sum()
        loss.backward()
        
        if x.grad.shape != x.shape:
            return False, "gradient shape mismatch"
        
        if np.all(x.grad == 0):
            return False, "gradient is all zeros"
        
        # With 8x8 -> 2x2, each 4x4 region gets 1/16 gradient
        expected_grad = 1.0 / 16.0
        if not np.allclose(x.grad, expected_grad, rtol=1e-5):
            return False, f"gradient {x.grad[0,0,0,0]}, expected {expected_grad}"
        
        return True, "AdaptiveAvgPool backward works"
    except Exception as e:
        return False, str(e)


def test_adaptive_avgpool_different_sizes() -> Tuple[bool, str]:
    """Test AdaptiveAvgPool2d with various input sizes."""
    try:
        np.random.seed(42)
        pool = AdaptiveAvgPool2d(output_size=(1, 1))
        
        for h, w in [(4, 4), (7, 7), (13, 13)]:
            x = Tensor(np.random.randn(1, 3, h, w))
            y = pool(x)
            
            if y is None:
                return False, f"returned None for {h}x{w}"
            
            if y.shape != (1, 3, 1, 1):
                return False, f"shape {y.shape} for {h}x{w}"
            
            # Verify global average
            expected = np.mean(x.data, axis=(2, 3), keepdims=True)
            if not np.allclose(y.data, expected, rtol=1e-5):
                return False, f"value mismatch for {h}x{w}"
        
        return True, "Works with different input sizes"
    except Exception as e:
        return False, str(e)


def test_maxpool_numerical_gradient() -> Tuple[bool, str]:
    """Numerical gradient check for max pooling."""
    try:
        np.random.seed(42)
        pool = MaxPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 2, 4, 4))
        
        y = pool(x)
        if y is None:
            return False, "forward returned None"
        
        y.sum().backward()
        analytic_grad = x.grad.copy()
        
        eps = 1e-5
        numeric_grad = np.zeros_like(x.data)
        
        for idx in np.ndindex(x.data.shape):
            x_data = x.data.copy()
            x_data[idx] += eps
            x_plus = Tensor(x_data)
            y_plus = pool(x_plus)
            loss_plus = y_plus.sum().data if y_plus is not None else 0
            
            x_data = x.data.copy()
            x_data[idx] -= eps
            x_minus = Tensor(x_data)
            y_minus = pool(x_minus)
            loss_minus = y_minus.sum().data if y_minus is not None else 0
            
            numeric_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        if not np.allclose(analytic_grad, numeric_grad, rtol=1e-3, atol=1e-5):
            return False, "Gradient mismatch"
        
        return True, "Numerical gradient check passed"
    except Exception as e:
        return False, str(e)


def test_avgpool_numerical_gradient() -> Tuple[bool, str]:
    """Numerical gradient check for average pooling."""
    try:
        np.random.seed(42)
        pool = AvgPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 2, 4, 4))
        
        y = pool(x)
        if y is None:
            return False, "forward returned None"
        
        y.sum().backward()
        analytic_grad = x.grad.copy()
        
        eps = 1e-5
        numeric_grad = np.zeros_like(x.data)
        
        for idx in np.ndindex(x.data.shape):
            x_data = x.data.copy()
            x_data[idx] += eps
            x_plus = Tensor(x_data)
            y_plus = pool(x_plus)
            loss_plus = y_plus.sum().data if y_plus is not None else 0
            
            x_data = x.data.copy()
            x_data[idx] -= eps
            x_minus = Tensor(x_data)
            y_minus = pool(x_minus)
            loss_minus = y_minus.sum().data if y_minus is not None else 0
            
            numeric_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        if not np.allclose(analytic_grad, numeric_grad, rtol=1e-3, atol=1e-5):
            return False, "Gradient mismatch"
        
        return True, "Numerical gradient check passed"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test pooling against PyTorch implementation."""
    try:
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        x_np = np.random.randn(2, 3, 8, 8)
        
        our_maxpool = MaxPool2d(kernel_size=2, stride=2)
        our_x1 = Tensor(x_np.copy())
        our_y1 = our_maxpool(our_x1)
        
        torch_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        torch_x1 = torch.tensor(x_np, requires_grad=True)
        torch_y1 = torch_maxpool(torch_x1)
        
        if our_y1 is None:
            return False, "MaxPool returned None"
        
        if not np.allclose(our_y1.data, torch_y1.detach().numpy(), rtol=1e-5):
            return False, "MaxPool forward mismatch"
        
        our_avgpool = AvgPool2d(kernel_size=2, stride=2)
        our_x2 = Tensor(x_np.copy())
        our_y2 = our_avgpool(our_x2)
        
        torch_avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        torch_x2 = torch.tensor(x_np, requires_grad=True)
        torch_y2 = torch_avgpool(torch_x2)
        
        if our_y2 is None:
            return False, "AvgPool returned None"
        
        if not np.allclose(our_y2.data, torch_y2.detach().numpy(), rtol=1e-5):
            return False, "AvgPool forward mismatch"
        
        our_y1.sum().backward()
        torch_y1.sum().backward()
        
        if not np.allclose(our_x1.grad, torch_x1.grad.numpy(), rtol=1e-5):
            return False, "MaxPool backward mismatch"
        
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("maxpool_forward_basic", test_maxpool_forward_basic),
        ("maxpool_forward_shape", test_maxpool_forward_shape),
        ("maxpool_stride", test_maxpool_stride),
        ("maxpool_backward_gradient", test_maxpool_backward_gradient),
        ("maxpool_backward_shape", test_maxpool_backward_shape),
        ("maxpool2d_module", test_maxpool2d_module),
        ("avgpool_forward_basic", test_avgpool_forward_basic),
        ("avgpool_forward_shape", test_avgpool_forward_shape),
        ("avgpool_backward_uniform", test_avgpool_backward_uniform),
        ("avgpool_backward_shape", test_avgpool_backward_shape),
        ("avgpool2d_module", test_avgpool2d_module),
        ("global_avgpool_flatten", test_global_avgpool_flatten),
        ("global_avgpool_no_flatten", test_global_avgpool_no_flatten),
        ("global_avgpool_values", test_global_avgpool_values),
        ("global_avgpool_backward", test_global_avgpool_backward),
        ("adaptive_avgpool_shape", test_adaptive_avgpool_shape),
        ("adaptive_avgpool_backward", test_adaptive_avgpool_backward),
        ("adaptive_avgpool_different_sizes", test_adaptive_avgpool_different_sizes),
        ("maxpool_numerical_gradient", test_maxpool_numerical_gradient),
        ("avgpool_numerical_gradient", test_avgpool_numerical_gradient),
        ("against_pytorch", test_against_pytorch),
    ]
    
    print(f"\n{'='*60}")
    print("Day 32: Pooling Operations - Tests")
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
