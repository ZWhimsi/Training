"""Test Suite for Day 32: Pooling Operations"""

import numpy as np
import pytest

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

def test_maxpool_forward_basic():
    """Test max pooling forward basic case."""
    x = np.array([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]]).astype(np.float64)
        
    result = maxpool2d_forward(x, kernel_size=2, stride=2)
        
    assert not (result is None), "maxpool2d_forward returned None"
        
    out, mask = result
    expected = np.array([[[[6, 8], [14, 16]]]])
        
    assert np.allclose(out, expected), f"Values mismatch: {out}"
        
def test_maxpool_forward_shape():
    """Test max pooling output shape."""
    np.random.seed(42)
    x = np.random.randn(4, 8, 16, 16)
    result = maxpool2d_forward(x, kernel_size=2, stride=2)
        
    assert not (result is None), "returned None"
        
    out, mask = result
        
    assert not (out.shape != (4, 8, 8, 8)), f"shape {out.shape}, expected (4, 8, 8, 8)"
        
    # Verify max pooling property: output should be >= all values in window
    # Check a specific window
    window = x[0, 0, 0:2, 0:2]
    expected_max = np.max(window)
    assert np.isclose(out[0, 0, 0, 0], expected_max), f"output {out[0,0,0,0]} != max {expected_max}"
        
def test_maxpool_stride():
    """Test max pooling with different stride."""
    np.random.seed(42)
    x = np.arange(36).reshape(1, 1, 6, 6).astype(np.float64)
    result = maxpool2d_forward(x, kernel_size=2, stride=1)
        
    assert not (result is None), "returned None"
        
    out, mask = result
        
    assert not (out.shape != (1, 1, 5, 5)), f"shape {out.shape}, expected (1, 1, 5, 5)"
        
    # Verify values: with stride=1, windows overlap
    # First window [0,1,6,7] -> max=7
    assert np.isclose(out[0, 0, 0, 0], 7.0), f"position (0,0) = {out[0,0,0,0]}, expected 7"
    # Window at (0,1) is [1,2,7,8] -> max=8
    assert np.isclose(out[0, 0, 0, 1], 8.0), f"position (0,1) = {out[0,0,0,1]}, expected 8"
        
def test_maxpool_backward_gradient():
    """Test max pooling backward passes gradient to max elements."""
    x = np.array([[[[1, 2], [3, 4]]]]).astype(np.float64)
    result = maxpool2d_forward(x, kernel_size=2, stride=2)
        
    assert not (result is None), "forward returned None"
        
    out, mask = result
    dy = np.array([[[[1.0]]]])
        
    dx = maxpool2d_backward(dy, mask, x.shape, kernel_size=2, stride=2)
        
    assert not (dx is None), "backward returned None"
        
    expected = np.array([[[[0, 0], [0, 1]]]])
        
    assert np.allclose(dx, expected), f"gradient {dx}, expected {expected}"
        
def test_maxpool_backward_shape():
    """Test max pooling backward output shape."""
    np.random.seed(42)
    x = np.random.randn(2, 4, 8, 8)
    result = maxpool2d_forward(x, kernel_size=2, stride=2)
        
    assert not (result is None), "forward returned None"
        
    out, mask = result
    dy = np.random.randn(*out.shape)
        
    dx = maxpool2d_backward(dy, mask, x.shape, kernel_size=2, stride=2)
        
    assert not (dx is None), "backward returned None"
        
    assert not (dx.shape != x.shape), f"shape {dx.shape}, expected {x.shape}"
        
    # Verify sparsity: only max positions should have non-zero gradients
    non_zero_count = np.sum(dx != 0)
    expected_count = out.size  # One gradient per output position
    assert not (non_zero_count != expected_count), f"non-zero count {non_zero_count}, expected {expected_count}"
        
def test_maxpool2d_module():
    """Test MaxPool2d module end-to-end."""
    np.random.seed(42)
    pool = MaxPool2d(kernel_size=2, stride=2)
        
    x = Tensor(np.random.randn(2, 3, 8, 8))
    y = pool(x)
        
    assert not (y is None), "forward returned None"
        
    assert not (y.shape != (2, 3, 4, 4)), f"shape {y.shape}"
        
    # Verify max pooling values
    window = x.data[0, 0, 0:2, 0:2]
    expected_max = np.max(window)
    assert np.isclose(y.data[0, 0, 0, 0], expected_max), f"max value mismatch: {y.data[0,0,0,0]} vs {expected_max}"
        
    loss = y.sum()
    loss.backward()
        
    assert not (x.grad.shape != x.shape), "gradient shape mismatch"
        
    # Verify gradient sparsity for max pooling
    non_zero_grads = np.sum(x.grad != 0)
    assert not (non_zero_grads != y.data.size), f"gradient sparsity: {non_zero_grads} non-zero, expected {y.data.size}"
        
def test_avgpool_forward_basic():
    """Test average pooling forward basic case."""
    x = np.array([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]]).astype(np.float64)
        
    out = avgpool2d_forward(x, kernel_size=2, stride=2)
        
    assert not (out is None), "avgpool2d_forward returned None"
        
    expected = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])
        
    assert np.allclose(out, expected), f"Values mismatch: {out}"
        
def test_avgpool_forward_shape():
    """Test average pooling output shape."""
    np.random.seed(42)
    x = np.random.randn(4, 8, 16, 16)
    out = avgpool2d_forward(x, kernel_size=2, stride=2)
        
    assert not (out is None), "returned None"
        
    assert not (out.shape != (4, 8, 8, 8)), f"shape {out.shape}, expected (4, 8, 8, 8)"
        
    # Verify average pooling value
    window = x[0, 0, 0:2, 0:2]
    expected_avg = np.mean(window)
    assert np.isclose(out[0, 0, 0, 0], expected_avg), f"output {out[0,0,0,0]} != mean {expected_avg}"
        
def test_avgpool_backward_uniform():
    """Test average pooling distributes gradient uniformly."""
    x_shape = (1, 1, 4, 4)
    dy = np.array([[[[4.0, 4.0], [4.0, 4.0]]]])
        
    dx = avgpool2d_backward(dy, x_shape, kernel_size=2, stride=2)
        
    assert not (dx is None), "backward returned None"
        
    assert np.allclose(dx, 1.0), f"gradient not uniform: {dx}"
        
def test_avgpool_backward_shape():
    """Test average pooling backward output shape."""
    x_shape = (2, 4, 8, 8)
    dy = np.ones((2, 4, 4, 4))  # Use ones for predictable gradient
        
    dx = avgpool2d_backward(dy, x_shape, kernel_size=2, stride=2)
        
    assert not (dx is None), "backward returned None"
        
    assert not (dx.shape != x_shape), f"shape {dx.shape}, expected {x_shape}"
        
    # With ones gradient and 2x2 pool, each input should get 1/4 = 0.25
    expected_grad = 0.25
    assert np.allclose(dx, expected_grad), f"gradient {dx[0,0,0,0]}, expected {expected_grad}"
        
def test_avgpool2d_module():
    """Test AvgPool2d module end-to-end."""
    np.random.seed(42)
    pool = AvgPool2d(kernel_size=2, stride=2)
        
    x = Tensor(np.random.randn(2, 3, 8, 8))
    y = pool(x)
        
    assert not (y is None), "forward returned None"
        
    assert not (y.shape != (2, 3, 4, 4)), f"shape {y.shape}"
        
    # Verify average pooling value
    window = x.data[0, 0, 0:2, 0:2]
    expected_avg = np.mean(window)
    assert np.isclose(y.data[0, 0, 0, 0], expected_avg), f"avg mismatch: {y.data[0,0,0,0]} vs {expected_avg}"
        
    loss = y.sum()
    loss.backward()
        
    assert not (x.grad.shape != x.shape), "gradient shape mismatch"
        
    # For sum loss with avg pooling, gradient = 1 / pool_size = 0.25
    expected_grad = 0.25
    assert np.allclose(x.grad, expected_grad), f"gradient {x.grad[0,0,0,0]}, expected {expected_grad}"
        
def test_global_avgpool_flatten():
    """Test GlobalAvgPool2d with flatten=True."""
    np.random.seed(42)
    gap = GlobalAvgPool2d(flatten=True)
    x = Tensor(np.random.randn(2, 16, 4, 4))
    y = gap(x)
        
    assert not (y is None), "forward returned None"
        
    assert not (y.shape != (2, 16)), f"shape {y.shape}, expected (2, 16)"
        
    # Verify global average values
    expected = np.mean(x.data, axis=(2, 3))
    assert np.allclose(y.data, expected), f"values mismatch, diff: {np.max(np.abs(y.data - expected))}"
        
def test_global_avgpool_no_flatten():
    """Test GlobalAvgPool2d with flatten=False."""
    np.random.seed(42)
    gap = GlobalAvgPool2d(flatten=False)
    x = Tensor(np.random.randn(2, 16, 4, 4))
    y = gap(x)
        
    assert not (y is None), "forward returned None"
        
    assert not (y.shape != (2, 16, 1, 1)), f"shape {y.shape}, expected (2, 16, 1, 1)"
        
    # Verify global average values
    expected = np.mean(x.data, axis=(2, 3), keepdims=True)
    assert np.allclose(y.data, expected), f"values mismatch, diff: {np.max(np.abs(y.data - expected))}"
        
def test_global_avgpool_values():
    """Test GlobalAvgPool2d computes correct values."""
    np.random.seed(42)
    gap = GlobalAvgPool2d(flatten=True)
    x = Tensor(np.random.randn(2, 3, 4, 4))
    y = gap(x)
        
    assert not (y is None), "forward returned None"
        
    expected = np.mean(x.data, axis=(2, 3))
        
    assert np.allclose(y.data, expected), "values mismatch"
        
def test_global_avgpool_backward():
    """Test GlobalAvgPool2d backward pass."""
    np.random.seed(42)
    gap = GlobalAvgPool2d(flatten=True)
    x = Tensor(np.random.randn(2, 3, 4, 4))
    y = gap(x)
        
    assert not (y is None), "forward returned None"
        
    loss = y.sum()
    loss.backward()
        
    expected_grad = 1.0 / (4 * 4)
        
    assert np.allclose(x.grad, expected_grad), f"gradient mismatch"
        
def test_adaptive_avgpool_shape():
    """Test AdaptiveAvgPool2d output shape."""
    np.random.seed(42)
    pool = AdaptiveAvgPool2d(output_size=(2, 2))
    x = Tensor(np.random.randn(2, 3, 8, 8))
    y = pool(x)
        
    assert not (y is None), "forward returned None"
        
    assert not (y.shape != (2, 3, 2, 2)), f"shape {y.shape}, expected (2, 3, 2, 2)"
        
    # Verify adaptive pooling produces correct averages
    # With 8x8 -> 2x2, each output covers 4x4 region
    expected_00 = np.mean(x.data[0, 0, 0:4, 0:4])
    assert np.isclose(y.data[0, 0, 0, 0], expected_00, rtol=1e-5), f"value at (0,0): {y.data[0,0,0,0]} vs expected {expected_00}"
        
def test_adaptive_avgpool_backward():
    """Test AdaptiveAvgPool2d backward pass."""
    np.random.seed(42)
    pool = AdaptiveAvgPool2d(output_size=(2, 2))
    x = Tensor(np.random.randn(2, 3, 8, 8))
    y = pool(x)
        
    assert not (y is None), "forward returned None"
        
    loss = y.sum()
    loss.backward()
        
    assert not (x.grad.shape != x.shape), "gradient shape mismatch"
        
    assert not (np.all(x.grad == 0)), "gradient is all zeros"
        
    # With 8x8 -> 2x2, each 4x4 region gets 1/16 gradient
    expected_grad = 1.0 / 16.0
    assert np.allclose(x.grad, expected_grad, rtol=1e-5), f"gradient {x.grad[0,0,0,0]}, expected {expected_grad}"
        
def test_adaptive_avgpool_different_sizes():
    """Test AdaptiveAvgPool2d with various input sizes."""
    np.random.seed(42)
    pool = AdaptiveAvgPool2d(output_size=(1, 1))
        
    for h, w in [(4, 4), (7, 7), (13, 13)]:
        x = Tensor(np.random.randn(1, 3, h, w))
        y = pool(x)
            
        assert not (y is None), f"returned None for {h}x{w}"
            
        assert not (y.shape != (1, 3, 1, 1)), f"shape {y.shape} for {h}x{w}"
            
        # Verify global average
        expected = np.mean(x.data, axis=(2, 3), keepdims=True)
        assert np.allclose(y.data, expected, rtol=1e-5), f"value mismatch for {h}x{w}"
        
def test_maxpool_numerical_gradient():
    """Numerical gradient check for max pooling."""
    np.random.seed(42)
    pool = MaxPool2d(kernel_size=2, stride=2)
    x = Tensor(np.random.randn(1, 2, 4, 4))
        
    y = pool(x)
    assert not (y is None), "forward returned None"
        
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
        
    assert np.allclose(analytic_grad, numeric_grad, rtol=1e-3, atol=1e-5), "Gradient mismatch"
        
def test_avgpool_numerical_gradient():
    """Numerical gradient check for average pooling."""
    np.random.seed(42)
    pool = AvgPool2d(kernel_size=2, stride=2)
    x = Tensor(np.random.randn(1, 2, 4, 4))
        
    y = pool(x)
    assert not (y is None), "forward returned None"
        
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
        
    assert np.allclose(analytic_grad, numeric_grad, rtol=1e-3, atol=1e-5), "Gradient mismatch"
        
def test_against_pytorch():
    """Test pooling against PyTorch implementation."""
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
        
    assert not (our_y1 is None), "MaxPool returned None"
        
    assert np.allclose(our_y1.data, torch_y1.detach().numpy(), rtol=1e-5), "MaxPool forward mismatch"
        
    our_avgpool = AvgPool2d(kernel_size=2, stride=2)
    our_x2 = Tensor(x_np.copy())
    our_y2 = our_avgpool(our_x2)
        
    torch_avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    torch_x2 = torch.tensor(x_np, requires_grad=True)
    torch_y2 = torch_avgpool(torch_x2)
        
    assert not (our_y2 is None), "AvgPool returned None"
        
    assert np.allclose(our_y2.data, torch_y2.detach().numpy(), rtol=1e-5), "AvgPool forward mismatch"
        
    our_y1.sum().backward()
    torch_y1.sum().backward()
        
    assert np.allclose(our_x1.grad, torch_x1.grad.numpy(), rtol=1e-5), "MaxPool backward mismatch"
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])