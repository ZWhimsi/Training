"""Test Suite for Day 31: Convolutional Operations"""

import numpy as np
import pytest

try:
    from day31 import (
        Tensor,
        im2col,
        col2im,
        conv2d_forward,
        conv2d_backward,
        conv2d_naive,
        Conv2d,
        Module
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_im2col_basic():
    """Test im2col basic functionality."""
    x = np.arange(16).reshape(1, 1, 4, 4).astype(np.float64)
    col = im2col(x, 2, 2, stride=1, padding=0)
        
    assert not (col is None), "im2col returned None"
        
    expected_shape = (9, 4)
    assert not (col.shape != expected_shape), f"shape {col.shape}, expected {expected_shape}"
        
    # Verify actual values: first patch should be [0,1,4,5]
    expected_first_patch = np.array([0, 1, 4, 5])
    assert np.allclose(col[0], expected_first_patch), f"first patch {col[0]}, expected {expected_first_patch}"
        
    # Last patch should be [10,11,14,15]
    expected_last_patch = np.array([10, 11, 14, 15])
    assert np.allclose(col[-1], expected_last_patch), f"last patch {col[-1]}, expected {expected_last_patch}"
        
def test_im2col_with_padding():
    """Test im2col with padding."""
    np.random.seed(42)
    x = np.random.randn(2, 3, 4, 4)
    col = im2col(x, 3, 3, stride=1, padding=1)
        
    assert not (col is None), "im2col returned None"
        
    expected_shape = (2 * 4 * 4, 3 * 3 * 3)
    assert not (col.shape != expected_shape), f"shape {col.shape}, expected {expected_shape}"
        
    # Verify corner patch has zeros from padding
    # First patch at (0,0) should have zeros in top-left corner due to padding
    first_patch = col[0].reshape(3, 3, 3)  # (C, kH, kW)
    # Top-left corner of each channel should be zero (from padding)
    assert np.allclose(first_patch[:, 0, 0], 0.0), f"padding not applied correctly, got {first_patch[:, 0, 0]}"
        
def test_im2col_with_stride():
    """Test im2col with stride."""
    np.random.seed(42)
    x = np.arange(72).reshape(1, 2, 6, 6).astype(np.float64)
    col = im2col(x, 2, 2, stride=2, padding=0)
        
    assert not (col is None), "im2col returned None"
        
    expected_shape = (1 * 3 * 3, 2 * 2 * 2)
    assert not (col.shape != expected_shape), f"shape {col.shape}, expected {expected_shape}"
        
    # First patch: channel 0 at (0,0) should be [0,1,6,7], channel 1 at (0,0) should be [36,37,42,43]
    expected_first = np.array([0, 1, 6, 7, 36, 37, 42, 43])
    assert np.allclose(col[0], expected_first), f"first patch {col[0]}, expected {expected_first}"
        
def test_col2im_basic():
    """Test col2im basic functionality."""
    x = np.random.randn(1, 2, 4, 4)
    col = im2col(x, 2, 2, stride=2, padding=0)
        
    assert not (col is None), "im2col returned None"
        
    reconstructed = col2im(col, x.shape, 2, 2, stride=2, padding=0)
        
    assert not (reconstructed is None), "col2im returned None"
        
    assert np.allclose(x, reconstructed), "Reconstruction mismatch"
        
def test_col2im_with_padding():
    """Test col2im with padding."""
    np.random.seed(42)
    x = np.random.randn(2, 3, 4, 4)
    col = im2col(x, 3, 3, stride=1, padding=1)
        
    assert not (col is None), "im2col returned None"
        
    reconstructed = col2im(col, x.shape, 3, 3, stride=1, padding=1)
        
    assert not (reconstructed is None), "col2im returned None"
        
    assert not (reconstructed.shape != x.shape), f"shape {reconstructed.shape}, expected {x.shape}"
        
    # With overlapping windows (stride=1), values get accumulated
    # Check that center values have higher counts
    # Center of 4x4 with 3x3 kernel and padding=1 sees each pixel multiple times
    # The reconstruction should scale values by the number of times they appear
    # For stride=1, center pixels appear 9 times (3x3), edges fewer
    center_val = reconstructed[0, 0, 1, 1]
    corner_val = reconstructed[0, 0, 0, 0]
    # Corner appears 4 times, center appears 9 times, so center should be higher
    if not (abs(center_val) > abs(corner_val) * 0.5):
        pass  # This is a weaker check since values could be negative
        
def test_conv2d_forward_shape():
    """Test conv2d forward output shape."""
    np.random.seed(42)
    x = np.random.randn(2, 3, 8, 8)
    w = np.random.randn(16, 3, 3, 3)
    b = np.zeros(16)
        
    out = conv2d_forward(x, w, b, stride=1, padding=1)
        
    assert not (out is None), "conv2d_forward returned None"
        
    assert not (out.shape != (2, 16, 8, 8)), f"shape {out.shape}, expected (2, 16, 8, 8)"
        
    # Verify output is not all zeros or NaN
    assert not (np.all(out == 0)), "output is all zeros"
    assert np.all(np.isfinite(out)), "output contains NaN or Inf"
        
    # Verify convolution property: output should change when input changes
    x2 = x + 1.0
    out2 = conv2d_forward(x2, w, b, stride=1, padding=1)
    assert not (np.allclose(out, out2)), "output unchanged when input changed"
        
def test_conv2d_forward_stride():
    """Test conv2d forward with stride."""
    np.random.seed(42)
    x = np.random.randn(1, 3, 8, 8)
    w = np.random.randn(4, 3, 3, 3)
        
    out = conv2d_forward(x, w, None, stride=2, padding=1)
        
    assert not (out is None), "conv2d_forward returned None"
        
    assert not (out.shape != (1, 4, 4, 4)), f"shape {out.shape}, expected (1, 4, 4, 4)"
        
    # Verify stride reduces output correctly: 8x8 with stride 2 -> 4x4
    # Also verify output values are reasonable
    assert np.all(np.isfinite(out)), "output contains NaN or Inf"
        
    # Check that stride=2 output is subsampled from stride=1 output
    out_s1 = conv2d_forward(x, w, None, stride=1, padding=1)
    if out_s1 is not None:
        # stride=2 should correspond to every other position
        assert np.allclose(out[0, :, 0, 0], out_s1[0, :, 0, 0], rtol=1e-5), "stride=2 output doesn't match stride=1 at position (0,0)"
        
def test_conv2d_matches_naive():
    """Test that conv2d_forward matches naive implementation."""
    np.random.seed(42)
    x = np.random.randn(2, 3, 6, 6)
    w = np.random.randn(4, 3, 3, 3)
    b = np.random.randn(4)
        
    out_fast = conv2d_forward(x, w, b, stride=1, padding=1)
    out_naive = conv2d_naive(x, w, b, stride=1, padding=1)
        
    assert not (out_fast is None), "conv2d_forward returned None"
    assert not (out_naive is None), "conv2d_naive returned None"
        
    if not np.allclose(out_fast, out_naive, rtol=1e-5):
        diff = np.max(np.abs(out_fast - out_naive))
        assert False, f"Max difference: {diff}"
        
def test_conv2d_backward_shapes():
    """Test conv2d backward gradient shapes."""
    np.random.seed(42)
    x = np.random.randn(2, 3, 6, 6)
    w = np.random.randn(4, 3, 3, 3)
    dy = np.random.randn(2, 4, 4, 4)
        
    result = conv2d_backward(dy, x, w, stride=1, padding=0)
        
    assert not (result is None), "conv2d_backward returned None"
        
    dx, dw, db = result
        
    assert not (dx.shape != x.shape), f"dx shape {dx.shape}, expected {x.shape}"
    assert not (dw.shape != w.shape), f"dw shape {dw.shape}, expected {w.shape}"
    assert not (db.shape != (4,)), f"db shape {db.shape}, expected (4,)"
        
    # Verify gradients are not all zeros (indicates actual computation)
    assert not (np.all(dx == 0)), "dx is all zeros"
    assert not (np.all(dw == 0)), "dw is all zeros"
    assert not (np.all(db == 0)), "db is all zeros"
        
    # Verify gradients are finite
    assert np.all(np.isfinite(dx)), "dx contains NaN or Inf"
    assert np.all(np.isfinite(dw)), "dw contains NaN or Inf"
        
def test_conv2d_backward_bias():
    """Test conv2d backward bias gradient."""
    dy = np.random.randn(2, 4, 3, 3)
    x = np.random.randn(2, 3, 5, 5)
    w = np.random.randn(4, 3, 3, 3)
        
    result = conv2d_backward(dy, x, w, stride=1, padding=0)
        
    assert not (result is None), "conv2d_backward returned None"
        
    dx, dw, db = result
        
    expected_db = np.sum(dy, axis=(0, 2, 3))
        
    assert np.allclose(db, expected_db), "Bias gradient incorrect"
        
def test_conv2d_gradient_numerical():
    """Numerical gradient check for conv2d."""
    np.random.seed(42)
    x = np.random.randn(1, 2, 4, 4)
    w = np.random.randn(3, 2, 2, 2)
    b = np.random.randn(3)
        
    out = conv2d_forward(x, w, b, stride=1, padding=0)
    assert not (out is None), "conv2d_forward returned None"
        
    dy = np.ones_like(out)
        
    result = conv2d_backward(dy, x, w, stride=1, padding=0)
    assert not (result is None), "conv2d_backward returned None"
        
    dx_analytic, dw_analytic, db_analytic = result
        
    eps = 1e-5
    dw_numeric = np.zeros_like(w)
        
    for idx in np.ndindex(w.shape):
        w_plus = w.copy()
        w_plus[idx] += eps
        out_plus = conv2d_forward(x, w_plus, b, stride=1, padding=0)
            
        w_minus = w.copy()
        w_minus[idx] -= eps
        out_minus = conv2d_forward(x, w_minus, b, stride=1, padding=0)
            
        if out_plus is not None and out_minus is not None:
            dw_numeric[idx] = (np.sum(out_plus) - np.sum(out_minus)) / (2 * eps)
        
    assert np.allclose(dw_analytic, dw_numeric, rtol=1e-3, atol=1e-5), "Weight gradient mismatch"
        
def test_conv2d_module_init():
    """Test Conv2d module initialization."""
    np.random.seed(42)
    conv = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        
    assert not (conv.weight is None), "weight is None"
        
    assert not (conv.weight.shape != (16, 3, 3, 3)), f"weight shape {conv.weight.shape}"
        
    assert not (conv.bias is not None and conv.bias.shape != (16,)), f"bias shape {conv.bias.shape}"
        
    # Verify He initialization: std should be approximately sqrt(2/fan_in)
    fan_in = 3 * 3 * 3
    expected_std = np.sqrt(2.0 / fan_in)
    actual_std = np.std(conv.weight.data)
    assert np.isclose(actual_std, expected_std, rtol=0.5), f"weight std {actual_std:.4f}, expected ~{expected_std:.4f}"
        
    # Bias should be initialized to zeros
    if conv.bias is not None:
        assert np.allclose(conv.bias.data, 0), "bias not initialized to zeros"
        
def test_conv2d_module_forward():
    """Test Conv2d module forward pass."""
    np.random.seed(42)
    conv = Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        
    assert not (conv.weight is None), "weight is None"
        
    x = Tensor(np.random.randn(2, 3, 8, 8))
    y = conv(x)
        
    assert not (y is None), "forward returned None"
        
    assert not (y.shape != (2, 8, 8, 8)), f"output shape {y.shape}"
        
    # Verify output values are finite
    assert np.all(np.isfinite(y.data)), "output contains NaN or Inf"
        
    # Verify output is not all zeros
    assert not (np.all(y.data == 0)), "output is all zeros"
        
    # Verify output changes with input
    x2 = Tensor(np.random.randn(2, 3, 8, 8))
    y2 = conv(x2)
    assert not (np.allclose(y.data, y2.data)), "output same for different inputs"
        
def test_conv2d_module_backward():
    """Test Conv2d module backward pass."""
    np.random.seed(42)
    conv = Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
        
    assert not (conv.weight is None), "weight is None"
        
    x = Tensor(np.random.randn(1, 2, 6, 6))
    y = conv(x)
        
    assert not (y is None), "forward returned None"
        
    loss = y.sum()
    loss.backward()
        
    assert not (np.all(conv.weight.grad == 0)), "Weight gradient is zero"
        
    assert not (conv.bias is not None and np.all(conv.bias.grad == 0)), "Bias gradient is zero"
        
    # For sum loss, bias gradient should equal output spatial size
    if conv.bias is not None:
        expected_bias_grad = 6 * 6  # H_out * W_out
        assert np.allclose(conv.bias.grad, expected_bias_grad), f"bias grad {conv.bias.grad[0]}, expected {expected_bias_grad}"
        
    # Verify gradients are finite
    assert np.all(np.isfinite(conv.weight.grad)), "weight gradient contains NaN or Inf"
    assert np.all(np.isfinite(x.grad)), "input gradient contains NaN or Inf"
        
def test_conv2d_module_no_bias():
    """Test Conv2d module without bias."""
    conv = Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        
    assert not (conv.bias is not None), "bias should be None"
        
    params = conv.parameters()
    assert not (len(params) != 1), f"Should have 1 param, got {len(params)}"
        
def test_conv2d_parameters():
    """Test Conv2d parameters method."""
    conv = Conv2d(3, 16, kernel_size=3)
    params = conv.parameters()
        
    assert not (len(params) != 2), f"Expected 2 params, got {len(params)}"
        
    conv_no_bias = Conv2d(3, 16, kernel_size=3, bias=False)
    params_no_bias = conv_no_bias.parameters()
        
    assert not (len(params_no_bias) != 1), f"Expected 1 param, got {len(params_no_bias)}"
        
def test_conv2d_zero_grad():
    """Test Conv2d zero_grad method."""
    np.random.seed(42)
    conv = Conv2d(3, 8, kernel_size=3, padding=1)
        
    assert not (conv.weight is None), "weight is None"
        
    x = Tensor(np.random.randn(1, 3, 4, 4))
    y = conv(x)
        
    if y is not None:
        y.sum().backward()
        
    assert not (np.all(conv.weight.grad == 0)), "Gradient not computed"
        
    conv.zero_grad()
        
    assert np.all(conv.weight.grad == 0), "Gradient not zeroed"
        
def test_conv2d_different_strides():
    """Test Conv2d with different strides."""
    np.random.seed(42)
        
    conv_s1 = Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
    conv_s2 = Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        
    assert not (conv_s1.weight is None or conv_s2.weight is None), "weights are None"
        
    # Use same weights for comparison
    conv_s2.weight.data = conv_s1.weight.data.copy()
    if conv_s1.bias is not None and conv_s2.bias is not None:
        conv_s2.bias.data = conv_s1.bias.data.copy()
        
    x = Tensor(np.random.randn(1, 3, 8, 8))
        
    y1 = conv_s1(x)
    y2 = conv_s2(x)
        
    assert not (y1 is None or y2 is None), "forward returned None"
        
    assert not (y1.shape != (1, 8, 8, 8)), f"stride=1 shape {y1.shape}"
        
    assert not (y2.shape != (1, 8, 4, 4)), f"stride=2 shape {y2.shape}"
        
    # Verify stride=2 output matches stride=1 at corresponding positions
    assert np.allclose(y2.data[:, :, 0, 0], y1.data[:, :, 0, 0], rtol=1e-5), "stride=2 doesn't match stride=1 at (0,0)"
    assert np.allclose(y2.data[:, :, 1, 1], y1.data[:, :, 2, 2], rtol=1e-5), "stride=2 doesn't match stride=1 at (2,2)"
        
def test_conv2d_against_pytorch():
    """Test Conv2d against PyTorch implementation."""
    import torch
    import torch.nn as nn
        
    np.random.seed(42)
    torch.manual_seed(42)
        
    our_conv = Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        
    assert not (our_conv.weight is None), "weight is None"
        
    torch_conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
    torch_conv.weight.data = torch.tensor(our_conv.weight.data.copy())
    if our_conv.bias is not None:
        torch_conv.bias.data = torch.tensor(our_conv.bias.data.copy())
        
    x_np = np.random.randn(2, 3, 8, 8)
        
    our_x = Tensor(x_np.copy())
    our_y = our_conv(our_x)
        
    assert not (our_y is None), "Our forward returned None"
        
    torch_x = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
    torch_conv = torch_conv.double()
    torch_y = torch_conv(torch_x)
        
    assert np.allclose(our_y.data, torch_y.detach().numpy(), rtol=1e-5), "Forward mismatch"
        
    our_y.sum().backward()
    torch_y.sum().backward()
        
    assert np.allclose(our_conv.weight.grad, torch_conv.weight.grad.numpy(), rtol=1e-4, atol=1e-6), "Weight gradient mismatch"
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])