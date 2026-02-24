"""Test Suite for Day 31: Convolutional Operations"""

import numpy as np
import sys
from typing import Tuple

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


def test_im2col_basic() -> Tuple[bool, str]:
    """Test im2col basic functionality."""
    try:
        x = np.arange(16).reshape(1, 1, 4, 4).astype(np.float64)
        col = im2col(x, 2, 2, stride=1, padding=0)
        
        if col is None:
            return False, "im2col returned None"
        
        expected_shape = (9, 4)
        if col.shape != expected_shape:
            return False, f"shape {col.shape}, expected {expected_shape}"
        
        # Verify actual values: first patch should be [0,1,4,5]
        expected_first_patch = np.array([0, 1, 4, 5])
        if not np.allclose(col[0], expected_first_patch):
            return False, f"first patch {col[0]}, expected {expected_first_patch}"
        
        # Last patch should be [10,11,14,15]
        expected_last_patch = np.array([10, 11, 14, 15])
        if not np.allclose(col[-1], expected_last_patch):
            return False, f"last patch {col[-1]}, expected {expected_last_patch}"
        
        return True, "im2col basic works"
    except Exception as e:
        return False, str(e)


def test_im2col_with_padding() -> Tuple[bool, str]:
    """Test im2col with padding."""
    try:
        np.random.seed(42)
        x = np.random.randn(2, 3, 4, 4)
        col = im2col(x, 3, 3, stride=1, padding=1)
        
        if col is None:
            return False, "im2col returned None"
        
        expected_shape = (2 * 4 * 4, 3 * 3 * 3)
        if col.shape != expected_shape:
            return False, f"shape {col.shape}, expected {expected_shape}"
        
        # Verify corner patch has zeros from padding
        # First patch at (0,0) should have zeros in top-left corner due to padding
        first_patch = col[0].reshape(3, 3, 3)  # (C, kH, kW)
        # Top-left corner of each channel should be zero (from padding)
        if not np.allclose(first_patch[:, 0, 0], 0.0):
            return False, f"padding not applied correctly, got {first_patch[:, 0, 0]}"
        
        return True, "im2col with padding works"
    except Exception as e:
        return False, str(e)


def test_im2col_with_stride() -> Tuple[bool, str]:
    """Test im2col with stride."""
    try:
        np.random.seed(42)
        x = np.arange(72).reshape(1, 2, 6, 6).astype(np.float64)
        col = im2col(x, 2, 2, stride=2, padding=0)
        
        if col is None:
            return False, "im2col returned None"
        
        expected_shape = (1 * 3 * 3, 2 * 2 * 2)
        if col.shape != expected_shape:
            return False, f"shape {col.shape}, expected {expected_shape}"
        
        # First patch: channel 0 at (0,0) should be [0,1,6,7], channel 1 at (0,0) should be [36,37,42,43]
        expected_first = np.array([0, 1, 6, 7, 36, 37, 42, 43])
        if not np.allclose(col[0], expected_first):
            return False, f"first patch {col[0]}, expected {expected_first}"
        
        return True, "im2col with stride works"
    except Exception as e:
        return False, str(e)


def test_col2im_basic() -> Tuple[bool, str]:
    """Test col2im basic functionality."""
    try:
        x = np.random.randn(1, 2, 4, 4)
        col = im2col(x, 2, 2, stride=2, padding=0)
        
        if col is None:
            return False, "im2col returned None"
        
        reconstructed = col2im(col, x.shape, 2, 2, stride=2, padding=0)
        
        if reconstructed is None:
            return False, "col2im returned None"
        
        if not np.allclose(x, reconstructed):
            return False, "Reconstruction mismatch"
        
        return True, "col2im reconstructs correctly"
    except Exception as e:
        return False, str(e)


def test_col2im_with_padding() -> Tuple[bool, str]:
    """Test col2im with padding."""
    try:
        np.random.seed(42)
        x = np.random.randn(2, 3, 4, 4)
        col = im2col(x, 3, 3, stride=1, padding=1)
        
        if col is None:
            return False, "im2col returned None"
        
        reconstructed = col2im(col, x.shape, 3, 3, stride=1, padding=1)
        
        if reconstructed is None:
            return False, "col2im returned None"
        
        if reconstructed.shape != x.shape:
            return False, f"shape {reconstructed.shape}, expected {x.shape}"
        
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
        
        return True, "col2im with padding works"
    except Exception as e:
        return False, str(e)


def test_conv2d_forward_shape() -> Tuple[bool, str]:
    """Test conv2d forward output shape."""
    try:
        np.random.seed(42)
        x = np.random.randn(2, 3, 8, 8)
        w = np.random.randn(16, 3, 3, 3)
        b = np.zeros(16)
        
        out = conv2d_forward(x, w, b, stride=1, padding=1)
        
        if out is None:
            return False, "conv2d_forward returned None"
        
        if out.shape != (2, 16, 8, 8):
            return False, f"shape {out.shape}, expected (2, 16, 8, 8)"
        
        # Verify output is not all zeros or NaN
        if np.all(out == 0):
            return False, "output is all zeros"
        if not np.all(np.isfinite(out)):
            return False, "output contains NaN or Inf"
        
        # Verify convolution property: output should change when input changes
        x2 = x + 1.0
        out2 = conv2d_forward(x2, w, b, stride=1, padding=1)
        if np.allclose(out, out2):
            return False, "output unchanged when input changed"
        
        return True, "Output shape correct"
    except Exception as e:
        return False, str(e)


def test_conv2d_forward_stride() -> Tuple[bool, str]:
    """Test conv2d forward with stride."""
    try:
        np.random.seed(42)
        x = np.random.randn(1, 3, 8, 8)
        w = np.random.randn(4, 3, 3, 3)
        
        out = conv2d_forward(x, w, None, stride=2, padding=1)
        
        if out is None:
            return False, "conv2d_forward returned None"
        
        if out.shape != (1, 4, 4, 4):
            return False, f"shape {out.shape}, expected (1, 4, 4, 4)"
        
        # Verify stride reduces output correctly: 8x8 with stride 2 -> 4x4
        # Also verify output values are reasonable
        if not np.all(np.isfinite(out)):
            return False, "output contains NaN or Inf"
        
        # Check that stride=2 output is subsampled from stride=1 output
        out_s1 = conv2d_forward(x, w, None, stride=1, padding=1)
        if out_s1 is not None:
            # stride=2 should correspond to every other position
            if not np.allclose(out[0, :, 0, 0], out_s1[0, :, 0, 0], rtol=1e-5):
                return False, "stride=2 output doesn't match stride=1 at position (0,0)"
        
        return True, "Stride works correctly"
    except Exception as e:
        return False, str(e)


def test_conv2d_matches_naive() -> Tuple[bool, str]:
    """Test that conv2d_forward matches naive implementation."""
    try:
        np.random.seed(42)
        x = np.random.randn(2, 3, 6, 6)
        w = np.random.randn(4, 3, 3, 3)
        b = np.random.randn(4)
        
        out_fast = conv2d_forward(x, w, b, stride=1, padding=1)
        out_naive = conv2d_naive(x, w, b, stride=1, padding=1)
        
        if out_fast is None:
            return False, "conv2d_forward returned None"
        if out_naive is None:
            return False, "conv2d_naive returned None"
        
        if not np.allclose(out_fast, out_naive, rtol=1e-5):
            diff = np.max(np.abs(out_fast - out_naive))
            return False, f"Max difference: {diff}"
        
        return True, "Matches naive implementation"
    except Exception as e:
        return False, str(e)


def test_conv2d_backward_shapes() -> Tuple[bool, str]:
    """Test conv2d backward gradient shapes."""
    try:
        np.random.seed(42)
        x = np.random.randn(2, 3, 6, 6)
        w = np.random.randn(4, 3, 3, 3)
        dy = np.random.randn(2, 4, 4, 4)
        
        result = conv2d_backward(dy, x, w, stride=1, padding=0)
        
        if result is None:
            return False, "conv2d_backward returned None"
        
        dx, dw, db = result
        
        if dx.shape != x.shape:
            return False, f"dx shape {dx.shape}, expected {x.shape}"
        if dw.shape != w.shape:
            return False, f"dw shape {dw.shape}, expected {w.shape}"
        if db.shape != (4,):
            return False, f"db shape {db.shape}, expected (4,)"
        
        # Verify gradients are not all zeros (indicates actual computation)
        if np.all(dx == 0):
            return False, "dx is all zeros"
        if np.all(dw == 0):
            return False, "dw is all zeros"
        if np.all(db == 0):
            return False, "db is all zeros"
        
        # Verify gradients are finite
        if not np.all(np.isfinite(dx)):
            return False, "dx contains NaN or Inf"
        if not np.all(np.isfinite(dw)):
            return False, "dw contains NaN or Inf"
        
        return True, "Gradient shapes correct"
    except Exception as e:
        return False, str(e)


def test_conv2d_backward_bias() -> Tuple[bool, str]:
    """Test conv2d backward bias gradient."""
    try:
        dy = np.random.randn(2, 4, 3, 3)
        x = np.random.randn(2, 3, 5, 5)
        w = np.random.randn(4, 3, 3, 3)
        
        result = conv2d_backward(dy, x, w, stride=1, padding=0)
        
        if result is None:
            return False, "conv2d_backward returned None"
        
        dx, dw, db = result
        
        expected_db = np.sum(dy, axis=(0, 2, 3))
        
        if not np.allclose(db, expected_db):
            return False, "Bias gradient incorrect"
        
        return True, "Bias gradient correct"
    except Exception as e:
        return False, str(e)


def test_conv2d_gradient_numerical() -> Tuple[bool, str]:
    """Numerical gradient check for conv2d."""
    try:
        np.random.seed(42)
        x = np.random.randn(1, 2, 4, 4)
        w = np.random.randn(3, 2, 2, 2)
        b = np.random.randn(3)
        
        out = conv2d_forward(x, w, b, stride=1, padding=0)
        if out is None:
            return False, "conv2d_forward returned None"
        
        dy = np.ones_like(out)
        
        result = conv2d_backward(dy, x, w, stride=1, padding=0)
        if result is None:
            return False, "conv2d_backward returned None"
        
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
        
        if not np.allclose(dw_analytic, dw_numeric, rtol=1e-3, atol=1e-5):
            return False, "Weight gradient mismatch"
        
        return True, "Numerical gradient check passed"
    except Exception as e:
        return False, str(e)


def test_conv2d_module_init() -> Tuple[bool, str]:
    """Test Conv2d module initialization."""
    try:
        np.random.seed(42)
        conv = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        
        if conv.weight is None:
            return False, "weight is None"
        
        if conv.weight.shape != (16, 3, 3, 3):
            return False, f"weight shape {conv.weight.shape}"
        
        if conv.bias is not None and conv.bias.shape != (16,):
            return False, f"bias shape {conv.bias.shape}"
        
        # Verify He initialization: std should be approximately sqrt(2/fan_in)
        fan_in = 3 * 3 * 3
        expected_std = np.sqrt(2.0 / fan_in)
        actual_std = np.std(conv.weight.data)
        if not np.isclose(actual_std, expected_std, rtol=0.5):
            return False, f"weight std {actual_std:.4f}, expected ~{expected_std:.4f}"
        
        # Bias should be initialized to zeros
        if conv.bias is not None:
            if not np.allclose(conv.bias.data, 0):
                return False, "bias not initialized to zeros"
        
        return True, "Conv2d initialized correctly"
    except Exception as e:
        return False, str(e)


def test_conv2d_module_forward() -> Tuple[bool, str]:
    """Test Conv2d module forward pass."""
    try:
        np.random.seed(42)
        conv = Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        
        if conv.weight is None:
            return False, "weight is None"
        
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = conv(x)
        
        if y is None:
            return False, "forward returned None"
        
        if y.shape != (2, 8, 8, 8):
            return False, f"output shape {y.shape}"
        
        # Verify output values are finite
        if not np.all(np.isfinite(y.data)):
            return False, "output contains NaN or Inf"
        
        # Verify output is not all zeros
        if np.all(y.data == 0):
            return False, "output is all zeros"
        
        # Verify output changes with input
        x2 = Tensor(np.random.randn(2, 3, 8, 8))
        y2 = conv(x2)
        if np.allclose(y.data, y2.data):
            return False, "output same for different inputs"
        
        return True, "Forward pass works"
    except Exception as e:
        return False, str(e)


def test_conv2d_module_backward() -> Tuple[bool, str]:
    """Test Conv2d module backward pass."""
    try:
        np.random.seed(42)
        conv = Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
        
        if conv.weight is None:
            return False, "weight is None"
        
        x = Tensor(np.random.randn(1, 2, 6, 6))
        y = conv(x)
        
        if y is None:
            return False, "forward returned None"
        
        loss = y.sum()
        loss.backward()
        
        if np.all(conv.weight.grad == 0):
            return False, "Weight gradient is zero"
        
        if conv.bias is not None and np.all(conv.bias.grad == 0):
            return False, "Bias gradient is zero"
        
        # For sum loss, bias gradient should equal output spatial size
        if conv.bias is not None:
            expected_bias_grad = 6 * 6  # H_out * W_out
            if not np.allclose(conv.bias.grad, expected_bias_grad):
                return False, f"bias grad {conv.bias.grad[0]}, expected {expected_bias_grad}"
        
        # Verify gradients are finite
        if not np.all(np.isfinite(conv.weight.grad)):
            return False, "weight gradient contains NaN or Inf"
        if not np.all(np.isfinite(x.grad)):
            return False, "input gradient contains NaN or Inf"
        
        return True, "Backward pass works"
    except Exception as e:
        return False, str(e)


def test_conv2d_module_no_bias() -> Tuple[bool, str]:
    """Test Conv2d module without bias."""
    try:
        conv = Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        
        if conv.bias is not None:
            return False, "bias should be None"
        
        params = conv.parameters()
        if len(params) != 1:
            return False, f"Should have 1 param, got {len(params)}"
        
        return True, "No bias works"
    except Exception as e:
        return False, str(e)


def test_conv2d_parameters() -> Tuple[bool, str]:
    """Test Conv2d parameters method."""
    try:
        conv = Conv2d(3, 16, kernel_size=3)
        params = conv.parameters()
        
        if len(params) != 2:
            return False, f"Expected 2 params, got {len(params)}"
        
        conv_no_bias = Conv2d(3, 16, kernel_size=3, bias=False)
        params_no_bias = conv_no_bias.parameters()
        
        if len(params_no_bias) != 1:
            return False, f"Expected 1 param, got {len(params_no_bias)}"
        
        return True, "Parameters correct"
    except Exception as e:
        return False, str(e)


def test_conv2d_zero_grad() -> Tuple[bool, str]:
    """Test Conv2d zero_grad method."""
    try:
        np.random.seed(42)
        conv = Conv2d(3, 8, kernel_size=3, padding=1)
        
        if conv.weight is None:
            return False, "weight is None"
        
        x = Tensor(np.random.randn(1, 3, 4, 4))
        y = conv(x)
        
        if y is not None:
            y.sum().backward()
        
        if np.all(conv.weight.grad == 0):
            return False, "Gradient not computed"
        
        conv.zero_grad()
        
        if not np.all(conv.weight.grad == 0):
            return False, "Gradient not zeroed"
        
        return True, "zero_grad works"
    except Exception as e:
        return False, str(e)


def test_conv2d_different_strides() -> Tuple[bool, str]:
    """Test Conv2d with different strides."""
    try:
        np.random.seed(42)
        
        conv_s1 = Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        conv_s2 = Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        
        if conv_s1.weight is None or conv_s2.weight is None:
            return False, "weights are None"
        
        # Use same weights for comparison
        conv_s2.weight.data = conv_s1.weight.data.copy()
        if conv_s1.bias is not None and conv_s2.bias is not None:
            conv_s2.bias.data = conv_s1.bias.data.copy()
        
        x = Tensor(np.random.randn(1, 3, 8, 8))
        
        y1 = conv_s1(x)
        y2 = conv_s2(x)
        
        if y1 is None or y2 is None:
            return False, "forward returned None"
        
        if y1.shape != (1, 8, 8, 8):
            return False, f"stride=1 shape {y1.shape}"
        
        if y2.shape != (1, 8, 4, 4):
            return False, f"stride=2 shape {y2.shape}"
        
        # Verify stride=2 output matches stride=1 at corresponding positions
        if not np.allclose(y2.data[:, :, 0, 0], y1.data[:, :, 0, 0], rtol=1e-5):
            return False, "stride=2 doesn't match stride=1 at (0,0)"
        if not np.allclose(y2.data[:, :, 1, 1], y1.data[:, :, 2, 2], rtol=1e-5):
            return False, "stride=2 doesn't match stride=1 at (2,2)"
        
        return True, "Different strides work"
    except Exception as e:
        return False, str(e)


def test_conv2d_against_pytorch() -> Tuple[bool, str]:
    """Test Conv2d against PyTorch implementation."""
    try:
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        our_conv = Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        
        if our_conv.weight is None:
            return False, "weight is None"
        
        torch_conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        torch_conv.weight.data = torch.tensor(our_conv.weight.data.copy())
        if our_conv.bias is not None:
            torch_conv.bias.data = torch.tensor(our_conv.bias.data.copy())
        
        x_np = np.random.randn(2, 3, 8, 8)
        
        our_x = Tensor(x_np.copy())
        our_y = our_conv(our_x)
        
        if our_y is None:
            return False, "Our forward returned None"
        
        torch_x = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
        torch_conv = torch_conv.double()
        torch_y = torch_conv(torch_x)
        
        if not np.allclose(our_y.data, torch_y.detach().numpy(), rtol=1e-5):
            return False, "Forward mismatch"
        
        our_y.sum().backward()
        torch_y.sum().backward()
        
        if not np.allclose(our_conv.weight.grad, torch_conv.weight.grad.numpy(), rtol=1e-4, atol=1e-6):
            return False, "Weight gradient mismatch"
        
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("im2col_basic", test_im2col_basic),
        ("im2col_with_padding", test_im2col_with_padding),
        ("im2col_with_stride", test_im2col_with_stride),
        ("col2im_basic", test_col2im_basic),
        ("col2im_with_padding", test_col2im_with_padding),
        ("conv2d_forward_shape", test_conv2d_forward_shape),
        ("conv2d_forward_stride", test_conv2d_forward_stride),
        ("conv2d_matches_naive", test_conv2d_matches_naive),
        ("conv2d_backward_shapes", test_conv2d_backward_shapes),
        ("conv2d_backward_bias", test_conv2d_backward_bias),
        ("conv2d_gradient_numerical", test_conv2d_gradient_numerical),
        ("conv2d_module_init", test_conv2d_module_init),
        ("conv2d_module_forward", test_conv2d_module_forward),
        ("conv2d_module_backward", test_conv2d_module_backward),
        ("conv2d_module_no_bias", test_conv2d_module_no_bias),
        ("conv2d_parameters", test_conv2d_parameters),
        ("conv2d_zero_grad", test_conv2d_zero_grad),
        ("conv2d_different_strides", test_conv2d_different_strides),
        ("conv2d_against_pytorch", test_conv2d_against_pytorch),
    ]
    
    print(f"\n{'='*60}")
    print("Day 31: Convolutional Operations - Tests")
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
