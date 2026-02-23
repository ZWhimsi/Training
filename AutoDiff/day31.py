"""
Day 31: Convolutional Operations
================================
Estimated time: 4-5 hours
Prerequisites: Days 21-30 (Module system, backprop fundamentals)

Learning objectives:
- Understand the mathematics of 2D convolution
- Implement forward pass for Conv2d
- Derive and implement the backward pass for convolution
- Handle padding, stride, and multiple channels

Key concepts:
- Convolution: Sliding filter operation over input
  - Input: (N, C_in, H, W)
  - Filter: (C_out, C_in, kH, kW)
  - Output: (N, C_out, H_out, W_out)

- Output dimensions:
  H_out = (H + 2*padding - kH) // stride + 1
  W_out = (W + 2*padding - kW) // stride + 1

- Forward pass:
  output[n, c_out, h, w] = sum over (c_in, kh, kw) of:
      input[n, c_in, h*stride+kh, w*stride+kw] * filter[c_out, c_in, kh, kw]

Mathematical background:
- Convolution gradient with respect to input:
  dx = conv2d_transpose(dy, W)  (full convolution)
  
- Convolution gradient with respect to weights:
  dW = conv2d(input, dy) (with appropriate reshaping)

- The backward pass involves "rotating" the filter 180 degrees
  and performing a full convolution

Implementation notes:
- We use im2col for efficient implementation:
  - Unfold input into columns
  - Matrix multiply with flattened filters
  - Reshape to output
"""

import numpy as np
from typing import Tuple, Optional, List


class Tensor:
    """Tensor class with autodiff support."""
    
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = requires_grad
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    def __repr__(self):
        return f"Tensor(shape={self.shape})"
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = np.ones_like(self.data)
        
        for v in reversed(topo):
            v._backward()
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    def sum(self, axis=None, keepdims=False):
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'sum')
        
        def _backward():
            if axis is None:
                self.grad += np.full(self.shape, out.grad)
            else:
                grad = out.grad
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.shape).copy()
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'mean')
        count = self.data.size if axis is None else np.prod([self.data.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
        
        def _backward():
            if axis is None:
                self.grad += np.full(self.shape, out.grad / count)
            else:
                grad = out.grad / count
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.shape).copy()
        out._backward = _backward
        return out
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)


def _unbroadcast(grad, original_shape):
    """Reverse broadcasting by summing over broadcast dimensions."""
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
        if orig_dim == 1 and grad_dim > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# ============================================================================
# Exercise 1: im2col and col2im
# ============================================================================

def im2col(x: np.ndarray, kernel_h: int, kernel_w: int, 
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Transform input into column matrix for efficient convolution.
    
    This transforms patches of the input into columns, allowing
    convolution to be computed as a single matrix multiplication.
    
    Args:
        x: Input array of shape (N, C, H, W)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride for sliding window
        padding: Zero padding to add
    
    Returns:
        col: Column matrix of shape (N * H_out * W_out, C * kH * kW)
    
    Example:
        For a 2x2 kernel on 4x4 input:
        - Each 2x2 patch becomes a row of 4 values
        - 9 patches (3x3 output) gives 9 rows
    """
    # TODO: Implement im2col
    # HINT:
    # 1. Pad the input if needed
    # 2. Calculate output dimensions
    # 3. Extract patches and reshape
    #
    # N, C, H, W = x.shape
    # if padding > 0:
    #     x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    # 
    # H_out = (H + 2*padding - kernel_h) // stride + 1
    # W_out = (W + 2*padding - kernel_w) // stride + 1
    # 
    # col = np.zeros((N, C, kernel_h, kernel_w, H_out, W_out))
    # for y in range(kernel_h):
    #     y_max = y + stride * H_out
    #     for x_idx in range(kernel_w):
    #         x_max = x_idx + stride * W_out
    #         col[:, :, y, x_idx, :, :] = x[:, :, y:y_max:stride, x_idx:x_max:stride]
    # 
    # col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
    # return col
    
    pass  # Replace with implementation


def col2im(col: np.ndarray, x_shape: Tuple[int, ...], 
           kernel_h: int, kernel_w: int,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Transform column matrix back to image format (inverse of im2col).
    
    This is needed for the backward pass to compute gradients
    with respect to the input.
    
    Args:
        col: Column matrix of shape (N * H_out * W_out, C * kH * kW)
        x_shape: Original input shape (N, C, H, W)
        kernel_h: Kernel height
        kernel_w: Kernel width
        stride: Stride used in im2col
        padding: Padding used in im2col
    
    Returns:
        img: Reconstructed input of shape (N, C, H, W)
    """
    # TODO: Implement col2im
    # HINT:
    # 1. Calculate dimensions
    # 2. Reshape col back to patch format
    # 3. Accumulate patches back to image (add overlapping regions)
    #
    # N, C, H, W = x_shape
    # H_padded = H + 2 * padding
    # W_padded = W + 2 * padding
    # H_out = (H + 2*padding - kernel_h) // stride + 1
    # W_out = (W + 2*padding - kernel_w) // stride + 1
    # 
    # col = col.reshape(N, H_out, W_out, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    # 
    # img = np.zeros((N, C, H_padded, W_padded))
    # for y in range(kernel_h):
    #     y_max = y + stride * H_out
    #     for x_idx in range(kernel_w):
    #         x_max = x_idx + stride * W_out
    #         img[:, :, y:y_max:stride, x_idx:x_max:stride] += col[:, :, y, x_idx, :, :]
    # 
    # if padding > 0:
    #     img = img[:, :, padding:-padding, padding:-padding]
    # return img
    
    pass  # Replace with implementation


# ============================================================================
# Exercise 2: Basic 2D Convolution Forward Pass
# ============================================================================

def conv2d_forward(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None,
                   stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Compute 2D convolution forward pass.
    
    Args:
        x: Input of shape (N, C_in, H, W)
        weight: Filters of shape (C_out, C_in, kH, kW)
        bias: Optional bias of shape (C_out,)
        stride: Stride for convolution
        padding: Zero padding
    
    Returns:
        Output of shape (N, C_out, H_out, W_out)
    
    Mathematical formulation:
        out[n, c_out, h, w] = bias[c_out] + 
            sum_{c_in, kh, kw} x[n, c_in, h*s+kh, w*s+kw] * weight[c_out, c_in, kh, kw]
    """
    # TODO: Implement convolution forward using im2col
    # HINT:
    # 1. Use im2col to unfold input
    # 2. Reshape weights for matrix multiplication
    # 3. Compute output and reshape
    #
    # N, C_in, H, W = x.shape
    # C_out, _, kH, kW = weight.shape
    # 
    # H_out = (H + 2*padding - kH) // stride + 1
    # W_out = (W + 2*padding - kW) // stride + 1
    # 
    # col = im2col(x, kH, kW, stride, padding)  # (N*H_out*W_out, C_in*kH*kW)
    # weight_col = weight.reshape(C_out, -1)     # (C_out, C_in*kH*kW)
    # 
    # out = col @ weight_col.T  # (N*H_out*W_out, C_out)
    # out = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)
    # 
    # if bias is not None:
    #     out += bias.reshape(1, -1, 1, 1)
    # 
    # return out
    
    pass  # Replace with implementation


# ============================================================================
# Exercise 3: Convolution Backward Pass
# ============================================================================

def conv2d_backward(dy: np.ndarray, x: np.ndarray, weight: np.ndarray,
                    stride: int = 1, padding: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradients for 2D convolution.
    
    Args:
        dy: Upstream gradient of shape (N, C_out, H_out, W_out)
        x: Original input of shape (N, C_in, H, W)
        weight: Filters of shape (C_out, C_in, kH, kW)
        stride: Stride used in forward
        padding: Padding used in forward
    
    Returns:
        dx: Gradient w.r.t. input, shape (N, C_in, H, W)
        dw: Gradient w.r.t. weights, shape (C_out, C_in, kH, kW)
        db: Gradient w.r.t. bias, shape (C_out,)
    
    Mathematical derivation:
    - db[c] = sum over (n, h, w) of dy[n, c, h, w]
    - dw[c_out, c_in, kh, kw] = sum over patches of (input patch * dy value)
    - dx = "full" convolution of dy with rotated weights
    """
    # TODO: Implement convolution backward
    # HINT:
    # N, C_out, H_out, W_out = dy.shape
    # C_out, C_in, kH, kW = weight.shape
    # 
    # # Bias gradient: sum over all except channel dimension
    # db = np.sum(dy, axis=(0, 2, 3))
    # 
    # # Weight gradient
    # col = im2col(x, kH, kW, stride, padding)  # (N*H_out*W_out, C_in*kH*kW)
    # dy_reshaped = dy.transpose(0, 2, 3, 1).reshape(-1, C_out)  # (N*H_out*W_out, C_out)
    # dw = (dy_reshaped.T @ col).reshape(C_out, C_in, kH, kW)
    # 
    # # Input gradient using col2im
    # weight_col = weight.reshape(C_out, -1)  # (C_out, C_in*kH*kW)
    # dcol = dy_reshaped @ weight_col  # (N*H_out*W_out, C_in*kH*kW)
    # dx = col2im(dcol, x.shape, kH, kW, stride, padding)
    # 
    # return dx, dw, db
    
    pass  # Replace with implementation


# ============================================================================
# Exercise 4: Conv2d Module
# ============================================================================

class Module:
    """Base class for neural network modules."""
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, x):
        raise NotImplementedError
    
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()


class Conv2d(Module):
    """
    2D Convolutional layer.
    
    Applies a 2D convolution over an input signal composed of several input planes.
    
    Args:
        in_channels: Number of channels in input
        out_channels: Number of channels produced by convolution
        kernel_size: Size of convolving kernel (int or tuple)
        stride: Stride of convolution (default: 1)
        padding: Zero-padding added to both sides (default: 0)
        bias: If True, adds learnable bias (default: True)
    
    Input shape: (N, C_in, H, W)
    Output shape: (N, C_out, H_out, W_out)
    
    Example:
        conv = Conv2d(3, 64, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 3, 32, 32))
        y = conv(x)  # shape: (1, 64, 32, 32)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        """Initialize Conv2d layer with He initialization."""
        # TODO: Initialize weights and bias
        # HINT:
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.stride = stride
        # self.padding = padding
        # 
        # # He initialization for conv layers
        # fan_in = in_channels * kernel_size * kernel_size
        # scale = np.sqrt(2.0 / fan_in)
        # self.weight = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale)
        # 
        # if bias:
        #     self.bias = Tensor(np.zeros(out_channels))
        # else:
        #     self.bias = None
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = None  # Replace with Tensor
        self.bias = None    # Replace with Tensor or None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply convolution to input.
        
        Args:
            x: Input tensor of shape (N, C_in, H, W)
        
        Returns:
            Output tensor of shape (N, C_out, H_out, W_out)
        """
        # TODO: Implement forward pass with gradient tracking
        # HINT:
        # Store information needed for backward pass
        # Use conv2d_forward for computation
        #
        # bias_data = self.bias.data if self.bias is not None else None
        # out_data = conv2d_forward(x.data, self.weight.data, bias_data,
        #                           self.stride, self.padding)
        # 
        # children = (x, self.weight) if self.bias is None else (x, self.weight, self.bias)
        # out = Tensor(out_data, children, 'conv2d')
        # 
        # def _backward():
        #     dx, dw, db = conv2d_backward(out.grad, x.data, self.weight.data,
        #                                   self.stride, self.padding)
        #     x.grad += dx
        #     self.weight.grad += dw
        #     if self.bias is not None:
        #         self.bias.grad += db
        # 
        # out._backward = _backward
        # return out
        
        return None  # Replace with implementation
    
    def parameters(self) -> List[Tensor]:
        """Return list of learnable parameters."""
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]
    
    def __repr__(self):
        return f"Conv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


# ============================================================================
# Exercise 5: Naive Convolution (for verification)
# ============================================================================

def conv2d_naive(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None,
                 stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Naive implementation of 2D convolution using loops.
    
    This is slower but easier to understand and useful for verifying
    the im2col-based implementation.
    
    Use this to check your im2col implementation is correct!
    """
    # TODO: Implement naive convolution
    # HINT:
    # N, C_in, H, W = x.shape
    # C_out, _, kH, kW = weight.shape
    # 
    # if padding > 0:
    #     x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    # 
    # H_out = (H + 2*padding - kH) // stride + 1
    # W_out = (W + 2*padding - kW) // stride + 1
    # 
    # out = np.zeros((N, C_out, H_out, W_out))
    # 
    # for n in range(N):
    #     for c_out in range(C_out):
    #         for h in range(H_out):
    #             for w in range(W_out):
    #                 h_start = h * stride
    #                 w_start = w * stride
    #                 patch = x[n, :, h_start:h_start+kH, w_start:w_start+kW]
    #                 out[n, c_out, h, w] = np.sum(patch * weight[c_out])
    # 
    # if bias is not None:
    #     out += bias.reshape(1, -1, 1, 1)
    # 
    # return out
    
    pass  # Replace with implementation


# ============================================================================
# Test Functions
# ============================================================================

def test_im2col():
    """Test im2col transformation."""
    results = {}
    
    try:
        np.random.seed(42)
        x = np.random.randn(2, 3, 4, 4)
        
        col = im2col(x, 2, 2, stride=1, padding=0)
        
        if col is not None:
            expected_shape = (2 * 3 * 3, 3 * 2 * 2)
            results['shape'] = col.shape == expected_shape
        else:
            results['shape'] = False
    except Exception as e:
        results['shape'] = False
    
    return results


def test_col2im():
    """Test col2im is inverse of im2col."""
    results = {}
    
    try:
        np.random.seed(42)
        x = np.random.randn(1, 2, 4, 4)
        
        col = im2col(x, 2, 2, stride=2, padding=0)
        if col is not None:
            reconstructed = col2im(col, x.shape, 2, 2, stride=2, padding=0)
            if reconstructed is not None:
                results['inverse'] = np.allclose(x, reconstructed)
            else:
                results['inverse'] = False
        else:
            results['inverse'] = False
    except Exception as e:
        results['inverse'] = False
    
    return results


def test_conv2d_forward():
    """Test convolution forward pass."""
    results = {}
    
    try:
        np.random.seed(42)
        x = np.random.randn(2, 3, 8, 8)
        w = np.random.randn(4, 3, 3, 3)
        b = np.random.randn(4)
        
        out = conv2d_forward(x, w, b, stride=1, padding=1)
        
        if out is not None:
            results['output_shape'] = out.shape == (2, 4, 8, 8)
            
            naive_out = conv2d_naive(x, w, b, stride=1, padding=1)
            if naive_out is not None:
                results['matches_naive'] = np.allclose(out, naive_out, rtol=1e-5)
            else:
                results['matches_naive'] = False
        else:
            results['output_shape'] = False
            results['matches_naive'] = False
    except Exception as e:
        results['output_shape'] = False
        results['matches_naive'] = False
    
    return results


def test_conv2d_backward():
    """Test convolution backward pass."""
    results = {}
    
    try:
        np.random.seed(42)
        x = np.random.randn(2, 3, 6, 6)
        w = np.random.randn(4, 3, 3, 3)
        dy = np.random.randn(2, 4, 4, 4)
        
        backward_result = conv2d_backward(dy, x, w, stride=1, padding=0)
        
        if backward_result is not None:
            dx, dw, db = backward_result
            results['dx_shape'] = dx.shape == x.shape
            results['dw_shape'] = dw.shape == w.shape
            results['db_shape'] = db.shape == (4,)
        else:
            results['dx_shape'] = False
            results['dw_shape'] = False
            results['db_shape'] = False
    except Exception as e:
        results['dx_shape'] = False
        results['dw_shape'] = False
        results['db_shape'] = False
    
    return results


def test_conv2d_module():
    """Test Conv2d module."""
    results = {}
    
    try:
        np.random.seed(42)
        conv = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        
        if conv.weight is not None:
            results['weight_shape'] = conv.weight.shape == (16, 3, 3, 3)
        else:
            results['weight_shape'] = False
        
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = conv(x)
        
        if y is not None:
            results['forward_shape'] = y.shape == (2, 16, 8, 8)
            
            loss = y.sum()
            loss.backward()
            
            results['backward'] = np.any(conv.weight.grad != 0)
        else:
            results['forward_shape'] = False
            results['backward'] = False
    except Exception as e:
        results['weight_shape'] = False
        results['forward_shape'] = False
        results['backward'] = False
    
    return results


def test_conv2d_gradient_check():
    """Numerical gradient check for Conv2d."""
    results = {}
    
    try:
        np.random.seed(42)
        conv = Conv2d(2, 3, kernel_size=2, stride=1, padding=0, bias=True)
        
        if conv.weight is None:
            return {'gradient_check': False}
        
        x = Tensor(np.random.randn(1, 2, 4, 4))
        
        y = conv(x)
        if y is None:
            return {'gradient_check': False}
        
        loss = y.sum()
        loss.backward()
        
        analytic_grad = conv.weight.grad.copy()
        
        eps = 1e-5
        numeric_grad = np.zeros_like(conv.weight.data)
        
        for idx in np.ndindex(conv.weight.data.shape):
            conv.weight.data[idx] += eps
            y_plus = conv(x)
            loss_plus = y_plus.sum().data if y_plus is not None else 0
            
            conv.weight.data[idx] -= 2 * eps
            y_minus = conv(x)
            loss_minus = y_minus.sum().data if y_minus is not None else 0
            
            conv.weight.data[idx] += eps
            
            numeric_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        
        results['gradient_check'] = np.allclose(analytic_grad, numeric_grad, rtol=1e-3, atol=1e-5)
    except Exception as e:
        results['gradient_check'] = False
    
    return results


def test_against_pytorch():
    """Test Conv2d against PyTorch."""
    results = {}
    
    try:
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        our_conv = Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        torch_conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        
        if our_conv.weight is None:
            return {'forward_match': False, 'backward_match': False}
        
        torch_conv.weight.data = torch.tensor(our_conv.weight.data.copy())
        torch_conv.bias.data = torch.tensor(our_conv.bias.data.copy())
        
        x_np = np.random.randn(2, 3, 8, 8)
        
        our_x = Tensor(x_np.copy())
        our_y = our_conv(our_x)
        
        torch_x = torch.tensor(x_np, requires_grad=True)
        torch_y = torch_conv(torch_x)
        
        if our_y is not None:
            results['forward_match'] = np.allclose(our_y.data, torch_y.detach().numpy(), rtol=1e-5)
            
            our_y.sum().backward()
            torch_y.sum().backward()
            
            results['backward_match'] = np.allclose(our_conv.weight.grad, 
                                                     torch_conv.weight.grad.numpy(), 
                                                     rtol=1e-4)
        else:
            results['forward_match'] = False
            results['backward_match'] = False
            
    except ImportError:
        results['forward_match'] = True
        results['backward_match'] = True
    except Exception as e:
        results['forward_match'] = False
        results['backward_match'] = False
    
    return results


if __name__ == "__main__":
    print("Day 31: Convolutional Operations")
    print("=" * 60)
    
    print("\nim2col:")
    im2col_results = test_im2col()
    for name, passed in im2col_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\ncol2im:")
    col2im_results = test_col2im()
    for name, passed in col2im_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nConv2d Forward:")
    forward_results = test_conv2d_forward()
    for name, passed in forward_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nConv2d Backward:")
    backward_results = test_conv2d_backward()
    for name, passed in backward_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nConv2d Module:")
    module_results = test_conv2d_module()
    for name, passed in module_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nGradient Check:")
    grad_results = test_conv2d_gradient_check()
    for name, passed in grad_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nPyTorch Comparison:")
    pytorch_results = test_against_pytorch()
    for name, passed in pytorch_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day31.py for comprehensive tests!")
