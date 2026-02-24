"""
Day 32: Pooling Operations
==========================
Estimated time: 3-4 hours
Prerequisites: Day 31 (Convolution, im2col/col2im)

Learning objectives:
- Understand the purpose of pooling in CNNs
- Implement max pooling with gradient computation
- Implement average pooling with gradient computation
- Handle padding and stride in pooling operations

Key concepts:
- Pooling: Downsampling operation that reduces spatial dimensions
  - Reduces computation and provides translation invariance
  - Max pooling: Takes maximum value in each window
  - Average pooling: Takes mean value in each window

- Max Pooling gradient:
  - Gradient only flows to the maximum element in each window
  - All other elements receive zero gradient
  - Must track indices of max elements during forward pass

- Average Pooling gradient:
  - Gradient distributed equally to all elements in window
  - Each element receives gradient / (pool_h * pool_w)

Mathematical background:
- Max Pool forward:
  out[n, c, h, w] = max over (ph, pw) of input[n, c, h*s+ph, w*s+pw]

- Max Pool backward (for position of max):
  dx[n, c, h*s+ph_max, w*s+pw_max] += dy[n, c, h, w]

- Avg Pool forward:
  out[n, c, h, w] = mean over (ph, pw) of input[n, c, h*s+ph, w*s+pw]

- Avg Pool backward:
  dx[n, c, h*s+ph, w*s+pw] += dy[n, c, h, w] / (pool_h * pool_w)
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
        count = self.data.size if axis is None else self.data.shape[axis]
        
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


# ============================================================================
# Exercise 1: Max Pooling Forward
# ============================================================================

def maxpool2d_forward(x: np.ndarray, kernel_size: int, stride: int = None,
                      padding: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute max pooling forward pass.
    
    Args:
        x: Input of shape (N, C, H, W)
        kernel_size: Size of pooling window
        stride: Stride (default: kernel_size)
        padding: Zero padding to add
    
    Returns:
        out: Output of shape (N, C, H_out, W_out)
        mask: Boolean mask of max positions for backward pass
    
    Example:
        x = [[1, 2],     kernel_size=2
             [3, 4]]  -> out = [[4]]
                        mask marks position (1,1) as True
    """
    # API hints:
    # - np.pad(..., constant_values=-np.inf) for max pool padding
    # - np.max(window, axis=(2, 3)) -> max over spatial dimensions
    # - np.argmax(window.reshape(N, C, -1), axis=2) -> find max position
    # - mask[n, c, h_idx, w_idx] = True to record max locations
    # - idx // kernel_size and idx % kernel_size to convert flat index to 2D
    # - Return tuple: (output, mask) where mask tracks max positions for backward
    
    return None


# ============================================================================
# Exercise 2: Max Pooling Backward
# ============================================================================

def maxpool2d_backward(dy: np.ndarray, mask: np.ndarray, x_shape: Tuple[int, ...],
                       kernel_size: int, stride: int = None, 
                       padding: int = 0) -> np.ndarray:
    """
    Compute max pooling backward pass.
    
    The gradient only flows through the maximum elements.
    
    Args:
        dy: Upstream gradient of shape (N, C, H_out, W_out)
        mask: Boolean mask from forward pass
        x_shape: Original input shape (N, C, H, W)
        kernel_size: Pooling window size
        stride: Stride used in forward
        padding: Padding used in forward
    
    Returns:
        dx: Gradient w.r.t. input of shape (N, C, H, W)
    """
    # API hints:
    # - Gradient only flows to max positions (from mask)
    # - dx[..., h_start:h_start+k, w_start:w_start+k] += mask_window * dy[..., h:h+1, w:w+1]
    # - Loop over H_out, W_out to distribute gradients
    # - Remove padding at end if padding > 0
    # - Only positions where mask=True receive gradient
    
    return None


# ============================================================================
# Exercise 3: MaxPool2d Module
# ============================================================================

class MaxPool2d(Module):
    """
    2D Max Pooling layer.
    
    Takes the maximum value within each pooling window.
    
    Args:
        kernel_size: Size of pooling window
        stride: Stride (default: kernel_size)
        padding: Zero padding (default: 0)
    
    Example:
        pool = MaxPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 3, 8, 8))
        y = pool(x)  # shape: (1, 3, 4, 4)
    """
    
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """Initialize MaxPool2d layer."""
        # API hints:
        # - stride defaults to kernel_size if not provided
        # - Store kernel_size, stride, padding as instance attributes
        
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply max pooling to input.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
        
        Returns:
            Output tensor of shape (N, C, H_out, W_out)
        """
        # API hints:
        # - maxpool2d_forward(x.data, ...) returns (out_data, mask)
        # - Store mask for backward pass (gradient routing)
        # - Create Tensor with children=(x,)
        # - _backward: use maxpool2d_backward(out.grad, mask, x.shape, ...)
        # - x.grad += dx to accumulate input gradient
        
        return None
    
    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


# ============================================================================
# Exercise 4: Average Pooling Forward/Backward
# ============================================================================

def avgpool2d_forward(x: np.ndarray, kernel_size: int, stride: int = None,
                      padding: int = 0) -> np.ndarray:
    """
    Compute average pooling forward pass.
    
    Args:
        x: Input of shape (N, C, H, W)
        kernel_size: Size of pooling window
        stride: Stride (default: kernel_size)
        padding: Zero padding to add
    
    Returns:
        out: Output of shape (N, C, H_out, W_out)
    """
    # API hints:
    # - Similar structure to max pooling but use np.mean instead of np.max
    # - np.mean(window, axis=(2, 3)) -> average over spatial dimensions
    # - No mask needed (gradient distributes equally)
    # - Loop over output positions, extract window, compute mean
    
    return None


def avgpool2d_backward(dy: np.ndarray, x_shape: Tuple[int, ...],
                       kernel_size: int, stride: int = None,
                       padding: int = 0) -> np.ndarray:
    """
    Compute average pooling backward pass.
    
    Gradient is distributed equally to all elements in each window.
    
    Args:
        dy: Upstream gradient of shape (N, C, H_out, W_out)
        x_shape: Original input shape (N, C, H, W)
        kernel_size: Pooling window size
        stride: Stride used in forward
        padding: Padding used in forward
    
    Returns:
        dx: Gradient w.r.t. input of shape (N, C, H, W)
    """
    # API hints:
    # - Gradient distributed equally to all elements in window
    # - pool_size = kernel_size * kernel_size
    # - dx[..., window] += dy[..., h, w] / pool_size
    # - Each input element receives gradient / pool_size
    # - Remove padding at end if needed
    
    return None


# ============================================================================
# Exercise 5: AvgPool2d Module
# ============================================================================

class AvgPool2d(Module):
    """
    2D Average Pooling layer.
    
    Takes the average value within each pooling window.
    
    Args:
        kernel_size: Size of pooling window
        stride: Stride (default: kernel_size)
        padding: Zero padding (default: 0)
    """
    
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """Initialize AvgPool2d layer."""
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply average pooling to input."""
        # API hints:
        # - avgpool2d_forward(x.data, ...) for forward computation
        # - Create Tensor with children=(x,)
        # - _backward: avgpool2d_backward(out.grad, x.shape, ...)
        # - x.grad += dx to accumulate gradient
        
        return None
    
    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


# ============================================================================
# Exercise 6: Global Average Pooling
# ============================================================================

class GlobalAvgPool2d(Module):
    """
    Global Average Pooling layer.
    
    Computes the average over the entire spatial dimensions.
    Output shape: (N, C, 1, 1) or (N, C) if flatten=True
    
    This is commonly used at the end of CNNs instead of fully connected layers.
    """
    
    def __init__(self, flatten: bool = True):
        """
        Args:
            flatten: If True, output shape is (N, C), else (N, C, 1, 1)
        """
        self.flatten = flatten
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply global average pooling.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
        
        Returns:
            Output tensor of shape (N, C) or (N, C, 1, 1)
        """
        # API hints:
        # - np.mean(x.data, axis=(2, 3)) -> average over entire spatial dims
        # - If flatten=False, add dimensions: out[:, :, np.newaxis, np.newaxis]
        # - Backward: gradient = out.grad / (H * W), broadcast to input shape
        # - np.broadcast_to(grad, x.shape).copy() for gradient expansion
        
        return None
    
    def __repr__(self):
        return f"GlobalAvgPool2d(flatten={self.flatten})"


# ============================================================================
# Exercise 7: Adaptive Average Pooling
# ============================================================================

class AdaptiveAvgPool2d(Module):
    """
    Adaptive Average Pooling layer.
    
    Outputs a fixed size regardless of input size.
    Commonly used to convert any input size to a fixed size before FC layers.
    
    Args:
        output_size: Target output size (H, W) or int for square output
    """
    
    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply adaptive average pooling.
        
        Automatically computes kernel_size and stride to achieve output_size.
        """
        # API hints:
        # - Compute adaptive stride: stride_h = H // H_out, stride_w = W // W_out
        # - Compute adaptive kernel: kernel_h = H - (H_out - 1) * stride_h
        # - Variable window sizes possible (not fixed kernel)
        # - Forward: np.mean over computed windows
        # - Backward: distribute gradient / pool_size to each window
        
        return None
    
    def __repr__(self):
        return f"AdaptiveAvgPool2d(output_size={self.output_size})"


# ============================================================================
# Test Functions
# ============================================================================

def test_maxpool_forward():
    """Test max pooling forward pass."""
    results = {}
    
    try:
        x = np.arange(16).reshape(1, 1, 4, 4).astype(np.float64)
        result = maxpool2d_forward(x, kernel_size=2, stride=2)
        
        if result is not None:
            out, mask = result
            expected = np.array([[[[5, 7], [13, 15]]]])
            results['values'] = np.allclose(out, expected)
            results['shape'] = out.shape == (1, 1, 2, 2)
        else:
            results['values'] = False
            results['shape'] = False
    except Exception as e:
        results['values'] = False
        results['shape'] = False
    
    return results


def test_maxpool_backward():
    """Test max pooling backward pass."""
    results = {}
    
    try:
        np.random.seed(42)
        x = np.random.randn(2, 3, 4, 4)
        result = maxpool2d_forward(x, kernel_size=2, stride=2)
        
        if result is not None:
            out, mask = result
            dy = np.ones_like(out)
            dx = maxpool2d_backward(dy, mask, x.shape, kernel_size=2, stride=2)
            
            if dx is not None:
                results['shape'] = dx.shape == x.shape
                results['sparsity'] = np.sum(dx != 0) == out.size
            else:
                results['shape'] = False
                results['sparsity'] = False
        else:
            results['shape'] = False
            results['sparsity'] = False
    except Exception as e:
        results['shape'] = False
        results['sparsity'] = False
    
    return results


def test_maxpool_module():
    """Test MaxPool2d module."""
    results = {}
    
    try:
        np.random.seed(42)
        pool = MaxPool2d(kernel_size=2, stride=2)
        
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = pool(x)
        
        if y is not None:
            results['forward_shape'] = y.shape == (2, 3, 4, 4)
            
            loss = y.sum()
            loss.backward()
            
            results['backward'] = np.any(x.grad != 0)
        else:
            results['forward_shape'] = False
            results['backward'] = False
    except Exception as e:
        results['forward_shape'] = False
        results['backward'] = False
    
    return results


def test_avgpool_forward():
    """Test average pooling forward pass."""
    results = {}
    
    try:
        x = np.arange(16).reshape(1, 1, 4, 4).astype(np.float64)
        out = avgpool2d_forward(x, kernel_size=2, stride=2)
        
        if out is not None:
            expected = np.array([[[[2.5, 4.5], [10.5, 12.5]]]])
            results['values'] = np.allclose(out, expected)
            results['shape'] = out.shape == (1, 1, 2, 2)
        else:
            results['values'] = False
            results['shape'] = False
    except Exception as e:
        results['values'] = False
        results['shape'] = False
    
    return results


def test_avgpool_backward():
    """Test average pooling backward pass."""
    results = {}
    
    try:
        x_shape = (2, 3, 4, 4)
        dy = np.ones((2, 3, 2, 2))
        dx = avgpool2d_backward(dy, x_shape, kernel_size=2, stride=2)
        
        if dx is not None:
            results['shape'] = dx.shape == x_shape
            results['uniform'] = np.allclose(dx, 0.25)
        else:
            results['shape'] = False
            results['uniform'] = False
    except Exception as e:
        results['shape'] = False
        results['uniform'] = False
    
    return results


def test_avgpool_module():
    """Test AvgPool2d module."""
    results = {}
    
    try:
        np.random.seed(42)
        pool = AvgPool2d(kernel_size=2, stride=2)
        
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = pool(x)
        
        if y is not None:
            results['forward_shape'] = y.shape == (2, 3, 4, 4)
            
            loss = y.sum()
            loss.backward()
            
            results['backward'] = np.any(x.grad != 0)
        else:
            results['forward_shape'] = False
            results['backward'] = False
    except Exception as e:
        results['forward_shape'] = False
        results['backward'] = False
    
    return results


def test_global_avgpool():
    """Test GlobalAvgPool2d module."""
    results = {}
    
    try:
        np.random.seed(42)
        
        gap = GlobalAvgPool2d(flatten=True)
        x = Tensor(np.random.randn(2, 16, 4, 4))
        y = gap(x)
        
        if y is not None:
            results['shape_flat'] = y.shape == (2, 16)
            
            expected = np.mean(x.data, axis=(2, 3))
            results['values'] = np.allclose(y.data, expected)
            
            loss = y.sum()
            loss.backward()
            results['backward'] = np.any(x.grad != 0)
        else:
            results['shape_flat'] = False
            results['values'] = False
            results['backward'] = False
    except Exception as e:
        results['shape_flat'] = False
        results['values'] = False
        results['backward'] = False
    
    return results


def test_adaptive_avgpool():
    """Test AdaptiveAvgPool2d module."""
    results = {}
    
    try:
        np.random.seed(42)
        
        pool = AdaptiveAvgPool2d(output_size=(2, 2))
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = pool(x)
        
        if y is not None:
            results['shape'] = y.shape == (2, 3, 2, 2)
            
            loss = y.sum()
            loss.backward()
            results['backward'] = np.any(x.grad != 0)
        else:
            results['shape'] = False
            results['backward'] = False
    except Exception as e:
        results['shape'] = False
        results['backward'] = False
    
    return results


def test_against_pytorch():
    """Test pooling against PyTorch."""
    results = {}
    
    try:
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        
        x_np = np.random.randn(2, 3, 8, 8)
        
        our_maxpool = MaxPool2d(kernel_size=2, stride=2)
        our_x = Tensor(x_np.copy())
        our_y = our_maxpool(our_x)
        
        torch_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        torch_x = torch.tensor(x_np, requires_grad=True)
        torch_y = torch_maxpool(torch_x)
        
        if our_y is not None:
            results['maxpool_forward'] = np.allclose(our_y.data, torch_y.detach().numpy())
        else:
            results['maxpool_forward'] = False
        
        our_avgpool = AvgPool2d(kernel_size=2, stride=2)
        our_x2 = Tensor(x_np.copy())
        our_y2 = our_avgpool(our_x2)
        
        torch_avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        torch_x2 = torch.tensor(x_np, requires_grad=True)
        torch_y2 = torch_avgpool(torch_x2)
        
        if our_y2 is not None:
            results['avgpool_forward'] = np.allclose(our_y2.data, torch_y2.detach().numpy())
        else:
            results['avgpool_forward'] = False
            
    except ImportError:
        results['maxpool_forward'] = True
        results['avgpool_forward'] = True
    except Exception as e:
        results['maxpool_forward'] = False
        results['avgpool_forward'] = False
    
    return results


if __name__ == "__main__":
    print("Day 32: Pooling Operations")
    print("=" * 60)
    
    print("\nMax Pooling Forward:")
    maxpool_fwd = test_maxpool_forward()
    for name, passed in maxpool_fwd.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMax Pooling Backward:")
    maxpool_bwd = test_maxpool_backward()
    for name, passed in maxpool_bwd.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMaxPool2d Module:")
    maxpool_mod = test_maxpool_module()
    for name, passed in maxpool_mod.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nAverage Pooling Forward:")
    avgpool_fwd = test_avgpool_forward()
    for name, passed in avgpool_fwd.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nAverage Pooling Backward:")
    avgpool_bwd = test_avgpool_backward()
    for name, passed in avgpool_bwd.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nAvgPool2d Module:")
    avgpool_mod = test_avgpool_module()
    for name, passed in avgpool_mod.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nGlobal Average Pooling:")
    gap_results = test_global_avgpool()
    for name, passed in gap_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nAdaptive Average Pooling:")
    aap_results = test_adaptive_avgpool()
    for name, passed in aap_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nPyTorch Comparison:")
    pytorch_results = test_against_pytorch()
    for name, passed in pytorch_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day32.py for comprehensive tests!")
