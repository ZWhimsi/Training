"""
Day 18: Max and Min Operations
==============================
Estimated time: 2-3 hours
Prerequisites: Days 12-17 (Tensor class, reshape operations)

Learning objectives:
- Implement max/min operations with gradient flow
- Understand argmax tracking for gradient routing
- Handle ties in max/min (multiple maxima)
- Implement clamp/clip operations

Mathematical background:
========================
Max operation is NOT differentiable everywhere, but we define subgradients:

For y = max(x):
- Gradient flows ONLY to the maximum element(s)
- If x[i] == max(x), then dy/dx[i] = 1
- If x[i] < max(x), then dy/dx[i] = 0

When there are ties (multiple elements equal to max):
- Option 1: Split gradient equally among tied elements
- Option 2: Give gradient to first max only
- We use Option 1 for better gradient flow

Max along axis:
- max(X, axis=0): Column-wise max, gradient goes to max element per column
- max(X, axis=1): Row-wise max, gradient goes to max element per row

Clamp/Clip operation:
y = clamp(x, min_val, max_val)
- If x < min_val: y = min_val, dy/dx = 0
- If x > max_val: y = max_val, dy/dx = 0  
- Otherwise: y = x, dy/dx = 1

ReLU as a special case:
relu(x) = max(x, 0) = clamp(x, min=0, max=inf)
- dy/dx = 1 if x > 0, else 0
"""

import numpy as np
from typing import Tuple, Optional, Union


class Tensor:
    """Tensor class with max/min operations."""
    
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
        return f"Tensor(shape={self.shape}, data=\n{self.data})"
    
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
    
    @staticmethod
    def unbroadcast(grad, original_shape):
        """Reduce gradient to original shape."""
        while grad.ndim > len(original_shape):
            grad = grad.sum(axis=0)
        for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
            if orig_dim == 1 and grad_dim > 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    
    # Basic operations (provided)
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += Tensor.unbroadcast(out.grad, self.shape)
            other.grad += Tensor.unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += Tensor.unbroadcast(other.data * out.grad, self.shape)
            other.grad += Tensor.unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def sum(self, axis=None, keepdims=False):
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'sum')
        
        def _backward():
            if axis is None:
                self.grad += np.full(self.shape, out.grad)
            else:
                grad = out.grad
                if not keepdims:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                self.grad += np.broadcast_to(grad, self.shape).copy()
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 1: Max (all elements)
    # ========================================================================
    
    def max(self, axis=None, keepdims=False):
        """
        Find maximum value along specified axis.
        
        Args:
            axis: Axis along which to find max. None means all elements.
            keepdims: Keep reduced dimensions as size 1.
        
        Returns:
            Tensor with max values
        
        Gradient: Only flows to element(s) that achieved the maximum.
        If there are ties, gradient is split equally.
        """
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'max')
        
        # API hints:
        # - Create mask: (self.data == max_value).astype(float)
        # - Normalize mask for ties: mask / mask.sum(...)
        # - np.expand_dims(arr, axis=axis) -> add dim if not keepdims
        # - np.broadcast_to(grad, shape) -> broadcast gradient
        # - Gradient only flows to max position(s)
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 2: Min Operation
    # ========================================================================
    
    def min(self, axis=None, keepdims=False):
        """
        Find minimum value along specified axis.
        
        Args:
            axis: Axis along which to find min. None means all elements.
            keepdims: Keep reduced dimensions as size 1.
        
        Returns:
            Tensor with min values
        
        Gradient: Same as max, but for minimum elements.
        """
        result = np.min(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'min')
        
        # API hints:
        # - Same logic as max, but for minimum values
        # - Create mask: (self.data == min_value).astype(float)
        # - Normalize mask for ties
        # - Gradient only flows to min position(s)
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 3: Argmax (returns indices, no gradient)
    # ========================================================================
    
    def argmax(self, axis=None):
        """
        Return indices of maximum values.
        
        Note: This operation has no gradient (indices are discrete).
        
        Args:
            axis: Axis along which to find argmax. None flattens first.
        
        Returns:
            numpy array of indices (not a Tensor)
        """
        # API hints:
        # - np.argmax(data, axis=axis) -> indices of max values
        return None  # Replace
    
    def argmin(self, axis=None):
        """Return indices of minimum values."""
        # API hints:
        # - np.argmin(data, axis=axis) -> indices of min values
        return None  # Replace
    
    # ========================================================================
    # Exercise 4: Clamp/Clip Operation
    # ========================================================================
    
    def clamp(self, min_val=None, max_val=None):
        """
        Clamp values to be within [min_val, max_val].
        
        Args:
            min_val: Minimum value (None for no lower bound)
            max_val: Maximum value (None for no upper bound)
        
        Returns:
            Clamped tensor
        
        Gradient: 1 where value is in range, 0 where clamped.
        """
        # API hints:
        # - np.clip(data, min_val, max_val) -> clamp values
        # - Tensor(result, children, op) -> create output
        # - Gradient: 1 where not clamped, 0 where clamped
        # - (self.data >= min_val) -> boolean mask
        # - (self.data <= max_val) -> boolean mask
        
        # TODO: Implement forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # Alias for clamp
    def clip(self, min_val=None, max_val=None):
        return self.clamp(min_val, max_val)
    
    # ========================================================================
    # Exercise 5: ReLU Activation
    # ========================================================================
    
    def relu(self):
        """
        Rectified Linear Unit: relu(x) = max(x, 0)
        
        Returns:
            Tensor with ReLU applied
        
        Gradient: 1 if x > 0, 0 if x <= 0
        
        Note: At x=0, gradient is technically undefined.
        We use 0 (subgradient convention).
        """
        # API hints:
        # - np.maximum(data, 0) -> element-wise max with 0
        # - Tensor(result, children, op) -> create output
        # - Gradient: 1 if input > 0, else 0
        # - (self.data > 0) -> boolean mask for gradient
        
        # TODO: Implement forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 6: Leaky ReLU
    # ========================================================================
    
    def leaky_relu(self, negative_slope=0.01):
        """
        Leaky ReLU: allows small gradient when x < 0.
        
        f(x) = x if x > 0
        f(x) = negative_slope * x if x <= 0
        
        Args:
            negative_slope: Slope for negative values (default 0.01)
        
        Gradient:
            df/dx = 1 if x > 0
            df/dx = negative_slope if x <= 0
        """
        # API hints:
        # - np.where(condition, x, y) -> x where True, y where False
        # - Forward: x if x > 0 else negative_slope * x
        # - Gradient: 1 if x > 0 else negative_slope
        
        # TODO: Implement forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 7: Element-wise Maximum of Two Tensors
    # ========================================================================
    
    def maximum(self, other):
        """
        Element-wise maximum: z = max(self, other)
        
        Args:
            other: Tensor or scalar to compare with
        
        Returns:
            Tensor with element-wise maximum
        
        Gradient: 
            dz/d(self) = 1 where self >= other, else 0
            dz/d(other) = 1 where other > self, else 0
        """
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        
        # API hints:
        # - np.maximum(a, b) -> element-wise maximum
        # - Tensor(result, children, op) -> create output
        # - Gradient to self where self >= other
        # - Gradient to other where other > self
        # - Tensor.unbroadcast(grad, shape) -> handle broadcasting
        
        # TODO: Implement forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 8: Element-wise Minimum of Two Tensors
    # ========================================================================
    
    def minimum(self, other):
        """
        Element-wise minimum: z = min(self, other)
        
        Args:
            other: Tensor or scalar to compare with
        
        Returns:
            Tensor with element-wise minimum
        """
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        
        # API hints:
        # - np.minimum(a, b) -> element-wise minimum
        # - Tensor(result, children, op) -> create output
        # - Gradient to self where self <= other
        # - Gradient to other where other < self
        # - Tensor.unbroadcast(grad, shape) -> handle broadcasting
        
        # TODO: Implement forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out


# ============================================================================
# Test Functions
# ============================================================================

def test_max_all():
    """Test max of all elements."""
    results = {}
    
    x = Tensor([[1, 5, 3], [4, 2, 6]])
    y = x.max()
    
    if y is not None and y.data is not None:
        results['value'] = np.allclose(y.data, 6)
        y.backward()
        # Only position [1,2] (value 6) should have gradient
        expected_grad = np.array([[0, 0, 0], [0, 0, 1]])
        results['grad'] = np.allclose(x.grad, expected_grad) if x.grad is not None else False
    else:
        results['value'] = False
        results['grad'] = False
    
    return results


def test_max_axis():
    """Test max along axis."""
    results = {}
    
    x = Tensor([[1, 5, 3], [4, 2, 6]])
    
    # Max along axis 0 (column max)
    y0 = x.max(axis=0)
    if y0 is not None and y0.data is not None:
        results['axis0'] = np.allclose(y0.data, [4, 5, 6])
    else:
        results['axis0'] = False
    
    # Max along axis 1 (row max)
    x2 = Tensor([[1, 5, 3], [4, 2, 6]])
    y1 = x2.max(axis=1)
    if y1 is not None and y1.data is not None:
        results['axis1'] = np.allclose(y1.data, [5, 6])
    else:
        results['axis1'] = False
    
    return results


def test_max_gradient_axis():
    """Test max gradient along axis."""
    results = {}
    
    x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
    y = x.max(axis=1)  # Row max: [5, 6]
    y.backward()
    
    # Gradient should be 1 at max positions per row
    expected_grad = np.array([[0, 1, 0], [0, 0, 1]])
    
    if x.grad is not None:
        results['grad'] = np.allclose(x.grad, expected_grad)
    else:
        results['grad'] = False
    
    return results


def test_max_ties():
    """Test max with tied values."""
    results = {}
    
    x = Tensor([3.0, 3.0, 1.0, 3.0])  # Three max values
    y = x.max()
    y.backward()
    
    # Gradient should be split: 1/3 each to the tied maxima
    expected_grad = np.array([1/3, 1/3, 0, 1/3])
    
    if x.grad is not None:
        results['split_grad'] = np.allclose(x.grad, expected_grad)
    else:
        results['split_grad'] = False
    
    return results


def test_min():
    """Test min operation."""
    results = {}
    
    x = Tensor([[1, 5, 3], [4, 2, 6]])
    y = x.min()
    
    if y is not None and y.data is not None:
        results['value'] = np.allclose(y.data, 1)
        y.backward()
        expected_grad = np.array([[1, 0, 0], [0, 0, 0]])
        results['grad'] = np.allclose(x.grad, expected_grad) if x.grad is not None else False
    else:
        results['value'] = False
        results['grad'] = False
    
    return results


def test_clamp():
    """Test clamp operation."""
    results = {}
    
    x = Tensor([-2, -1, 0, 1, 2, 3])
    y = x.clamp(min_val=-1, max_val=2)
    
    if y is not None and y.data is not None:
        expected = np.array([-1, -1, 0, 1, 2, 2])
        results['values'] = np.allclose(y.data, expected)
        
        y.backward()
        # Gradient is 0 where clamped, 1 otherwise
        expected_grad = np.array([0, 1, 1, 1, 1, 0])
        results['grad'] = np.allclose(x.grad, expected_grad) if x.grad is not None else False
    else:
        results['values'] = False
        results['grad'] = False
    
    return results


def test_relu():
    """Test ReLU activation."""
    results = {}
    
    x = Tensor([-2, -1, 0, 1, 2])
    y = x.relu()
    
    if y is not None and y.data is not None:
        expected = np.array([0, 0, 0, 1, 2])
        results['values'] = np.allclose(y.data, expected)
        
        y.backward()
        expected_grad = np.array([0, 0, 0, 1, 1])
        results['grad'] = np.allclose(x.grad, expected_grad) if x.grad is not None else False
    else:
        results['values'] = False
        results['grad'] = False
    
    return results


def test_leaky_relu():
    """Test Leaky ReLU activation."""
    results = {}
    
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = x.leaky_relu(negative_slope=0.1)
    
    if y is not None and y.data is not None:
        expected = np.array([-0.2, -0.1, 0, 1, 2])
        results['values'] = np.allclose(y.data, expected)
        
        y.backward()
        expected_grad = np.array([0.1, 0.1, 0.1, 1, 1])
        results['grad'] = np.allclose(x.grad, expected_grad) if x.grad is not None else False
    else:
        results['values'] = False
        results['grad'] = False
    
    return results


def test_maximum_tensors():
    """Test element-wise maximum of two tensors."""
    results = {}
    
    a = Tensor([1, 4, 3])
    b = Tensor([2, 2, 5])
    c = a.maximum(b)
    
    if c is not None and c.data is not None:
        expected = np.array([2, 4, 5])
        results['values'] = np.allclose(c.data, expected)
        
        c.backward()
        results['grad_a'] = np.allclose(a.grad, [0, 1, 0]) if a.grad is not None else False
        results['grad_b'] = np.allclose(b.grad, [1, 0, 1]) if b.grad is not None else False
    else:
        results['values'] = False
        results['grad_a'] = False
        results['grad_b'] = False
    
    return results


def test_argmax():
    """Test argmax operation."""
    results = {}
    
    x = Tensor([[1, 5, 3], [4, 2, 6]])
    
    # Argmax all (flattened)
    idx_all = x.argmax()
    results['argmax_all'] = idx_all == 5 if idx_all is not None else False
    
    # Argmax axis 0
    idx_axis0 = x.argmax(axis=0)
    results['argmax_axis0'] = np.array_equal(idx_axis0, [1, 0, 1]) if idx_axis0 is not None else False
    
    # Argmax axis 1
    idx_axis1 = x.argmax(axis=1)
    results['argmax_axis1'] = np.array_equal(idx_axis1, [1, 2]) if idx_axis1 is not None else False
    
    return results


if __name__ == "__main__":
    print("Day 18: Max and Min Operations")
    print("=" * 60)
    
    print("\nMax All Elements:")
    max_all_results = test_max_all()
    for name, passed in max_all_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMax Along Axis:")
    max_axis_results = test_max_axis()
    for name, passed in max_axis_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMax Gradient Along Axis:")
    max_grad_results = test_max_gradient_axis()
    for name, passed in max_grad_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMax with Ties:")
    ties_results = test_max_ties()
    for name, passed in ties_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMin Operation:")
    min_results = test_min()
    for name, passed in min_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nClamp Operation:")
    clamp_results = test_clamp()
    for name, passed in clamp_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nReLU Activation:")
    relu_results = test_relu()
    for name, passed in relu_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLeaky ReLU:")
    leaky_results = test_leaky_relu()
    for name, passed in leaky_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nElement-wise Maximum:")
    max_elem_results = test_maximum_tensors()
    for name, passed in max_elem_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nArgmax:")
    argmax_results = test_argmax()
    for name, passed in argmax_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day18.py for comprehensive tests!")
