"""
Day 15: Sum and Mean Reduction Operations
=========================================
Estimated time: 2-3 hours
Prerequisites: Days 12-14 (Tensor class, broadcasting, operations)

Learning objectives:
- Implement sum reduction with gradient broadcasting
- Implement mean reduction with proper gradient scaling
- Understand keepdims and its effect on gradients
- Handle reduction along specific axes

Key concepts:
- Sum reduction: Collapses dimensions by summing
  - Gradient: Broadcasts the upstream gradient back to input shape
  
- Mean reduction: Sum divided by count
  - Gradient: Upstream gradient divided by count, broadcast to input shape
  
- keepdims: Keeps reduced dimensions as size 1
  - Important for proper broadcasting in neural networks

Examples:
- sum([1,2,3]) = 6, gradient is [1,1,1]
- mean([1,2,3]) = 2, gradient is [1/3, 1/3, 1/3]
- sum([[1,2],[3,4]], axis=0) = [4,6], gradient broadcasts rows
"""

import numpy as np
from typing import Tuple, Optional, Union


class Tensor:
    """Tensor class with reduction operations."""
    
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
    def size(self):
        return self.data.size
    
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
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data / other.data, (self, other), '/')
        
        def _backward():
            self.grad += Tensor.unbroadcast((1/other.data) * out.grad, self.shape)
            other.grad += Tensor.unbroadcast((-self.data/(other.data**2)) * out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __pow__(self, n):
        out = Tensor(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n-1)) * out.grad
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 1: Sum Reduction (all elements)
    # ========================================================================
    
    def sum(self, axis=None, keepdims=False):
        """
        Sum elements along specified axis(es).
        
        Args:
            axis: Axis or axes along which to sum.
                  None means sum all elements (returns scalar-shaped tensor).
            keepdims: If True, reduced axes are kept as dimensions of size 1.
        
        Returns:
            Tensor with sum result
        
        Gradient: The upstream gradient is broadcast back to input shape.
        For sum, each input element contributes equally (with coefficient 1).
        """
        # TODO: Implement forward pass
        # HINT: result = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = None  # Replace: Tensor(result, (self,), 'sum')
        
        # TODO: Implement backward pass
        def _backward():
            # Gradient needs to be broadcast back to self.shape
            # If keepdims=False, we need to expand dims to match
            # HINT: 
            # if axis is None:
            #     self.grad += np.full(self.shape, out.grad)
            # else:
            #     grad = out.grad
            #     if not keepdims:
            #         # Expand dims for broadcasting
            #         if isinstance(axis, int):
            #             grad = np.expand_dims(grad, axis=axis)
            #         else:
            #             for ax in sorted(axis):
            #                 grad = np.expand_dims(grad, axis=ax)
            #     self.grad += np.broadcast_to(grad, self.shape)
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 2: Mean Reduction
    # ========================================================================
    
    def mean(self, axis=None, keepdims=False):
        """
        Compute mean along specified axis(es).
        
        Args:
            axis: Axis or axes along which to compute mean.
                  None means mean of all elements.
            keepdims: If True, reduced axes are kept as dimensions of size 1.
        
        Returns:
            Tensor with mean result
        
        Gradient: Same as sum, but divided by the number of elements
        that were averaged.
        """
        # TODO: Implement forward pass
        # HINT: result = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = None  # Replace: Tensor(result, (self,), 'mean')
        
        # TODO: Compute the count of elements being averaged
        if axis is None:
            count = self.data.size
        elif isinstance(axis, int):
            count = self.data.shape[axis]
        else:
            count = np.prod([self.data.shape[ax] for ax in axis])
        
        # TODO: Implement backward pass
        def _backward():
            # Same as sum gradient, but divided by count
            # HINT: 
            # if axis is None:
            #     self.grad += np.full(self.shape, out.grad / count)
            # else:
            #     grad = out.grad / count
            #     if not keepdims:
            #         if isinstance(axis, int):
            #             grad = np.expand_dims(grad, axis=axis)
            #         else:
            #             for ax in sorted(axis):
            #                 grad = np.expand_dims(grad, axis=ax)
            #     self.grad += np.broadcast_to(grad, self.shape)
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 3: Max Reduction (bonus - more complex gradient)
    # ========================================================================
    
    def max(self, axis=None, keepdims=False):
        """
        Find maximum along specified axis(es).
        
        Args:
            axis: Axis or axes along which to find max.
            keepdims: If True, reduced axes are kept as dimensions of size 1.
        
        Returns:
            Tensor with max result
        
        Gradient: Only the maximum element(s) receive gradient.
        This is a "hard" selection operation.
        """
        # TODO: Implement forward pass
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        out = None  # Replace: Tensor(result, (self,), 'max')
        
        # TODO: Implement backward pass
        def _backward():
            # Create mask of where max values are
            # Gradient only flows to max positions
            # HINT:
            # if axis is None:
            #     mask = (self.data == out.data)
            # else:
            #     # Expand out.data for broadcasting comparison
            #     expanded = out.data if keepdims else np.expand_dims(out.data, axis=axis)
            #     mask = (self.data == expanded)
            # # Normalize mask if multiple max values
            # mask = mask / mask.sum(axis=axis, keepdims=True)
            # grad = out.grad if keepdims else np.expand_dims(out.grad, axis=axis)
            # self.grad += mask * np.broadcast_to(grad, self.shape)
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 4: Variance (uses mean and sum)
    # ========================================================================
    
    def var(self, axis=None, keepdims=False):
        """
        Compute variance along specified axis(es).
        
        var(x) = mean((x - mean(x))^2)
        
        This is implemented using existing operations, so gradients
        flow automatically through the computation graph!
        """
        # TODO: Implement variance using mean
        # HINT:
        # mean_val = self.mean(axis=axis, keepdims=True)  # Keep dims for broadcast
        # diff = self - mean_val
        # sq_diff = diff ** 2
        # return sq_diff.mean(axis=axis, keepdims=keepdims)
        return None  # Replace


# ============================================================================
# Exercise 5: Test Sum Reduction
# ============================================================================

def test_sum_all():
    """Test sum of all elements."""
    results = {}
    
    x = Tensor([[1, 2, 3], [4, 5, 6]])  # Sum = 21
    y = x.sum()
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, 21)
        y.backward()
        # Gradient should be all 1s
        results['grad'] = np.allclose(x.grad, np.ones((2, 3))) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


def test_sum_axis():
    """Test sum along specific axis."""
    results = {}
    
    # Sum along axis 0: column sums
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    y = x.sum(axis=0)
    
    if y is not None and y.data is not None:
        results['axis0_forward'] = np.allclose(y.data, [5, 7, 9])
        y.backward()
        results['axis0_grad'] = np.allclose(x.grad, np.ones((2, 3))) if x.grad is not None else False
    else:
        results['axis0_forward'] = False
        results['axis0_grad'] = False
    
    # Sum along axis 1: row sums
    x2 = Tensor([[1, 2, 3], [4, 5, 6]])
    y2 = x2.sum(axis=1)
    
    if y2 is not None and y2.data is not None:
        results['axis1_forward'] = np.allclose(y2.data, [6, 15])
        y2.backward()
        results['axis1_grad'] = np.allclose(x2.grad, np.ones((2, 3))) if x2.grad is not None else False
    else:
        results['axis1_forward'] = False
        results['axis1_grad'] = False
    
    return results


def test_sum_keepdims():
    """Test sum with keepdims=True."""
    results = {}
    
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    y = x.sum(axis=1, keepdims=True)
    
    if y is not None and y.data is not None:
        results['shape'] = y.shape == (2, 1)
        results['values'] = np.allclose(y.data, [[6], [15]])
    else:
        results['shape'] = False
        results['values'] = False
    
    return results


def test_mean_all():
    """Test mean of all elements."""
    results = {}
    
    x = Tensor([[1, 2, 3], [4, 5, 6]])  # Mean = 3.5
    y = x.mean()
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, 3.5)
        y.backward()
        # Gradient should be 1/6 for each element
        results['grad'] = np.allclose(x.grad, np.full((2, 3), 1/6)) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


def test_mean_axis():
    """Test mean along specific axis."""
    results = {}
    
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = x.mean(axis=0)  # Column means: [2.5, 3.5, 4.5]
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, [2.5, 3.5, 4.5])
        y.backward()
        # Gradient should be 1/2 (divided by number of rows)
        results['grad'] = np.allclose(x.grad, np.full((2, 3), 0.5)) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


def test_chain_rule_reduction():
    """Test chain rule with reductions."""
    results = {}
    
    # y = sum(x^2)
    # dy/dx = 2x
    x = Tensor([1.0, 2.0, 3.0])
    y = (x ** 2).sum()  # 1 + 4 + 9 = 14
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, 14)
        y.backward()
        results['grad'] = np.allclose(x.grad, [2, 4, 6]) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


def test_loss_function():
    """Test typical loss function (MSE)."""
    results = {}
    
    # MSE = mean((pred - target)^2)
    pred = Tensor([1.0, 2.0, 3.0])
    target = Tensor([1.5, 2.0, 2.5])
    
    diff = pred - target  # [-0.5, 0, 0.5]
    sq_diff = diff ** 2   # [0.25, 0, 0.25]
    mse = sq_diff.mean()  # 0.5 / 3 = 0.1667
    
    if mse is not None and mse.data is not None:
        results['forward'] = np.allclose(mse.data, 1/6)
        mse.backward()
        # d(MSE)/d(pred) = 2 * (pred - target) / n = 2 * [-0.5, 0, 0.5] / 3
        expected_grad = 2 * np.array([-0.5, 0, 0.5]) / 3
        results['grad'] = np.allclose(pred.grad, expected_grad) if pred.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


if __name__ == "__main__":
    print("Day 15: Sum and Mean Reductions")
    print("=" * 60)
    
    print("\nSum All Elements:")
    sum_all_results = test_sum_all()
    for name, passed in sum_all_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSum Along Axis:")
    sum_axis_results = test_sum_axis()
    for name, passed in sum_axis_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSum with keepdims:")
    keepdims_results = test_sum_keepdims()
    for name, passed in keepdims_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMean All Elements:")
    mean_all_results = test_mean_all()
    for name, passed in mean_all_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMean Along Axis:")
    mean_axis_results = test_mean_axis()
    for name, passed in mean_axis_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nChain Rule with Reduction:")
    chain_results = test_chain_rule_reduction()
    for name, passed in chain_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLoss Function (MSE):")
    loss_results = test_loss_function()
    for name, passed in loss_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day15.py for comprehensive tests!")
