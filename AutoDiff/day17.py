"""
Day 17: Reshape and View Operations
====================================
Estimated time: 2-3 hours
Prerequisites: Days 12-16 (Tensor class, matmul)

Learning objectives:
- Implement reshape with gradient tracking
- Understand view vs copy semantics
- Implement flatten and squeeze operations
- Handle unsqueeze for dimension expansion
- Learn about contiguous memory and its implications

Mathematical background:
========================
Reshape operations change the shape without changing the data:
- Forward: Simply reorganize memory layout
- Backward: Reshape gradient back to original shape

Key insight: Reshape is its own inverse for gradients!
If forward is: reshape(x, new_shape) 
Then backward is: reshape(grad, original_shape)

Important operations:
1. reshape(shape) - Arbitrary shape change (product must match)
2. flatten() - Collapse all dimensions to 1D
3. squeeze(axis) - Remove dimensions of size 1
4. unsqueeze(axis) - Add dimension of size 1
5. expand(shape) - Broadcast to larger shape (view only)

Memory layout matters:
- Row-major (C order): Last axis changes fastest
- Reshape preserves element order in memory
- Example: [[1,2],[3,4]] flattened = [1,2,3,4] (row by row)

Common uses in neural networks:
- Flattening conv outputs for fully connected layers
- Reshaping for attention (batch, seq, heads, dim)
- Broadcasting for efficient operations
"""

import numpy as np
from typing import Tuple, Optional, Union, List


class Tensor:
    """Tensor class with reshape operations."""
    
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
    # Exercise 1: Reshape Operation
    # ========================================================================
    
    def reshape(self, *shape):
        """
        Reshape tensor to new shape.
        
        Args:
            shape: New shape. One dimension can be -1 (inferred).
        
        Returns:
            Tensor with new shape
        
        Gradient: Reshape gradient back to original shape.
        
        Example:
            x = Tensor([[1,2,3],[4,5,6]])  # shape (2,3)
            y = x.reshape(3, 2)             # shape (3,2)
            y.data = [[1,2],[3,4],[5,6]]
        """
        # Handle tuple or multiple args
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        
        original_shape = self.shape
        
        # TODO: Implement forward pass
        # HINT: result = np.reshape(self.data, shape)
        out = None  # Replace: Tensor(result, (self,), 'reshape')
        
        # TODO: Implement backward pass
        def _backward():
            # Simply reshape gradient back to original shape
            # HINT: self.grad += np.reshape(out.grad, original_shape)
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 2: Flatten Operation
    # ========================================================================
    
    def flatten(self, start_dim=0, end_dim=-1):
        """
        Flatten dimensions from start_dim to end_dim into single dimension.
        
        Args:
            start_dim: First dimension to flatten (default: 0)
            end_dim: Last dimension to flatten (default: -1, meaning last)
        
        Returns:
            Flattened tensor
        
        Example:
            x.shape = (2, 3, 4)
            x.flatten().shape = (24,)
            x.flatten(start_dim=1).shape = (2, 12)
        """
        original_shape = self.shape
        
        # Handle negative indices
        if end_dim < 0:
            end_dim = self.ndim + end_dim
        if start_dim < 0:
            start_dim = self.ndim + start_dim
        
        # Calculate new shape
        new_shape = (
            list(original_shape[:start_dim]) +
            [np.prod(original_shape[start_dim:end_dim + 1])] +
            list(original_shape[end_dim + 1:])
        )
        
        # TODO: Implement forward pass
        # HINT: result = np.reshape(self.data, new_shape)
        out = None  # Replace: Tensor(result, (self,), 'flatten')
        
        # TODO: Implement backward pass
        def _backward():
            # Reshape gradient back to original shape
            # HINT: self.grad += np.reshape(out.grad, original_shape)
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 3: Squeeze Operation (Remove size-1 dimensions)
    # ========================================================================
    
    def squeeze(self, axis=None):
        """
        Remove dimensions of size 1.
        
        Args:
            axis: Specific axis to squeeze. If None, squeeze all size-1 dims.
        
        Returns:
            Squeezed tensor
        
        Gradient: Unsqueeze gradient back (add removed dimensions).
        
        Example:
            x.shape = (1, 3, 1, 4)
            x.squeeze().shape = (3, 4)
            x.squeeze(axis=0).shape = (3, 1, 4)
        """
        original_shape = self.shape
        
        # TODO: Implement forward pass
        # HINT: result = np.squeeze(self.data, axis=axis)
        out = None  # Replace: Tensor(result, (self,), 'squeeze')
        
        # TODO: Implement backward pass
        def _backward():
            # Reshape gradient back to include squeezed dimensions
            # HINT: self.grad += np.reshape(out.grad, original_shape)
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 4: Unsqueeze Operation (Add size-1 dimension)
    # ========================================================================
    
    def unsqueeze(self, axis):
        """
        Add a dimension of size 1 at the specified axis.
        
        Args:
            axis: Position for new dimension
        
        Returns:
            Tensor with added dimension
        
        Gradient: Remove the added dimension from gradient.
        
        Example:
            x.shape = (3, 4)
            x.unsqueeze(0).shape = (1, 3, 4)
            x.unsqueeze(1).shape = (3, 1, 4)
            x.unsqueeze(-1).shape = (3, 4, 1)
        """
        original_shape = self.shape
        
        # TODO: Implement forward pass
        # HINT: result = np.expand_dims(self.data, axis=axis)
        out = None  # Replace: Tensor(result, (self,), 'unsqueeze')
        
        # TODO: Implement backward pass
        def _backward():
            # Remove added dimension from gradient
            # HINT: self.grad += np.squeeze(out.grad, axis=axis)
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 5: View (alias for reshape with contiguity check)
    # ========================================================================
    
    def view(self, *shape):
        """
        View tensor with new shape (same as reshape in NumPy).
        
        In PyTorch, view requires contiguous memory. In NumPy/our implementation,
        reshape handles this automatically.
        
        Args:
            shape: New shape
        
        Returns:
            Tensor with new shape (view of same data)
        """
        # For simplicity, just use reshape
        return self.reshape(*shape)
    
    # ========================================================================
    # Exercise 6: Expand (Broadcast to larger shape)
    # ========================================================================
    
    def expand(self, *shape):
        """
        Expand tensor to larger shape by broadcasting.
        
        Can only expand dimensions of size 1 to larger size.
        -1 means keep the current size.
        
        Args:
            shape: Target shape
        
        Returns:
            Expanded tensor (broadcasts on the fly)
        
        Gradient: Sum over expanded dimensions.
        
        Example:
            x.shape = (1, 3)
            x.expand(4, 3).shape = (4, 3)  # Broadcasts first dim
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        
        # Handle -1 (keep dimension)
        final_shape = list(shape)
        for i, (s, orig) in enumerate(zip(final_shape, self.shape)):
            if s == -1:
                final_shape[i] = orig
        
        original_shape = self.shape
        
        # TODO: Implement forward pass
        # HINT: result = np.broadcast_to(self.data, final_shape)
        # Note: Need to copy because broadcast_to returns a view
        out = None  # Replace: Tensor(result.copy(), (self,), 'expand')
        
        # TODO: Implement backward pass
        def _backward():
            # Sum gradient over expanded dimensions
            # HINT: self.grad += Tensor.unbroadcast(out.grad, original_shape)
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 7: Permute (Generalized transpose)
    # ========================================================================
    
    def permute(self, *dims):
        """
        Permute tensor dimensions.
        
        Args:
            dims: New order of dimensions
        
        Returns:
            Permuted tensor
        
        Gradient: Permute gradient with inverse permutation.
        
        Example:
            x.shape = (2, 3, 4)
            x.permute(2, 0, 1).shape = (4, 2, 3)
        """
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        
        # Compute inverse permutation
        inv_dims = np.argsort(dims)
        
        # TODO: Implement forward pass
        # HINT: result = np.transpose(self.data, dims)
        out = None  # Replace: Tensor(result, (self,), 'permute')
        
        # TODO: Implement backward pass
        def _backward():
            # Permute gradient back with inverse permutation
            # HINT: self.grad += np.transpose(out.grad, inv_dims)
            pass  # Replace
        
        out._backward = _backward
        return out


# ============================================================================
# Test Functions
# ============================================================================

def test_reshape_basic():
    """Test basic reshape."""
    results = {}
    
    x = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    y = x.reshape(3, 2)
    
    if y is not None and y.data is not None:
        results['shape'] = y.shape == (3, 2)
        expected = np.array([[1, 2], [3, 4], [5, 6]])
        results['values'] = np.allclose(y.data, expected)
    else:
        results['shape'] = False
        results['values'] = False
    
    return results


def test_reshape_gradient():
    """Test reshape gradient."""
    results = {}
    
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    y = x.reshape(6)  # (6,)
    loss = y.sum()
    
    if loss is not None:
        loss.backward()
        results['grad_shape'] = x.grad.shape == (2, 3) if x.grad is not None else False
        results['grad_values'] = np.allclose(x.grad, np.ones((2, 3))) if x.grad is not None else False
    else:
        results['grad_shape'] = False
        results['grad_values'] = False
    
    return results


def test_flatten():
    """Test flatten operation."""
    results = {}
    
    x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
    y = x.flatten()
    
    if y is not None and y.data is not None:
        results['shape'] = y.shape == (8,)
        results['values'] = np.allclose(y.data, [1, 2, 3, 4, 5, 6, 7, 8])
        
        y.backward()
        results['grad_shape'] = x.grad.shape == (2, 2, 2) if x.grad is not None else False
    else:
        results['shape'] = False
        results['values'] = False
        results['grad_shape'] = False
    
    return results


def test_flatten_partial():
    """Test partial flatten (keeping batch dim)."""
    results = {}
    
    x = Tensor(np.arange(24).reshape(2, 3, 4))  # (2, 3, 4)
    y = x.flatten(start_dim=1)  # (2, 12)
    
    if y is not None and y.data is not None:
        results['shape'] = y.shape == (2, 12)
        
        y.backward()
        results['grad_shape'] = x.grad.shape == (2, 3, 4) if x.grad is not None else False
    else:
        results['shape'] = False
        results['grad_shape'] = False
    
    return results


def test_squeeze():
    """Test squeeze operation."""
    results = {}
    
    x = Tensor(np.arange(6).reshape(1, 2, 1, 3))  # (1, 2, 1, 3)
    y = x.squeeze()
    
    if y is not None and y.data is not None:
        results['shape'] = y.shape == (2, 3)
        
        y.backward()
        results['grad_shape'] = x.grad.shape == (1, 2, 1, 3) if x.grad is not None else False
    else:
        results['shape'] = False
        results['grad_shape'] = False
    
    return results


def test_squeeze_axis():
    """Test squeeze with specific axis."""
    results = {}
    
    x = Tensor(np.ones((1, 3, 1, 4)))
    y = x.squeeze(axis=0)  # Remove first dim only
    
    if y is not None and y.data is not None:
        results['shape'] = y.shape == (3, 1, 4)
    else:
        results['shape'] = False
    
    return results


def test_unsqueeze():
    """Test unsqueeze operation."""
    results = {}
    
    x = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    
    y0 = x.unsqueeze(0)
    y1 = x.unsqueeze(1)
    yn1 = x.unsqueeze(-1)
    
    if y0 is not None and y1 is not None and yn1 is not None:
        results['unsqueeze_0'] = y0.shape == (1, 2, 3) if y0.data is not None else False
        results['unsqueeze_1'] = y1.shape == (2, 1, 3) if y1.data is not None else False
        results['unsqueeze_-1'] = yn1.shape == (2, 3, 1) if yn1.data is not None else False
    else:
        results['unsqueeze_0'] = False
        results['unsqueeze_1'] = False
        results['unsqueeze_-1'] = False
    
    return results


def test_unsqueeze_gradient():
    """Test unsqueeze gradient."""
    results = {}
    
    x = Tensor([1.0, 2.0, 3.0])  # (3,)
    y = x.unsqueeze(0)  # (1, 3)
    loss = y.sum()
    
    if loss is not None:
        loss.backward()
        results['grad_shape'] = x.grad.shape == (3,) if x.grad is not None else False
        results['grad_values'] = np.allclose(x.grad, [1, 1, 1]) if x.grad is not None else False
    else:
        results['grad_shape'] = False
        results['grad_values'] = False
    
    return results


def test_expand():
    """Test expand operation."""
    results = {}
    
    x = Tensor([[1], [2], [3]])  # (3, 1)
    y = x.expand(3, 4)  # (3, 4)
    
    if y is not None and y.data is not None:
        results['shape'] = y.shape == (3, 4)
        expected = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
        results['values'] = np.allclose(y.data, expected)
    else:
        results['shape'] = False
        results['values'] = False
    
    return results


def test_expand_gradient():
    """Test expand gradient."""
    results = {}
    
    x = Tensor([[1.0], [2.0]])  # (2, 1)
    y = x.expand(2, 3)  # (2, 3)
    loss = y.sum()
    
    if loss is not None:
        loss.backward()
        # Gradient should be summed over expanded dimension
        results['grad_shape'] = x.grad.shape == (2, 1) if x.grad is not None else False
        # Each row was expanded to 3 elements, so gradient is 3
        results['grad_values'] = np.allclose(x.grad, [[3], [3]]) if x.grad is not None else False
    else:
        results['grad_shape'] = False
        results['grad_values'] = False
    
    return results


def test_permute():
    """Test permute operation."""
    results = {}
    
    x = Tensor(np.arange(24).reshape(2, 3, 4))  # (2, 3, 4)
    y = x.permute(2, 0, 1)  # (4, 2, 3)
    
    if y is not None and y.data is not None:
        results['shape'] = y.shape == (4, 2, 3)
        
        y.backward()
        results['grad_shape'] = x.grad.shape == (2, 3, 4) if x.grad is not None else False
    else:
        results['shape'] = False
        results['grad_shape'] = False
    
    return results


def test_reshape_infer():
    """Test reshape with -1 (inferred dimension)."""
    results = {}
    
    x = Tensor(np.arange(12))  # (12,)
    y = x.reshape(3, -1)  # Should infer (3, 4)
    
    if y is not None and y.data is not None:
        results['shape'] = y.shape == (3, 4)
    else:
        results['shape'] = False
    
    return results


def test_chain_reshape():
    """Test chained reshape operations."""
    results = {}
    
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    y = x.reshape(4).reshape(1, 4).reshape(2, 2)  # Back to (2, 2)
    loss = y.sum()
    
    if loss is not None:
        loss.backward()
        results['forward'] = np.allclose(y.data, x.data)
        results['grad'] = np.allclose(x.grad, np.ones((2, 2))) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


if __name__ == "__main__":
    print("Day 17: Reshape and View Operations")
    print("=" * 60)
    
    print("\nReshape Basic:")
    reshape_results = test_reshape_basic()
    for name, passed in reshape_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nReshape Gradient:")
    reshape_grad = test_reshape_gradient()
    for name, passed in reshape_grad.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nFlatten:")
    flatten_results = test_flatten()
    for name, passed in flatten_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nPartial Flatten:")
    partial_results = test_flatten_partial()
    for name, passed in partial_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSqueeze:")
    squeeze_results = test_squeeze()
    for name, passed in squeeze_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nUnsqueeze:")
    unsqueeze_results = test_unsqueeze()
    for name, passed in unsqueeze_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nExpand:")
    expand_results = test_expand()
    for name, passed in expand_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nPermute:")
    permute_results = test_permute()
    for name, passed in permute_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day17.py for comprehensive tests!")
