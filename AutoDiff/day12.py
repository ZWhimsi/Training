"""
Day 12: Tensor Class - Multi-dimensional Autodiff
=================================================
Estimated time: 2-3 hours
Prerequisites: Days 8-10 (Value class, backward passes)

Learning objectives:
- Transition from scalar Value to multi-dimensional Tensor
- Understand tensor shape and data layout
- Implement basic tensor operations
- Prepare foundation for neural network computations

Key concepts:
- Shape: The dimensions of a tensor (e.g., (2, 3) = 2 rows, 3 cols)
- Data: Stored as flat list, accessed via index computation
- Gradients: Same shape as data, accumulated during backward pass

This builds on the Value class concepts but extends to arrays.
"""

import numpy as np
from typing import Tuple, List, Union, Optional


class Tensor:
    """
    Multi-dimensional tensor with automatic differentiation support.
    
    Data is stored as a numpy array for efficient computation.
    """
    
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        """
        Initialize a Tensor.
        
        Args:
            data: Input data (list, numpy array, or scalar)
            _children: Parent tensors in computation graph
            _op: Operation that created this tensor
            requires_grad: Whether to track gradients
        """
        # API hints:
        # - np.array(data, dtype=np.float64) -> convert to numpy array
        # - np.zeros_like(arr) -> create zeros array with same shape
        
        # TODO: Convert data to numpy array
        self.data = None  # Replace
        
        # TODO: Initialize gradient to zeros with same shape
        self.grad = None  # Replace
        
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = requires_grad
    
    # ========================================================================
    # Exercise 1: Shape and Size Properties
    # ========================================================================
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        # API hints:
        # - self.data.shape -> numpy array shape attribute
        return None  # Replace
    
    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        # API hints:
        # - self.data.ndim -> number of dimensions
        return None  # Replace
    
    @property
    def size(self) -> int:
        """Return total number of elements."""
        # API hints:
        # - self.data.size -> total element count
        return None  # Replace
    
    # ========================================================================
    # Exercise 2: String Representation
    # ========================================================================
    
    def __repr__(self):
        """String representation of the tensor."""
        # API hints:
        # - f-string with self.shape and self.data
        return None  # Replace
    
    # ========================================================================
    # Exercise 3: Backward Pass
    # ========================================================================
    
    def backward(self):
        """
        Compute gradients for all tensors in the computation graph.
        
        Same algorithm as Value.backward(), but:
        - Initialize gradient with ones of same shape
        - Works with numpy arrays instead of scalars
        """
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # API hints:
        # - np.ones_like(self.data) -> array of ones with same shape
        # - reversed(topo) -> iterate in reverse order
        # - v._backward() -> call backward function
        
        # TODO: Initialize gradient of output
        pass  # Replace
        
        # TODO: Call backward on each node in reverse order
        pass  # Replace
    
    # ========================================================================
    # Exercise 4: Element Access
    # ========================================================================
    
    def __getitem__(self, idx):
        """
        Get element or slice of tensor.
        
        Note: This is a simplified version without gradient tracking
        for the indexed operation.
        """
        # API hints:
        # - self.data[idx] -> index into numpy array
        return None  # Replace
    
    def item(self):
        """
        Return the tensor as a Python scalar (only for single-element tensors).
        """
        # API hints:
        # - self.data.item() -> convert single-element array to scalar
        return None  # Replace
    
    # ========================================================================
    # Exercise 5: Tensor Creation Methods
    # ========================================================================
    
    @staticmethod
    def zeros(shape, requires_grad=True):
        """Create a tensor filled with zeros."""
        # API hints:
        # - np.zeros(shape) -> array of zeros
        # - Tensor(data, requires_grad=...) -> create tensor
        return None  # Replace
    
    @staticmethod
    def ones(shape, requires_grad=True):
        """Create a tensor filled with ones."""
        # API hints:
        # - np.ones(shape) -> array of ones
        return None  # Replace
    
    @staticmethod
    def randn(shape, requires_grad=True):
        """Create a tensor with random normal values."""
        # API hints:
        # - np.random.randn(*shape) -> random normal array
        return None  # Replace
    
    @staticmethod
    def from_numpy(arr, requires_grad=True):
        """Create a tensor from a numpy array."""
        # API hints:
        # - arr.copy() -> copy array to avoid aliasing
        return None  # Replace
    
    # ========================================================================
    # Exercise 6: Basic Operations (scalar)
    # ========================================================================
    
    def __add__(self, other):
        """
        Element-wise addition.
        
        Supports Tensor + Tensor and Tensor + scalar.
        Gradient: dz/d(self) = 1, dz/d(other) = 1
        """
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        
        # API hints:
        # - self.data + other.data -> element-wise addition
        # - Tensor(data, children, op) -> create output tensor
        # - self.grad += out.grad -> accumulate gradient (addition gradient is 1)
        
        # TODO: Implement forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        """
        Element-wise multiplication.
        
        Supports Tensor * Tensor and Tensor * scalar.
        Gradient: dz/d(self) = other, dz/d(other) = self
        """
        if isinstance(other, (int, float)):
            other = Tensor(np.full_like(self.data, other))
        
        # API hints:
        # - self.data * other.data -> element-wise multiplication
        # - Tensor(data, children, op) -> create output tensor
        # - d(a*b)/da = b, d(a*b)/db = a
        # - self.grad += local_grad * out.grad
        
        # TODO: Implement forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        """Negation: -tensor"""
        return self * -1


# ============================================================================
# Exercise 7: Verify Basic Operations
# ============================================================================

def test_tensor_creation():
    """Test tensor creation and basic properties."""
    # TODO: Test different creation methods
    results = {}
    
    # Create from list
    t1 = Tensor([[1, 2], [3, 4]])
    results['from_list'] = t1.shape == (2, 2) if t1.shape else False
    
    # Create zeros
    t2 = Tensor.zeros((3, 4))
    results['zeros'] = t2.shape == (3, 4) if t2 and t2.shape else False
    
    # Create ones
    t3 = Tensor.ones((2, 3))
    if t3 is not None and t3.data is not None:
        results['ones'] = np.allclose(t3.data, 1.0)
    else:
        results['ones'] = False
    
    return results


def test_tensor_operations():
    """Test basic tensor operations."""
    results = {}
    
    # Addition
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])
    c = a + b
    if c is not None and c.data is not None:
        results['add'] = np.allclose(c.data, [[6, 8], [10, 12]])
    else:
        results['add'] = False
    
    # Multiplication
    d = a * b
    if d is not None and d.data is not None:
        results['mul'] = np.allclose(d.data, [[5, 12], [21, 32]])
    else:
        results['mul'] = False
    
    # Scalar operations
    e = a * 2
    if e is not None and e.data is not None:
        results['scalar_mul'] = np.allclose(e.data, [[2, 4], [6, 8]])
    else:
        results['scalar_mul'] = False
    
    return results


def test_tensor_backward():
    """Test gradient computation."""
    results = {}
    
    # Simple addition backward
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    if c is not None:
        c.backward()
        
        if a.grad is not None:
            results['add_grad_a'] = np.allclose(a.grad, [1, 1, 1])
        else:
            results['add_grad_a'] = False
        
        if b.grad is not None:
            results['add_grad_b'] = np.allclose(b.grad, [1, 1, 1])
        else:
            results['add_grad_b'] = False
    else:
        results['add_grad_a'] = False
        results['add_grad_b'] = False
    
    # Multiplication backward
    x = Tensor([2.0, 3.0])
    y = Tensor([4.0, 5.0])
    z = x * y
    if z is not None:
        z.backward()
        
        # dz/dx = y, dz/dy = x
        if x.grad is not None:
            results['mul_grad_x'] = np.allclose(x.grad, [4, 5])
        else:
            results['mul_grad_x'] = False
        
        if y.grad is not None:
            results['mul_grad_y'] = np.allclose(y.grad, [2, 3])
        else:
            results['mul_grad_y'] = False
    else:
        results['mul_grad_x'] = False
        results['mul_grad_y'] = False
    
    return results


if __name__ == "__main__":
    print("Day 12: Tensor Class")
    print("=" * 50)
    
    print("\nTensor Creation Tests:")
    creation_results = test_tensor_creation()
    for name, passed in creation_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nTensor Operation Tests:")
    op_results = test_tensor_operations()
    for name, passed in op_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nTensor Backward Tests:")
    back_results = test_tensor_backward()
    for name, passed in back_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day12.py for comprehensive tests!")
