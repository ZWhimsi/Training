"""
Day 13: Broadcasting Rules for Tensors
======================================
Estimated time: 2-3 hours
Prerequisites: Day 12 (Tensor class basics)

Learning objectives:
- Understand NumPy-style broadcasting rules
- Implement broadcasting for tensors of different shapes
- Handle gradient computation with broadcasting
- Understand how to unbroadcast gradients

Key concepts:
Broadcasting allows operations between tensors of different shapes:
1. Dimensions are compared right-to-left
2. Dimensions are compatible if equal or one of them is 1
3. Missing dimensions (on the left) are treated as 1

Examples:
- (3,) + (1,) -> (3,)          # scalar broadcast
- (2, 3) + (3,) -> (2, 3)      # row broadcast  
- (2, 3) + (2, 1) -> (2, 3)    # column broadcast
- (2, 1, 3) + (4, 3) -> (2, 4, 3)  # general broadcast

Gradient unbroadcasting:
When computing gradients through broadcasted operations,
we must sum along the broadcasted dimensions to get the correct gradient shape.
"""

import numpy as np
from typing import Tuple, List


class Tensor:
    """Tensor class with broadcasting support."""
    
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
    
    # ========================================================================
    # Exercise 1: Check Broadcasting Compatibility
    # ========================================================================
    
    @staticmethod
    def broadcast_compatible(shape1: Tuple, shape2: Tuple) -> bool:
        """
        Check if two shapes are broadcast compatible.
        
        Rules:
        - Compare dimensions right-to-left
        - Dimensions must be equal OR one must be 1
        - Missing dimensions are treated as 1
        
        Args:
            shape1: First shape tuple
            shape2: Second shape tuple
        
        Returns:
            True if shapes can be broadcast together
        """
        # TODO: Implement broadcast compatibility check
        # HINT: Iterate from right to left, checking each dimension pair
        
        # Pad shorter shape with 1s on the left
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)
        
        # Right-align the shapes
        shape1_padded = (1,) * (max_len - len1) + shape1
        shape2_padded = (1,) * (max_len - len2) + shape2
        
        # API hints:
        # - Compatible if dims equal OR one is 1
        # - Return False immediately if incompatible found
        # - Return True if all dimensions compatible
        
        # TODO: Check each dimension pair
        for d1, d2 in zip(shape1_padded, shape2_padded):
            pass  # Replace with check
        
        return None  # Replace
    
    # ========================================================================
    # Exercise 2: Compute Broadcast Shape
    # ========================================================================
    
    @staticmethod
    def broadcast_shape(shape1: Tuple, shape2: Tuple) -> Tuple:
        """
        Compute the resulting shape after broadcasting.
        
        Args:
            shape1: First shape tuple
            shape2: Second shape tuple
        
        Returns:
            Resulting broadcast shape, or None if not compatible
        """
        if not Tensor.broadcast_compatible(shape1, shape2):
            return None
        
        # TODO: Compute the broadcast result shape
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)
        
        shape1_padded = (1,) * (max_len - len1) + shape1
        shape2_padded = (1,) * (max_len - len2) + shape2
        
        # API hints:
        # - max(d1, d2) -> resulting dimension
        # - tuple(list) -> convert list to tuple
        
        result = []
        for d1, d2 in zip(shape1_padded, shape2_padded):
            pass  # Replace
        
        return None  # Replace
    
    # ========================================================================
    # Exercise 3: Unbroadcast Gradient
    # ========================================================================
    
    @staticmethod
    def unbroadcast(grad: np.ndarray, original_shape: Tuple) -> np.ndarray:
        """
        Reduce gradient to original shape by summing along broadcast dimensions.
        
        This is crucial for correct gradient computation!
        
        When a tensor is broadcast, the same value participates in multiple
        computations. The gradient must be summed over these repeated uses.
        
        Args:
            grad: Gradient with broadcast shape
            original_shape: Shape to reduce to
        
        Returns:
            Gradient with original_shape
        """
        # API hints:
        # - grad.sum(axis=0) -> sum along first axis (removes leading dim)
        # - grad.sum(axis=i, keepdims=True) -> sum but keep dimension
        # - grad.ndim -> number of dimensions
        # - Sum along axes that were broadcast (size 1 -> larger)
        
        # TODO: Handle added leading dimensions
        while grad.ndim > len(original_shape):
            pass  # Replace
        
        # TODO: Handle broadcast dimensions (where original was 1)
        for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
            if orig_dim == 1 and grad_dim > 1:
                pass  # Replace
        
        return None  # Replace
    
    # ========================================================================
    # Exercise 4: Addition with Broadcasting
    # ========================================================================
    
    def __add__(self, other):
        """
        Element-wise addition with broadcasting support.
        """
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        
        # Forward: NumPy handles broadcasting automatically
        out = Tensor(self.data + other.data, (self, other), '+')
        
        # API hints:
        # - Tensor.unbroadcast(grad, shape) -> reduce grad to original shape
        # - self.grad += ... -> accumulate gradient
        
        # Backward: Must unbroadcast gradients
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    # ========================================================================
    # Exercise 5: Multiplication with Broadcasting
    # ========================================================================
    
    def __mul__(self, other):
        """
        Element-wise multiplication with broadcasting support.
        """
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        
        out = Tensor(self.data * other.data, (self, other), '*')
        
        # API hints:
        # - d(a*b)/da = b, d(a*b)/db = a
        # - Multiply local gradient by upstream gradient
        # - Tensor.unbroadcast(grad, shape) -> reduce to original shape
        
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other


# ============================================================================
# Exercise 6: Test Broadcasting Rules
# ============================================================================

def test_broadcast_compatibility():
    """Test broadcast compatibility checks."""
    test_cases = [
        ((3,), (1,), True),           # scalar broadcast
        ((2, 3), (3,), True),         # row broadcast
        ((2, 3), (2, 1), True),       # column broadcast
        ((2, 1, 3), (4, 3), True),    # general broadcast
        ((2, 3), (4,), False),        # incompatible
        ((2, 3), (2, 4), False),      # incompatible
        ((3, 4, 5), (4, 1), True),    # 3D broadcast
        ((1, 5), (3, 1), True),       # cross broadcast
    ]
    
    results = {}
    for shape1, shape2, expected in test_cases:
        actual = Tensor.broadcast_compatible(shape1, shape2)
        results[f"{shape1} + {shape2}"] = actual == expected if actual is not None else False
    
    return results


def test_broadcast_shape():
    """Test broadcast shape computation."""
    test_cases = [
        ((3,), (1,), (3,)),
        ((2, 3), (3,), (2, 3)),
        ((2, 3), (2, 1), (2, 3)),
        ((2, 1, 3), (4, 3), (2, 4, 3)),
        ((1, 5), (3, 1), (3, 5)),
    ]
    
    results = {}
    for shape1, shape2, expected in test_cases:
        actual = Tensor.broadcast_shape(shape1, shape2)
        results[f"{shape1} + {shape2}"] = actual == expected if actual else False
    
    return results


def test_broadcast_forward():
    """Test forward pass with broadcasting."""
    results = {}
    
    # Row broadcast: (2, 3) + (3,)
    a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = Tensor([10, 20, 30])            # (3,)
    c = a + b
    expected = np.array([[11, 22, 33], [14, 25, 36]])
    results['row_broadcast'] = np.allclose(c.data, expected)
    
    # Column broadcast: (2, 3) + (2, 1)
    d = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    e = Tensor([[10], [20]])            # (2, 1)
    f = d + e
    expected = np.array([[11, 12, 13], [24, 25, 26]])
    results['column_broadcast'] = np.allclose(f.data, expected)
    
    # Scalar broadcast
    g = Tensor([1, 2, 3])
    h = g * 2
    results['scalar_broadcast'] = np.allclose(h.data, [2, 4, 6])
    
    return results


# ============================================================================
# Exercise 7: Test Gradient Unbroadcasting
# ============================================================================

def test_unbroadcast():
    """Test gradient unbroadcasting."""
    results = {}
    
    # Test: sum reduction for added dimension
    grad = np.ones((2, 3))
    orig_shape = (3,)
    unbroadcasted = Tensor.unbroadcast(grad, orig_shape)
    if unbroadcasted is not None:
        results['sum_added_dim'] = unbroadcasted.shape == (3,) and np.allclose(unbroadcasted, [2, 2, 2])
    else:
        results['sum_added_dim'] = False
    
    # Test: sum reduction for broadcast 1
    grad2 = np.ones((2, 3))
    orig_shape2 = (2, 1)
    unbroadcasted2 = Tensor.unbroadcast(grad2, orig_shape2)
    if unbroadcasted2 is not None:
        results['sum_broadcast_1'] = unbroadcasted2.shape == (2, 1) and np.allclose(unbroadcasted2, [[3], [3]])
    else:
        results['sum_broadcast_1'] = False
    
    return results


def test_broadcast_backward():
    """Test backward pass with broadcasting."""
    results = {}
    
    # Test: (2, 3) + (3,) - gradient should sum along axis 0
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    b = Tensor([1.0, 1.0, 1.0])                      # (3,)
    c = a + b
    c.backward()
    
    # da should be (2, 3) of ones
    # db should be (3,) summed from (2, 3) = [2, 2, 2]
    if a.grad is not None and b.grad is not None:
        results['add_grad_a'] = np.allclose(a.grad, np.ones((2, 3)))
        results['add_grad_b'] = np.allclose(b.grad, [2, 2, 2])
    else:
        results['add_grad_a'] = False
        results['add_grad_b'] = False
    
    # Test: (2, 3) * (2, 1) - multiplication gradient
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    y = Tensor([[2.0], [3.0]])                      # (2, 1)
    z = x * y
    z.backward()
    
    # dz/dx = y broadcast -> [[2,2,2], [3,3,3]]
    # dz/dy = x summed along axis 1 -> [[6], [15]]
    if x.grad is not None and y.grad is not None:
        expected_x_grad = np.array([[2, 2, 2], [3, 3, 3]])
        expected_y_grad = np.array([[6], [15]])
        results['mul_grad_x'] = np.allclose(x.grad, expected_x_grad)
        results['mul_grad_y'] = np.allclose(y.grad, expected_y_grad)
    else:
        results['mul_grad_x'] = False
        results['mul_grad_y'] = False
    
    return results


if __name__ == "__main__":
    print("Day 13: Broadcasting Rules")
    print("=" * 50)
    
    print("\nBroadcast Compatibility Tests:")
    compat_results = test_broadcast_compatibility()
    for name, passed in compat_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nBroadcast Shape Tests:")
    shape_results = test_broadcast_shape()
    for name, passed in shape_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nBroadcast Forward Tests:")
    forward_results = test_broadcast_forward()
    for name, passed in forward_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nUnbroadcast Tests:")
    unbroadcast_results = test_unbroadcast()
    for name, passed in unbroadcast_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nBroadcast Backward Tests:")
    backward_results = test_broadcast_backward()
    for name, passed in backward_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day13.py for comprehensive tests!")
