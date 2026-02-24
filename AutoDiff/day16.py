"""
Day 16: Matrix Multiplication Gradient
=======================================
Estimated time: 2-3 hours
Prerequisites: Days 12-15 (Tensor class, broadcasting, reductions)

Learning objectives:
- Implement matrix multiplication forward pass
- Derive and implement matmul gradient correctly
- Handle batch matrix multiplication
- Understand the relationship between matmul and its transpose

Mathematical background:
========================
Matrix multiplication: C = A @ B where A: (m, k), B: (k, n) -> C: (m, n)

Gradient derivation:
- dL/dA = dL/dC @ B.T   (shape: (m,n) @ (n,k) = (m,k) ✓)
- dL/dB = A.T @ dL/dC   (shape: (k,m) @ (m,n) = (k,n) ✓)

Intuition:
- Each element A[i,j] contributes to all C[i,:] with weights B[j,:]
- Each element B[i,j] contributes to all C[:,j] with weights A[:,i]

Batch matmul: (..., m, k) @ (..., k, n) -> (..., m, n)
- Gradients computed similarly for each batch

Examples:
- A = [[1,2],[3,4]] (2x2), B = [[5,6],[7,8]] (2x2)
- C = A @ B = [[19,22],[43,50]]
- For loss L = sum(C), dL/dC = ones
- dL/dA = dL/dC @ B.T = [[11,15],[11,15]]
"""

import numpy as np
from typing import Tuple, Optional, Union


class Tensor:
    """Tensor class with matrix multiplication support."""
    
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
    def T(self):
        """Transpose of the tensor (last two axes)."""
        return self.transpose()
    
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
                self.grad += np.broadcast_to(grad, self.shape)
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 1: Transpose Operation
    # ========================================================================
    
    def transpose(self, axes=None):
        """
        Transpose tensor. By default, reverses all axes.
        For 2D tensor, swaps rows and columns.
        
        Args:
            axes: Optional permutation of axes. None reverses all axes.
        
        Returns:
            Transposed tensor
        
        Gradient: Transpose gradient with inverse permutation
        """
        if axes is None:
            result = np.transpose(self.data)
            inv_axes = None
        else:
            result = np.transpose(self.data, axes)
            inv_axes = np.argsort(axes)
        
        out = Tensor(result, (self,), 'T')
        
        # API hints:
        # - np.transpose(arr) -> reverse all axes
        # - np.transpose(arr, axes) -> permute with specific order
        # - Inverse permutation undoes the transpose
        # - np.argsort(axes) -> compute inverse permutation
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 2: Matrix Multiplication (matmul)
    # ========================================================================
    
    def matmul(self, other):
        """
        Matrix multiplication: C = self @ other
        
        For 2D tensors A (m,k) and B (k,n):
            C[i,j] = sum_l A[i,l] * B[l,j]
        
        Gradient:
            dL/dA = dL/dC @ B.T
            dL/dB = A.T @ dL/dC
        
        Args:
            other: Tensor to multiply with
        
        Returns:
            Result of matrix multiplication
        """
        if isinstance(other, (int, float)):
            raise ValueError("matmul requires tensor operand")
        
        # API hints:
        # - np.matmul(A, B) -> matrix multiplication
        # - Tensor(result, children, op) -> create output
        # - dL/dA = dL/dC @ B.T (gradient w.r.t. first operand)
        # - dL/dB = A.T @ dL/dC (gradient w.r.t. second operand)
        # - np.swapaxes(arr, -1, -2) -> transpose last two axes
        
        # TODO: Implement forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        """Overload @ operator for matrix multiplication."""
        return self.matmul(other)
    
    # ========================================================================
    # Exercise 3: Vector-Matrix Multiplication
    # ========================================================================
    
    def matvec(self, vec):
        """
        Matrix-vector multiplication: y = A @ x
        
        For A (m,n) and x (n,):
            y[i] = sum_j A[i,j] * x[j]
        
        Gradient:
            dL/dA = outer(dL/dy, x)  -> (m,n)
            dL/dx = A.T @ dL/dy      -> (n,)
        
        Args:
            vec: Vector tensor of shape (n,) or (n, 1)
        
        Returns:
            Result vector of shape (m,) or (m, 1)
        """
        # Determine if vec is column vector (n,1) or flat (n,)
        is_column = vec.data.ndim == 2 and vec.data.shape[1] == 1
        vec_flat = vec.data.flatten() if is_column else vec.data
        
        # API hints:
        # - np.dot(matrix, vector) -> matrix-vector product
        # - np.outer(a, b) -> outer product (m,) x (n,) -> (m, n)
        # - dL/dA = outer(dL/dy, x) -> (m,n) gradient for matrix
        # - dL/dx = A.T @ dL/dy -> (n,) gradient for vector
        # - arr.reshape(shape) -> reshape result
        
        # TODO: Implement forward pass
        result = None  # Replace
        if is_column:
            result = result.reshape(-1, 1) if result is not None else None
        
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 4: Batch Matrix Multiplication
    # ========================================================================
    
    def bmm(self, other):
        """
        Batch matrix multiplication: C = batched A @ B
        
        For tensors A (b, m, k) and B (b, k, n):
            C[i] = A[i] @ B[i] for each batch i
        
        Gradient: Same as regular matmul, applied per batch
        
        Args:
            other: Tensor of shape (batch, k, n)
        
        Returns:
            Result of shape (batch, m, n)
        """
        assert self.data.ndim == 3 and other.data.ndim == 3, \
            "bmm requires 3D tensors"
        
        # API hints:
        # - np.matmul handles batch dimensions automatically
        # - Same gradient formulas as 2D matmul
        # - dL/dA = dL/dC @ B.T, dL/dB = A.T @ dL/dC
        # - np.swapaxes(arr, -1, -2) -> transpose last two axes
        
        # TODO: Implement forward pass (same as matmul for 3D)
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out


# ============================================================================
# Exercise 5: Linear Layer Implementation
# ============================================================================

class Linear:
    """
    Linear layer: y = x @ W + b
    
    This is the fundamental building block of neural networks.
    """
    
    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize linear layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: Whether to include bias term
        """
        # Xavier initialization
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(np.random.randn(in_features, out_features) * scale)
        self.bias = Tensor(np.zeros(out_features)) if bias else None
    
    def __call__(self, x):
        """
        Forward pass: y = x @ W + b
        
        Args:
            x: Input tensor of shape (batch, in_features)
        
        Returns:
            Output tensor of shape (batch, out_features)
        """
        # API hints:
        # - x @ self.weight or x.matmul(self.weight) -> matrix multiply
        # - out + self.bias -> add bias (broadcasts over batch)
        # - Check if self.bias is not None before adding
        
        # TODO: Implement forward pass
        return None  # Replace
    
    def parameters(self):
        """Return list of parameters."""
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]


# ============================================================================
# Test Functions
# ============================================================================

def test_transpose():
    """Test transpose operation."""
    results = {}
    
    # 2D transpose
    A = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    AT = A.T
    
    if AT is not None and AT.data is not None:
        results['shape'] = AT.shape == (3, 2)
        results['values'] = np.allclose(AT.data, [[1, 4], [2, 5], [3, 6]])
        
        # Test gradient
        AT.backward()
        results['grad_shape'] = A.grad.shape == (2, 3) if A.grad is not None else False
    else:
        results['shape'] = False
        results['values'] = False
        results['grad_shape'] = False
    
    return results


def test_matmul_2d():
    """Test 2D matrix multiplication."""
    results = {}
    
    # A (2,3) @ B (3,2) = C (2,2)
    A = Tensor([[1, 2, 3], [4, 5, 6]])
    B = Tensor([[7, 8], [9, 10], [11, 12]])
    C = A @ B
    
    expected = np.array([[58, 64], [139, 154]])
    
    if C is not None and C.data is not None:
        results['shape'] = C.shape == (2, 2)
        results['values'] = np.allclose(C.data, expected)
    else:
        results['shape'] = False
        results['values'] = False
    
    return results


def test_matmul_gradient():
    """Test matrix multiplication gradient."""
    results = {}
    
    # Simple case: A (2,2) @ B (2,2)
    A = Tensor([[1.0, 2.0], [3.0, 4.0]])
    B = Tensor([[5.0, 6.0], [7.0, 8.0]])
    C = A @ B
    
    # Loss = sum(C)
    loss = C.sum()
    
    if loss is not None:
        loss.backward()
        
        # dL/dA = ones @ B.T
        # B.T = [[5,7],[6,8]], ones @ B.T = [[11,15],[11,15]]
        expected_grad_A = np.array([[11, 15], [11, 15]])
        
        # dL/dB = A.T @ ones
        # A.T = [[1,3],[2,4]], A.T @ ones = [[4,4],[6,6]]
        expected_grad_B = np.array([[4, 4], [6, 6]])
        
        results['grad_A'] = np.allclose(A.grad, expected_grad_A) if A.grad is not None else False
        results['grad_B'] = np.allclose(B.grad, expected_grad_B) if B.grad is not None else False
    else:
        results['grad_A'] = False
        results['grad_B'] = False
    
    return results


def test_matmul_chain():
    """Test chain rule with matmul."""
    results = {}
    
    # y = A @ (B @ x)
    A = Tensor([[1.0, 0.0], [0.0, 1.0]])  # Identity
    B = Tensor([[2.0, 0.0], [0.0, 3.0]])  # Diagonal
    x = Tensor([[1.0], [1.0]])            # Column vector
    
    y = A @ (B @ x)  # Should be [[2], [3]]
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, [[2], [3]])
        
        y.backward()
        # dy/dx = A @ B @ I = A @ B
        expected_x_grad = np.array([[2], [3]])
        results['grad_x'] = np.allclose(x.grad, expected_x_grad) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad_x'] = False
    
    return results


def test_linear_layer():
    """Test linear layer implementation."""
    results = {}
    
    # Create linear layer
    linear = Linear(3, 2, bias=True)
    
    # Test forward pass
    x = Tensor([[1.0, 2.0, 3.0]])  # (1, 3)
    y = linear(x)
    
    if y is not None and y.data is not None:
        results['shape'] = y.shape == (1, 2)
        
        # Test backward
        y.backward()
        results['grad_weight_shape'] = linear.weight.grad.shape == (3, 2) if linear.weight.grad is not None else False
        results['grad_input'] = x.grad is not None and x.grad.shape == (1, 3)
    else:
        results['shape'] = False
        results['grad_weight_shape'] = False
        results['grad_input'] = False
    
    return results


if __name__ == "__main__":
    print("Day 16: Matrix Multiplication Gradient")
    print("=" * 60)
    
    print("\nTranspose Tests:")
    trans_results = test_transpose()
    for name, passed in trans_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMatmul 2D Tests:")
    mm_results = test_matmul_2d()
    for name, passed in mm_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMatmul Gradient Tests:")
    grad_results = test_matmul_gradient()
    for name, passed in grad_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMatmul Chain Rule Tests:")
    chain_results = test_matmul_chain()
    for name, passed in chain_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLinear Layer Tests:")
    linear_results = test_linear_layer()
    for name, passed in linear_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day16.py for comprehensive tests!")
