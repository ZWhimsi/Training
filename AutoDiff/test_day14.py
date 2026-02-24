"""Test Suite for Day 14: Matrix Operations"""

import numpy as np
import pytest

from day14 import Tensor


def test_matmul_basic():
    """Test basic matrix multiplication."""
    a = Tensor([[1, 2], [3, 4]])  # (2, 2)
    b = Tensor([[5, 6], [7, 8]])  # (2, 2)
    c = a @ b
    
    assert c is not None and c.data is not None, "matmul returned None"
    expected = [[19, 22], [43, 50]]
    assert np.allclose(c.data, expected), f"a @ b = {c.data}"


def test_matmul_different_shapes():
    """Test matmul with different shapes."""
    a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = Tensor([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    c = a @ b
    
    assert c is not None and c.shape is not None, "Shape is None"
    assert c.shape == (2, 2), f"Shape = {c.shape}, expected (2, 2)"
    
    expected = [[22, 28], [49, 64]]
    assert np.allclose(c.data, expected), f"a @ b = {c.data}"


def test_matmul_1d():
    """Test matrix-vector multiplication."""
    a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = Tensor([1, 2, 3])  # (3,)
    c = a @ b
    
    assert c is not None and c.shape is not None, "Shape is None"
    expected = [14, 32]  # [1+4+9, 4+10+18]
    assert np.allclose(c.data, expected), f"a @ b = {c.data}"


def test_matmul_backward():
    """Test backward pass for matmul."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[1.0, 0.0], [0.0, 1.0]])  # Identity
    c = a @ b
    c.backward()
    
    assert a.grad is not None and b.grad is not None, "Gradients are None"
    # For identity matrix, gradients should be well-defined
    assert a.grad.shape == a.data.shape, f"a.grad shape mismatch"
    assert b.grad.shape == b.data.shape, f"b.grad shape mismatch"


def test_transpose():
    """Test transpose operation."""
    a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = a.T
    
    assert b is not None and b.shape is not None, "Transpose returned None"
    assert b.shape == (3, 2), f"Shape = {b.shape}, expected (3, 2)"
    
    expected = [[1, 4], [2, 5], [3, 6]]
    assert np.allclose(b.data, expected), f"a.T = {b.data}"


def test_transpose_backward():
    """Test backward pass for transpose."""
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = a.T
    c = b.sum()
    c.backward()
    
    assert a.grad is not None, "Gradient is None"
    expected = [[1, 1, 1], [1, 1, 1]]
    assert np.allclose(a.grad, expected), f"da = {a.grad}"


def test_reshape():
    """Test reshape operation."""
    a = Tensor([1, 2, 3, 4, 5, 6])
    b = a.reshape(2, 3)
    
    assert b is not None and b.shape is not None, "Reshape returned None"
    assert b.shape == (2, 3), f"Shape = {b.shape}, expected (2, 3)"
    
    expected = [[1, 2, 3], [4, 5, 6]]
    assert np.allclose(b.data, expected), f"reshape = {b.data}"


def test_reshape_backward():
    """Test backward pass for reshape."""
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = a.reshape(2, 3)
    c = b.sum()
    c.backward()
    
    assert a.grad is not None, "Gradient is None"
    expected = [1, 1, 1, 1, 1, 1]
    assert np.allclose(a.grad, expected), f"da = {a.grad}"


def test_flatten():
    """Test flatten operation."""
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.flatten()
    
    assert b is not None and b.shape is not None, "Flatten returned None"
    assert b.shape == (6,), f"Shape = {b.shape}, expected (6,)"
    
    expected = [1, 2, 3, 4, 5, 6]
    assert np.allclose(b.data, expected), f"flatten = {b.data}"


def test_flatten_backward():
    """Test backward pass for flatten."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.flatten()
    c = b.sum()
    c.backward()
    
    assert a.grad is not None, "Gradient is None"
    expected = [[1, 1], [1, 1]]
    assert np.allclose(a.grad, expected), f"da = {a.grad}"


def test_squeeze():
    """Test squeeze operation."""
    a = Tensor([[[1, 2, 3]]])  # (1, 1, 3)
    b = a.squeeze()
    
    assert b is not None and b.shape is not None, "Squeeze returned None"
    assert b.shape == (3,), f"Shape = {b.shape}, expected (3,)"


def test_unsqueeze():
    """Test unsqueeze operation."""
    a = Tensor([1, 2, 3])  # (3,)
    b = a.unsqueeze(0)
    
    assert b is not None and b.shape is not None, "Unsqueeze returned None"
    assert b.shape == (1, 3), f"Shape = {b.shape}, expected (1, 3)"
    
    c = a.unsqueeze(1)
    assert c is not None and c.shape is not None, "Unsqueeze returned None"
    assert c.shape == (3, 1), f"Shape = {c.shape}, expected (3, 1)"


def test_matmul_chain():
    """Test chain of matmuls."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[1.0, 0.0], [0.0, 1.0]])
    c = Tensor([[2.0, 0.0], [0.0, 2.0]])
    
    d = a @ b @ c
    
    assert d is not None, "Chain matmul returned None"
    expected = [[2, 4], [6, 8]]  # a * 2
    assert np.allclose(d.data, expected), f"a @ I @ 2I = {d.data}"


def test_matmul_gradient_numerical():
    """Verify matmul gradient numerically."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    
    c = a @ b
    loss = c.sum()
    loss.backward()
    
    a_grad = a.grad.copy()
    b_grad = b.grad.copy()
    
    # Numerical gradient check
    eps = 1e-5
    for i in range(2):
        for j in range(2):
            a_plus = a.data.copy()
            a_plus[i, j] += eps
            c_plus = np.sum(a_plus @ b.data)
            
            a_minus = a.data.copy()
            a_minus[i, j] -= eps
            c_minus = np.sum(a_minus @ b.data)
            
            numerical = (c_plus - c_minus) / (2 * eps)
            analytical = a_grad[i, j]
            
            assert abs(numerical - analytical) <= 1e-4, f"Grad mismatch at ({i},{j}): {analytical} vs {numerical}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
