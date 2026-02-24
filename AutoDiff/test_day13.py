"""Test Suite for Day 13: Broadcasting"""

import numpy as np
import pytest

from day13 import Tensor


def test_broadcast_scalar_add():
    """Test broadcasting scalar addition."""
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = Tensor([10])
    c = a + b
    
    assert c is not None and c.data is not None, "Addition returned None"
    expected = [[11, 12, 13], [14, 15, 16]]
    assert np.allclose(c.data, expected), f"a + [10] = {c.data}"


def test_broadcast_row_add():
    """Test broadcasting row vector addition."""
    a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = Tensor([10, 20, 30])  # (3,)
    c = a + b
    
    assert c is not None and c.data is not None, "Addition returned None"
    expected = [[11, 22, 33], [14, 25, 36]]
    assert np.allclose(c.data, expected), f"Broadcast add: {c.data}"


def test_broadcast_col_add():
    """Test broadcasting column vector addition."""
    a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = Tensor([[10], [20]])  # (2, 1)
    c = a + b
    
    assert c is not None and c.data is not None, "Addition returned None"
    expected = [[11, 12, 13], [24, 25, 26]]
    assert np.allclose(c.data, expected), f"Broadcast add: {c.data}"


def test_broadcast_mul():
    """Test broadcasting multiplication."""
    a = Tensor([[1, 2], [3, 4]])  # (2, 2)
    b = Tensor([10, 100])  # (2,)
    c = a * b
    
    assert c is not None and c.data is not None, "Multiplication returned None"
    expected = [[10, 200], [30, 400]]
    assert np.allclose(c.data, expected), f"Broadcast mul: {c.data}"


def test_broadcast_shapes():
    """Test various broadcast shape combinations."""
    # (4, 1) + (1, 3) -> (4, 3)
    a = Tensor([[1], [2], [3], [4]])
    b = Tensor([[10, 20, 30]])
    c = a + b
    
    assert c is not None and c.shape is not None, "Shape is None"
    assert c.shape == (4, 3), f"Shape = {c.shape}, expected (4, 3)"


def test_backward_broadcast_scalar():
    """Test backward pass with scalar broadcasting."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    b = Tensor([10.0])  # (1,)
    c = a + b
    c.backward()
    
    assert a.grad is not None and b.grad is not None, "Gradients are None"
    
    # d(a+b)/da = 1s
    assert np.allclose(a.grad, [[1, 1], [1, 1]]), f"da = {a.grad}"
    
    # d(a+b)/db should sum all 1s = 4
    assert np.allclose(b.grad, [4]), f"db = {b.grad}, expected [4]"


def test_backward_broadcast_row():
    """Test backward pass with row broadcasting."""
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    b = Tensor([10.0, 20.0, 30.0])  # (3,)
    c = a * b
    c.backward()
    
    assert a.grad is not None and b.grad is not None, "Gradients are None"
    
    # da = b broadcast to each row
    expected_da = [[10, 20, 30], [10, 20, 30]]
    assert np.allclose(a.grad, expected_da), f"da = {a.grad}"
    
    # db = sum over rows of a
    expected_db = [1 + 4, 2 + 5, 3 + 6]  # [5, 7, 9]
    assert np.allclose(b.grad, expected_db), f"db = {b.grad}, expected {expected_db}"


def test_backward_broadcast_col():
    """Test backward pass with column broadcasting."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    b = Tensor([[10.0], [20.0]])  # (2, 1)
    c = a * b
    c.backward()
    
    assert a.grad is not None and b.grad is not None, "Gradients are None"
    
    # da = b broadcast
    expected_da = [[10, 10], [20, 20]]
    assert np.allclose(a.grad, expected_da), f"da = {a.grad}"
    
    # db = sum over columns of a
    expected_db = [[1 + 2], [3 + 4]]  # [[3], [7]]
    assert np.allclose(b.grad, expected_db), f"db = {b.grad}, expected {expected_db}"


def test_sum_keepdims():
    """Test sum with keepdims."""
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    s_all = a.sum()
    assert s_all is not None, "sum() returned None"
    assert np.isclose(s_all.data, 21.0), f"sum() = {s_all.data}, expected 21"
    
    s_axis0 = a.sum(axis=0)
    assert s_axis0 is not None, "sum(axis=0) returned None"
    assert np.allclose(s_axis0.data, [5, 7, 9]), f"sum(axis=0) = {s_axis0.data}"
    
    s_axis1 = a.sum(axis=1)
    assert s_axis1 is not None, "sum(axis=1) returned None"
    assert np.allclose(s_axis1.data, [6, 15]), f"sum(axis=1) = {s_axis1.data}"


def test_mean():
    """Test mean."""
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    m = a.mean()
    assert m is not None, "mean() returned None"
    assert np.isclose(m.data, 3.5), f"mean() = {m.data}, expected 3.5"


def test_sum_backward():
    """Test backward pass for sum."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    s = a.sum()
    s.backward()
    
    assert a.grad is not None, "Gradient is None"
    expected = [[1, 1], [1, 1]]
    assert np.allclose(a.grad, expected), f"da = {a.grad}"


def test_mean_backward():
    """Test backward pass for mean."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    m = a.mean()
    m.backward()
    
    assert a.grad is not None, "Gradient is None"
    expected = [[0.25, 0.25], [0.25, 0.25]]  # 1/4 for each element
    assert np.allclose(a.grad, expected), f"da = {a.grad}"


def test_broadcast_3d():
    """Test 3D broadcasting."""
    a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
    b = Tensor([10, 100])  # (2,)
    c = a + b
    
    assert c is not None and c.shape is not None, "Shape is None"
    assert c.shape == (2, 2, 2), f"Shape = {c.shape}"
    
    expected = [[[11, 102], [13, 104]], [[15, 106], [17, 108]]]
    assert np.allclose(c.data, expected), f"3D broadcast: {c.data}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
