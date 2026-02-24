"""Test Suite for Day 12: Tensor Class"""

import numpy as np
import pytest

from day12 import Tensor, test_tensor_creation, test_tensor_operations, test_tensor_backward


def test_tensor_init_list():
    """Test tensor initialization from list."""
    t = Tensor([[1, 2], [3, 4]])
    
    assert t.data is not None, "data is None"
    assert np.allclose(t.data, [[1, 2], [3, 4]]), f"Wrong data: {t.data}"


def test_tensor_init_numpy():
    """Test tensor initialization from numpy array."""
    arr = np.array([1.0, 2.0, 3.0])
    t = Tensor(arr)
    
    assert t.data is not None, "data is None"
    assert np.allclose(t.data, arr), f"Wrong data: {t.data}"


def test_shape_property():
    """Test shape property."""
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    
    assert t.shape is not None, "shape is None"
    assert t.shape == (2, 3), f"Wrong shape: {t.shape}"


def test_ndim_property():
    """Test ndim property."""
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([[1, 2], [3, 4]])
    t3 = Tensor([[[1]]])
    
    assert t1.ndim == 1 and t2.ndim == 2 and t3.ndim == 3, f"Wrong ndims: {t1.ndim}, {t2.ndim}, {t3.ndim}"


def test_size_property():
    """Test size property."""
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    
    assert t.size is not None, "size is None"
    assert t.size == 6, f"Wrong size: {t.size}"


def test_repr():
    """Test string representation."""
    t = Tensor([1, 2])
    s = repr(t)
    
    assert s is not None, "__repr__ returned None"
    assert 'Tensor' in s or 'tensor' in s.lower(), f"No 'Tensor' in repr: {s}"


def test_grad_initialization():
    """Test gradient initialization."""
    t = Tensor([[1, 2], [3, 4]])
    
    assert t.grad is not None, "grad is None"
    assert t.grad.shape == t.data.shape, f"grad shape {t.grad.shape} != data shape {t.data.shape}"
    assert np.allclose(t.grad, 0), "grad not initialized to zeros"


def test_zeros():
    """Test zeros creation."""
    t = Tensor.zeros((3, 4))
    
    assert t is not None, "zeros returned None"
    assert t.shape == (3, 4), f"Wrong shape: {t.shape}"
    assert np.allclose(t.data, 0), "Not all zeros"


def test_ones():
    """Test ones creation."""
    t = Tensor.ones((2, 5))
    
    assert t is not None, "ones returned None"
    assert t.shape == (2, 5), f"Wrong shape: {t.shape}"
    assert np.allclose(t.data, 1), "Not all ones"


def test_randn():
    """Test random normal creation."""
    t = Tensor.randn((10, 10))
    
    assert t is not None, "randn returned None"
    assert t.shape == (10, 10), f"Wrong shape: {t.shape}"
    # Check it's not all zeros or ones (random)
    assert not np.allclose(t.data, 0) and not np.allclose(t.data, 1), "Doesn't look random"


def test_from_numpy():
    """Test from_numpy creation."""
    arr = np.array([[1.5, 2.5], [3.5, 4.5]])
    t = Tensor.from_numpy(arr)
    
    assert t is not None, "from_numpy returned None"
    assert np.allclose(t.data, arr), "Data doesn't match"


def test_getitem():
    """Test element access."""
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    
    val = t[0, 1]
    assert val is not None, "__getitem__ returned None"
    assert val == 2, f"t[0,1] = {val}, expected 2"
    
    row = t[1]
    assert np.allclose(row, [4, 5, 6]), f"t[1] = {row}, expected [4,5,6]"


def test_item():
    """Test scalar extraction."""
    t = Tensor([42.0])
    val = t.item()
    
    assert val is not None, "item() returned None"
    assert val == 42.0, f"item() = {val}, expected 42.0"


def test_add_tensors():
    """Test tensor addition."""
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    c = a + b
    
    assert c is not None and c.data is not None, "Addition returned None"
    assert np.allclose(c.data, [5, 7, 9]), f"a + b = {c.data}"


def test_add_scalar():
    """Test tensor + scalar."""
    a = Tensor([1, 2, 3])
    b = a + 10
    
    assert b is not None and b.data is not None, "Scalar add returned None"
    assert np.allclose(b.data, [11, 12, 13]), f"a + 10 = {b.data}"


def test_mul_tensors():
    """Test tensor multiplication."""
    a = Tensor([2, 3, 4])
    b = Tensor([5, 6, 7])
    c = a * b
    
    assert c is not None and c.data is not None, "Multiplication returned None"
    assert np.allclose(c.data, [10, 18, 28]), f"a * b = {c.data}"


def test_mul_scalar():
    """Test tensor * scalar."""
    a = Tensor([1, 2, 3])
    b = a * 3
    
    assert b is not None and b.data is not None, "Scalar mul returned None"
    assert np.allclose(b.data, [3, 6, 9]), f"a * 3 = {b.data}"


def test_neg():
    """Test negation."""
    a = Tensor([1, -2, 3])
    b = -a
    
    assert b is not None and b.data is not None, "Negation returned None"
    assert np.allclose(b.data, [-1, 2, -3]), f"-a = {b.data}"


def test_backward_add():
    """Test backward pass for addition."""
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    c = a + b
    c.backward()
    
    assert a.grad is not None and b.grad is not None, "Gradients are None"
    assert np.allclose(a.grad, [1, 1]), f"da = {a.grad}, expected [1,1]"
    assert np.allclose(b.grad, [1, 1]), f"db = {b.grad}, expected [1,1]"


def test_backward_mul():
    """Test backward pass for multiplication."""
    x = Tensor([2.0, 3.0])
    y = Tensor([4.0, 5.0])
    z = x * y
    z.backward()
    
    assert x.grad is not None and y.grad is not None, "Gradients are None"
    # dz/dx = y, dz/dy = x
    assert np.allclose(x.grad, [4, 5]), f"dx = {x.grad}, expected [4,5]"
    assert np.allclose(y.grad, [2, 3]), f"dy = {y.grad}, expected [2,3]"


def test_backward_chain():
    """Test backward pass with chain rule."""
    x = Tensor([2.0, 3.0])
    y = x * x  # x^2
    y.backward()
    
    assert x.grad is not None, "Gradient is None"
    # d(x^2)/dx = 2x
    assert np.allclose(x.grad, [4, 6]), f"dx = {x.grad}, expected [4,6]"


def test_backward_complex():
    """Test backward pass on complex expression."""
    a = Tensor([2.0])
    b = Tensor([3.0])
    c = a * b + a  # 2*3 + 2 = 8
    c.backward()
    
    assert a.grad is not None and b.grad is not None, "Gradients are None"
    # dc/da = b + 1 = 4, dc/db = a = 2
    assert np.allclose(a.grad, [4]), f"da = {a.grad}, expected [4]"
    assert np.allclose(b.grad, [2]), f"db = {b.grad}, expected [2]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
