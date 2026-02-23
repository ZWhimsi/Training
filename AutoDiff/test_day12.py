"""Test Suite for Day 12: Tensor Class"""

import numpy as np
import sys
from typing import Tuple

try:
    from day12 import Tensor, test_tensor_creation, test_tensor_operations, test_tensor_backward
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_tensor_init_list() -> Tuple[bool, str]:
    """Test tensor initialization from list."""
    try:
        t = Tensor([[1, 2], [3, 4]])
        
        if t.data is None:
            return False, "data is None"
        if not np.allclose(t.data, [[1, 2], [3, 4]]):
            return False, f"Wrong data: {t.data}"
        return True, "Tensor from list"
    except Exception as e:
        return False, str(e)


def test_tensor_init_numpy() -> Tuple[bool, str]:
    """Test tensor initialization from numpy array."""
    try:
        arr = np.array([1.0, 2.0, 3.0])
        t = Tensor(arr)
        
        if t.data is None:
            return False, "data is None"
        if not np.allclose(t.data, arr):
            return False, f"Wrong data: {t.data}"
        return True, "Tensor from numpy"
    except Exception as e:
        return False, str(e)


def test_shape_property() -> Tuple[bool, str]:
    """Test shape property."""
    try:
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        
        if t.shape is None:
            return False, "shape is None"
        if t.shape != (2, 3):
            return False, f"Wrong shape: {t.shape}"
        return True, f"shape = {t.shape}"
    except Exception as e:
        return False, str(e)


def test_ndim_property() -> Tuple[bool, str]:
    """Test ndim property."""
    try:
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([[1, 2], [3, 4]])
        t3 = Tensor([[[1]]])
        
        if t1.ndim != 1 or t2.ndim != 2 or t3.ndim != 3:
            return False, f"Wrong ndims: {t1.ndim}, {t2.ndim}, {t3.ndim}"
        return True, "ndim correct for 1D, 2D, 3D"
    except Exception as e:
        return False, str(e)


def test_size_property() -> Tuple[bool, str]:
    """Test size property."""
    try:
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        
        if t.size is None:
            return False, "size is None"
        if t.size != 6:
            return False, f"Wrong size: {t.size}"
        return True, f"size = {t.size}"
    except Exception as e:
        return False, str(e)


def test_repr() -> Tuple[bool, str]:
    """Test string representation."""
    try:
        t = Tensor([1, 2])
        s = repr(t)
        
        if s is None:
            return False, "__repr__ returned None"
        if 'Tensor' not in s and 'tensor' not in s.lower():
            return False, f"No 'Tensor' in repr: {s}"
        return True, "repr works"
    except Exception as e:
        return False, str(e)


def test_grad_initialization() -> Tuple[bool, str]:
    """Test gradient initialization."""
    try:
        t = Tensor([[1, 2], [3, 4]])
        
        if t.grad is None:
            return False, "grad is None"
        if t.grad.shape != t.data.shape:
            return False, f"grad shape {t.grad.shape} != data shape {t.data.shape}"
        if not np.allclose(t.grad, 0):
            return False, "grad not initialized to zeros"
        return True, "grad initialized to zeros"
    except Exception as e:
        return False, str(e)


def test_zeros() -> Tuple[bool, str]:
    """Test zeros creation."""
    try:
        t = Tensor.zeros((3, 4))
        
        if t is None:
            return False, "zeros returned None"
        if t.shape != (3, 4):
            return False, f"Wrong shape: {t.shape}"
        if not np.allclose(t.data, 0):
            return False, "Not all zeros"
        return True, "zeros((3,4)) works"
    except Exception as e:
        return False, str(e)


def test_ones() -> Tuple[bool, str]:
    """Test ones creation."""
    try:
        t = Tensor.ones((2, 5))
        
        if t is None:
            return False, "ones returned None"
        if t.shape != (2, 5):
            return False, f"Wrong shape: {t.shape}"
        if not np.allclose(t.data, 1):
            return False, "Not all ones"
        return True, "ones((2,5)) works"
    except Exception as e:
        return False, str(e)


def test_randn() -> Tuple[bool, str]:
    """Test random normal creation."""
    try:
        t = Tensor.randn((10, 10))
        
        if t is None:
            return False, "randn returned None"
        if t.shape != (10, 10):
            return False, f"Wrong shape: {t.shape}"
        # Check it's not all zeros or ones (random)
        if np.allclose(t.data, 0) or np.allclose(t.data, 1):
            return False, "Doesn't look random"
        return True, "randn((10,10)) works"
    except Exception as e:
        return False, str(e)


def test_from_numpy() -> Tuple[bool, str]:
    """Test from_numpy creation."""
    try:
        arr = np.array([[1.5, 2.5], [3.5, 4.5]])
        t = Tensor.from_numpy(arr)
        
        if t is None:
            return False, "from_numpy returned None"
        if not np.allclose(t.data, arr):
            return False, "Data doesn't match"
        return True, "from_numpy works"
    except Exception as e:
        return False, str(e)


def test_getitem() -> Tuple[bool, str]:
    """Test element access."""
    try:
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        
        val = t[0, 1]
        if val is None:
            return False, "__getitem__ returned None"
        if val != 2:
            return False, f"t[0,1] = {val}, expected 2"
        
        row = t[1]
        if not np.allclose(row, [4, 5, 6]):
            return False, f"t[1] = {row}, expected [4,5,6]"
        return True, "indexing works"
    except Exception as e:
        return False, str(e)


def test_item() -> Tuple[bool, str]:
    """Test scalar extraction."""
    try:
        t = Tensor([42.0])
        val = t.item()
        
        if val is None:
            return False, "item() returned None"
        if val != 42.0:
            return False, f"item() = {val}, expected 42.0"
        return True, "item() works"
    except Exception as e:
        return False, str(e)


def test_add_tensors() -> Tuple[bool, str]:
    """Test tensor addition."""
    try:
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        
        if c is None or c.data is None:
            return False, "Addition returned None"
        if not np.allclose(c.data, [5, 7, 9]):
            return False, f"a + b = {c.data}"
        return True, "[1,2,3] + [4,5,6] = [5,7,9]"
    except Exception as e:
        return False, str(e)


def test_add_scalar() -> Tuple[bool, str]:
    """Test tensor + scalar."""
    try:
        a = Tensor([1, 2, 3])
        b = a + 10
        
        if b is None or b.data is None:
            return False, "Scalar add returned None"
        if not np.allclose(b.data, [11, 12, 13]):
            return False, f"a + 10 = {b.data}"
        return True, "[1,2,3] + 10 = [11,12,13]"
    except Exception as e:
        return False, str(e)


def test_mul_tensors() -> Tuple[bool, str]:
    """Test tensor multiplication."""
    try:
        a = Tensor([2, 3, 4])
        b = Tensor([5, 6, 7])
        c = a * b
        
        if c is None or c.data is None:
            return False, "Multiplication returned None"
        if not np.allclose(c.data, [10, 18, 28]):
            return False, f"a * b = {c.data}"
        return True, "[2,3,4] * [5,6,7] = [10,18,28]"
    except Exception as e:
        return False, str(e)


def test_mul_scalar() -> Tuple[bool, str]:
    """Test tensor * scalar."""
    try:
        a = Tensor([1, 2, 3])
        b = a * 3
        
        if b is None or b.data is None:
            return False, "Scalar mul returned None"
        if not np.allclose(b.data, [3, 6, 9]):
            return False, f"a * 3 = {b.data}"
        return True, "[1,2,3] * 3 = [3,6,9]"
    except Exception as e:
        return False, str(e)


def test_neg() -> Tuple[bool, str]:
    """Test negation."""
    try:
        a = Tensor([1, -2, 3])
        b = -a
        
        if b is None or b.data is None:
            return False, "Negation returned None"
        if not np.allclose(b.data, [-1, 2, -3]):
            return False, f"-a = {b.data}"
        return True, "-[1,-2,3] = [-1,2,-3]"
    except Exception as e:
        return False, str(e)


def test_backward_add() -> Tuple[bool, str]:
    """Test backward pass for addition."""
    try:
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = a + b
        c.backward()
        
        if a.grad is None or b.grad is None:
            return False, "Gradients are None"
        if not np.allclose(a.grad, [1, 1]):
            return False, f"da = {a.grad}, expected [1,1]"
        if not np.allclose(b.grad, [1, 1]):
            return False, f"db = {b.grad}, expected [1,1]"
        return True, "d(a+b)/da = d(a+b)/db = 1"
    except Exception as e:
        return False, str(e)


def test_backward_mul() -> Tuple[bool, str]:
    """Test backward pass for multiplication."""
    try:
        x = Tensor([2.0, 3.0])
        y = Tensor([4.0, 5.0])
        z = x * y
        z.backward()
        
        if x.grad is None or y.grad is None:
            return False, "Gradients are None"
        # dz/dx = y, dz/dy = x
        if not np.allclose(x.grad, [4, 5]):
            return False, f"dx = {x.grad}, expected [4,5]"
        if not np.allclose(y.grad, [2, 3]):
            return False, f"dy = {y.grad}, expected [2,3]"
        return True, "d(x*y)/dx = y, d(x*y)/dy = x"
    except Exception as e:
        return False, str(e)


def test_backward_chain() -> Tuple[bool, str]:
    """Test backward pass with chain rule."""
    try:
        x = Tensor([2.0, 3.0])
        y = x * x  # x^2
        y.backward()
        
        if x.grad is None:
            return False, "Gradient is None"
        # d(x^2)/dx = 2x
        if not np.allclose(x.grad, [4, 6]):
            return False, f"dx = {x.grad}, expected [4,6]"
        return True, "d(x²)/dx = 2x"
    except Exception as e:
        return False, str(e)


def test_backward_complex() -> Tuple[bool, str]:
    """Test backward pass on complex expression."""
    try:
        a = Tensor([2.0])
        b = Tensor([3.0])
        c = a * b + a  # 2*3 + 2 = 8
        c.backward()
        
        if a.grad is None or b.grad is None:
            return False, "Gradients are None"
        # dc/da = b + 1 = 4, dc/db = a = 2
        if not np.allclose(a.grad, [4]):
            return False, f"da = {a.grad}, expected [4]"
        if not np.allclose(b.grad, [2]):
            return False, f"db = {b.grad}, expected [2]"
        return True, "d(a*b+a)/da = b+1, db = a"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("tensor_init_list", test_tensor_init_list),
        ("tensor_init_numpy", test_tensor_init_numpy),
        ("shape_property", test_shape_property),
        ("ndim_property", test_ndim_property),
        ("size_property", test_size_property),
        ("repr", test_repr),
        ("grad_initialization", test_grad_initialization),
        ("zeros", test_zeros),
        ("ones", test_ones),
        ("randn", test_randn),
        ("from_numpy", test_from_numpy),
        ("getitem", test_getitem),
        ("item", test_item),
        ("add_tensors", test_add_tensors),
        ("add_scalar", test_add_scalar),
        ("mul_tensors", test_mul_tensors),
        ("mul_scalar", test_mul_scalar),
        ("neg", test_neg),
        ("backward_add", test_backward_add),
        ("backward_mul", test_backward_mul),
        ("backward_chain", test_backward_chain),
        ("backward_complex", test_backward_complex),
    ]
    
    print(f"\n{'='*50}\nDay 12: Tensor Class - Tests\n{'='*50}")
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    run_all_tests()
