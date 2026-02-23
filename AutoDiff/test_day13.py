"""Test Suite for Day 13: Broadcasting Rules"""

import numpy as np
import sys
from typing import Tuple

try:
    from day13 import (
        Tensor,
        test_broadcast_compatibility,
        test_broadcast_shape,
        test_broadcast_forward,
        test_unbroadcast,
        test_broadcast_backward
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_compat_same_shape() -> Tuple[bool, str]:
    """Test compatibility of same shapes."""
    try:
        result = Tensor.broadcast_compatible((2, 3), (2, 3))
        if result is None:
            return False, "Returned None"
        if not result:
            return False, "Same shapes should be compatible"
        return True, "(2,3) + (2,3)"
    except Exception as e:
        return False, str(e)


def test_compat_scalar() -> Tuple[bool, str]:
    """Test scalar broadcasting compatibility."""
    try:
        result = Tensor.broadcast_compatible((3, 4), ())
        if result is None:
            return False, "Returned None"
        if not result:
            return False, "Scalar broadcast should work"
        return True, "(3,4) + ()"
    except Exception as e:
        return False, str(e)


def test_compat_row() -> Tuple[bool, str]:
    """Test row broadcasting compatibility."""
    try:
        result = Tensor.broadcast_compatible((2, 3), (3,))
        if result is None:
            return False, "Returned None"
        if not result:
            return False, "Row broadcast should work"
        return True, "(2,3) + (3,)"
    except Exception as e:
        return False, str(e)


def test_compat_column() -> Tuple[bool, str]:
    """Test column broadcasting compatibility."""
    try:
        result = Tensor.broadcast_compatible((2, 3), (2, 1))
        if result is None:
            return False, "Returned None"
        if not result:
            return False, "Column broadcast should work"
        return True, "(2,3) + (2,1)"
    except Exception as e:
        return False, str(e)


def test_compat_incompatible() -> Tuple[bool, str]:
    """Test incompatible shapes."""
    try:
        result = Tensor.broadcast_compatible((2, 3), (4,))
        if result is None:
            return False, "Returned None"
        if result:
            return False, "(2,3) and (4,) should NOT be compatible"
        return True, "(2,3) + (4,) incompatible"
    except Exception as e:
        return False, str(e)


def test_broadcast_shape_row() -> Tuple[bool, str]:
    """Test broadcast shape for row broadcast."""
    try:
        result = Tensor.broadcast_shape((2, 3), (3,))
        if result is None:
            return False, "Returned None"
        if result != (2, 3):
            return False, f"Expected (2,3), got {result}"
        return True, "(2,3) + (3,) -> (2,3)"
    except Exception as e:
        return False, str(e)


def test_broadcast_shape_cross() -> Tuple[bool, str]:
    """Test broadcast shape for cross broadcast."""
    try:
        result = Tensor.broadcast_shape((1, 5), (3, 1))
        if result is None:
            return False, "Returned None"
        if result != (3, 5):
            return False, f"Expected (3,5), got {result}"
        return True, "(1,5) + (3,1) -> (3,5)"
    except Exception as e:
        return False, str(e)


def test_broadcast_shape_3d() -> Tuple[bool, str]:
    """Test broadcast shape for 3D broadcast."""
    try:
        result = Tensor.broadcast_shape((2, 1, 3), (4, 3))
        if result is None:
            return False, "Returned None"
        if result != (2, 4, 3):
            return False, f"Expected (2,4,3), got {result}"
        return True, "(2,1,3) + (4,3) -> (2,4,3)"
    except Exception as e:
        return False, str(e)


def test_unbroadcast_sum_first() -> Tuple[bool, str]:
    """Test unbroadcast by summing first dimension."""
    try:
        grad = np.ones((2, 3))
        result = Tensor.unbroadcast(grad, (3,))
        
        if result is None:
            return False, "Returned None"
        if result.shape != (3,):
            return False, f"Shape {result.shape}, expected (3,)"
        if not np.allclose(result, [2, 2, 2]):
            return False, f"Values {result}, expected [2,2,2]"
        return True, "(2,3) -> (3,) by summing"
    except Exception as e:
        return False, str(e)


def test_unbroadcast_sum_middle() -> Tuple[bool, str]:
    """Test unbroadcast by summing along broadcast-1 dimension."""
    try:
        grad = np.ones((2, 3))
        result = Tensor.unbroadcast(grad, (2, 1))
        
        if result is None:
            return False, "Returned None"
        if result.shape != (2, 1):
            return False, f"Shape {result.shape}, expected (2,1)"
        if not np.allclose(result, [[3], [3]]):
            return False, f"Values {result}, expected [[3],[3]]"
        return True, "(2,3) -> (2,1) by summing"
    except Exception as e:
        return False, str(e)


def test_unbroadcast_no_change() -> Tuple[bool, str]:
    """Test unbroadcast when shapes match."""
    try:
        grad = np.array([[1, 2], [3, 4]])
        result = Tensor.unbroadcast(grad, (2, 2))
        
        if result is None:
            return False, "Returned None"
        if not np.allclose(result, grad):
            return False, "Should not change when shapes match"
        return True, "No change when shapes match"
    except Exception as e:
        return False, str(e)


def test_add_row_broadcast_forward() -> Tuple[bool, str]:
    """Test forward addition with row broadcast."""
    try:
        a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        b = Tensor([10, 20, 30])             # (3,)
        c = a + b
        
        expected = np.array([[11, 22, 33], [14, 25, 36]])
        if not np.allclose(c.data, expected):
            return False, f"Got {c.data}"
        return True, "Row broadcast add"
    except Exception as e:
        return False, str(e)


def test_add_column_broadcast_forward() -> Tuple[bool, str]:
    """Test forward addition with column broadcast."""
    try:
        a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        b = Tensor([[10], [20]])            # (2, 1)
        c = a + b
        
        expected = np.array([[11, 12, 13], [24, 25, 26]])
        if not np.allclose(c.data, expected):
            return False, f"Got {c.data}"
        return True, "Column broadcast add"
    except Exception as e:
        return False, str(e)


def test_mul_broadcast_forward() -> Tuple[bool, str]:
    """Test forward multiplication with broadcast."""
    try:
        a = Tensor([[1, 2], [3, 4]])  # (2, 2)
        b = Tensor([2, 3])            # (2,)
        c = a * b
        
        expected = np.array([[2, 6], [6, 12]])
        if not np.allclose(c.data, expected):
            return False, f"Got {c.data}"
        return True, "Broadcast multiply"
    except Exception as e:
        return False, str(e)


def test_add_broadcast_backward() -> Tuple[bool, str]:
    """Test backward addition with broadcast."""
    try:
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        b = Tensor([1.0, 1.0, 1.0])                      # (3,)
        c = a + b
        c.backward()
        
        # da should be ones (2,3)
        if a.grad is None:
            return False, "a.grad is None"
        if not np.allclose(a.grad, np.ones((2, 3))):
            return False, f"a.grad = {a.grad}"
        
        # db should be [2, 2, 2] (summed over first axis)
        if b.grad is None:
            return False, "b.grad is None"
        if not np.allclose(b.grad, [2, 2, 2]):
            return False, f"b.grad = {b.grad}, expected [2,2,2]"
        
        return True, "Backward broadcast add"
    except Exception as e:
        return False, str(e)


def test_mul_broadcast_backward() -> Tuple[bool, str]:
    """Test backward multiplication with broadcast."""
    try:
        # (2, 3) * (2, 1)
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        y = Tensor([[2.0], [3.0]])                      # (2, 1)
        z = x * y
        z.backward()
        
        # dz/dx = y broadcast: [[2,2,2], [3,3,3]]
        if x.grad is None:
            return False, "x.grad is None"
        expected_x = np.array([[2, 2, 2], [3, 3, 3]])
        if not np.allclose(x.grad, expected_x):
            return False, f"x.grad = {x.grad}, expected {expected_x}"
        
        # dz/dy = x summed: [[1+2+3], [4+5+6]] = [[6], [15]]
        if y.grad is None:
            return False, "y.grad is None"
        expected_y = np.array([[6], [15]])
        if not np.allclose(y.grad, expected_y):
            return False, f"y.grad = {y.grad}, expected {expected_y}"
        
        return True, "Backward broadcast mul"
    except Exception as e:
        return False, str(e)


def test_scalar_broadcast_backward() -> Tuple[bool, str]:
    """Test backward with scalar broadcast."""
    try:
        a = Tensor([1.0, 2.0, 3.0])
        b = a * 2  # Scalar broadcast
        b.backward()
        
        if a.grad is None:
            return False, "a.grad is None"
        if not np.allclose(a.grad, [2, 2, 2]):
            return False, f"a.grad = {a.grad}"
        return True, "Scalar broadcast backward"
    except Exception as e:
        return False, str(e)


def test_cross_broadcast_backward() -> Tuple[bool, str]:
    """Test backward with cross broadcast."""
    try:
        # (1, 3) + (2, 1) -> (2, 3)
        a = Tensor([[1.0, 2.0, 3.0]])  # (1, 3)
        b = Tensor([[10.0], [20.0]])   # (2, 1)
        c = a + b
        c.backward()
        
        # da: sum over axis 0 -> (1, 3), all 2s
        if a.grad is None:
            return False, "a.grad is None"
        if not np.allclose(a.grad, [[2, 2, 2]]):
            return False, f"a.grad = {a.grad}"
        
        # db: sum over axis 1 -> (2, 1), all 3s
        if b.grad is None:
            return False, "b.grad is None"
        if not np.allclose(b.grad, [[3], [3]]):
            return False, f"b.grad = {b.grad}"
        
        return True, "Cross broadcast backward"
    except Exception as e:
        return False, str(e)


def test_chain_rule_broadcast() -> Tuple[bool, str]:
    """Test chain rule with broadcasting."""
    try:
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        w = Tensor([0.5, 0.5])                 # (2,)
        
        # y = x * w + x (uses x twice with broadcast)
        y = x * w + x
        y.backward()
        
        # dy/dx = w + 1 = [1.5, 1.5] broadcast to (2, 2)
        if x.grad is None:
            return False, "x.grad is None"
        expected = np.array([[1.5, 1.5], [1.5, 1.5]])
        if not np.allclose(x.grad, expected):
            return False, f"x.grad = {x.grad}"
        
        return True, "Chain rule with broadcast"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test against PyTorch reference (if available)."""
    try:
        import torch
        
        # Test case: (2, 3) * (3,) backward
        np_a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np_b = np.array([2.0, 3.0, 4.0])
        
        # Our implementation
        a = Tensor(np_a)
        b = Tensor(np_b)
        c = a * b
        c.backward()
        
        # PyTorch reference
        ta = torch.tensor(np_a, requires_grad=True)
        tb = torch.tensor(np_b, requires_grad=True)
        tc = ta * tb
        tc.backward(torch.ones_like(tc))
        
        if not np.allclose(a.grad, ta.grad.numpy()):
            return False, f"a.grad mismatch"
        if not np.allclose(b.grad, tb.grad.numpy()):
            return False, f"b.grad mismatch"
        
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("compat_same_shape", test_compat_same_shape),
        ("compat_scalar", test_compat_scalar),
        ("compat_row", test_compat_row),
        ("compat_column", test_compat_column),
        ("compat_incompatible", test_compat_incompatible),
        ("broadcast_shape_row", test_broadcast_shape_row),
        ("broadcast_shape_cross", test_broadcast_shape_cross),
        ("broadcast_shape_3d", test_broadcast_shape_3d),
        ("unbroadcast_sum_first", test_unbroadcast_sum_first),
        ("unbroadcast_sum_middle", test_unbroadcast_sum_middle),
        ("unbroadcast_no_change", test_unbroadcast_no_change),
        ("add_row_broadcast_forward", test_add_row_broadcast_forward),
        ("add_column_broadcast_forward", test_add_column_broadcast_forward),
        ("mul_broadcast_forward", test_mul_broadcast_forward),
        ("add_broadcast_backward", test_add_broadcast_backward),
        ("mul_broadcast_backward", test_mul_broadcast_backward),
        ("scalar_broadcast_backward", test_scalar_broadcast_backward),
        ("cross_broadcast_backward", test_cross_broadcast_backward),
        ("chain_rule_broadcast", test_chain_rule_broadcast),
        ("against_pytorch", test_against_pytorch),
    ]
    
    print(f"\n{'='*50}\nDay 13: Broadcasting Rules - Tests\n{'='*50}")
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
