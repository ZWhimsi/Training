"""Test Suite for Day 18: Max and Min Operations"""

import numpy as np
import sys
from typing import Tuple

try:
    from day18 import (
        Tensor,
        test_max_all,
        test_max_axis,
        test_max_gradient_axis,
        test_max_ties,
        test_min,
        test_clamp,
        test_relu,
        test_leaky_relu,
        test_maximum_tensors,
        test_argmax
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_max_all_forward() -> Tuple[bool, str]:
    """Test max of all elements forward."""
    try:
        x = Tensor([[1, 5, 3], [4, 2, 6]])
        y = x.max()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, 6):
            return False, f"max = {y.data}"
        return True, "max([[1,5,3],[4,2,6]]) = 6"
    except Exception as e:
        return False, str(e)


def test_max_all_backward() -> Tuple[bool, str]:
    """Test max of all elements backward."""
    try:
        x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        y = x.max()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        expected = np.array([[0, 0, 0], [0, 0, 1]])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "Gradient only at max"
    except Exception as e:
        return False, str(e)


def test_max_axis0_forward() -> Tuple[bool, str]:
    """Test max along axis 0."""
    try:
        x = Tensor([[1, 5, 3], [4, 2, 6]])
        y = x.max(axis=0)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (3,):
            return False, f"shape = {y.shape}"
        if not np.allclose(y.data, [4, 5, 6]):
            return False, f"values = {y.data}"
        return True, "max(axis=0) = [4,5,6]"
    except Exception as e:
        return False, str(e)


def test_max_axis1_forward() -> Tuple[bool, str]:
    """Test max along axis 1."""
    try:
        x = Tensor([[1, 5, 3], [4, 2, 6]])
        y = x.max(axis=1)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (2,):
            return False, f"shape = {y.shape}"
        if not np.allclose(y.data, [5, 6]):
            return False, f"values = {y.data}"
        return True, "max(axis=1) = [5,6]"
    except Exception as e:
        return False, str(e)


def test_max_axis0_backward() -> Tuple[bool, str]:
    """Test max axis 0 backward."""
    try:
        x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        y = x.max(axis=0)
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        expected = np.array([[0, 1, 0], [1, 0, 1]])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "Max positions get gradient"
    except Exception as e:
        return False, str(e)


def test_max_axis1_backward() -> Tuple[bool, str]:
    """Test max axis 1 backward."""
    try:
        x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        y = x.max(axis=1)
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        expected = np.array([[0, 1, 0], [0, 0, 1]])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "Row max gradient OK"
    except Exception as e:
        return False, str(e)


def test_max_keepdims() -> Tuple[bool, str]:
    """Test max with keepdims."""
    try:
        x = Tensor([[1, 5, 3], [4, 2, 6]])
        y = x.max(axis=1, keepdims=True)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (2, 1):
            return False, f"shape = {y.shape}"
        return True, "keepdims works"
    except Exception as e:
        return False, str(e)


def test_max_ties_equal_split() -> Tuple[bool, str]:
    """Test max with tied values splits gradient."""
    try:
        x = Tensor([5.0, 5.0, 1.0, 5.0])
        y = x.max()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        expected = np.array([1/3, 1/3, 0, 1/3])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "Ties split gradient"
    except Exception as e:
        return False, str(e)


def test_min_all_forward() -> Tuple[bool, str]:
    """Test min of all elements."""
    try:
        x = Tensor([[1, 5, 3], [4, 2, 6]])
        y = x.min()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, 1):
            return False, f"min = {y.data}"
        return True, "min = 1"
    except Exception as e:
        return False, str(e)


def test_min_backward() -> Tuple[bool, str]:
    """Test min backward."""
    try:
        x = Tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        y = x.min()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        expected = np.array([[1, 0, 0], [0, 0, 0]])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "Min gradient OK"
    except Exception as e:
        return False, str(e)


def test_min_axis() -> Tuple[bool, str]:
    """Test min along axis."""
    try:
        x = Tensor([[1, 5, 3], [4, 2, 6]])
        y = x.min(axis=1)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, [1, 2]):
            return False, f"values = {y.data}"
        return True, "min(axis=1) = [1,2]"
    except Exception as e:
        return False, str(e)


def test_clamp_forward() -> Tuple[bool, str]:
    """Test clamp forward pass."""
    try:
        x = Tensor([-3, -1, 0, 1, 3])
        y = x.clamp(min_val=-1, max_val=2)
        
        if y is None or y.data is None:
            return False, "Returned None"
        expected = np.array([-1, -1, 0, 1, 2])
        if not np.allclose(y.data, expected):
            return False, f"values = {y.data}"
        return True, "clamp(-1,2) works"
    except Exception as e:
        return False, str(e)


def test_clamp_backward() -> Tuple[bool, str]:
    """Test clamp backward pass."""
    try:
        x = Tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        y = x.clamp(min_val=-1, max_val=2)
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        expected = np.array([0, 1, 1, 1, 0])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "Clamp gradient OK"
    except Exception as e:
        return False, str(e)


def test_clamp_min_only() -> Tuple[bool, str]:
    """Test clamp with only min."""
    try:
        x = Tensor([-2, -1, 0, 1, 2])
        y = x.clamp(min_val=0)
        
        if y is None or y.data is None:
            return False, "Returned None"
        expected = np.array([0, 0, 0, 1, 2])
        if not np.allclose(y.data, expected):
            return False, f"values = {y.data}"
        return True, "clamp(min=0) = ReLU"
    except Exception as e:
        return False, str(e)


def test_relu_forward() -> Tuple[bool, str]:
    """Test ReLU forward."""
    try:
        x = Tensor([-2, -1, 0, 1, 2])
        y = x.relu()
        
        if y is None or y.data is None:
            return False, "Returned None"
        expected = np.array([0, 0, 0, 1, 2])
        if not np.allclose(y.data, expected):
            return False, f"values = {y.data}"
        return True, "ReLU forward OK"
    except Exception as e:
        return False, str(e)


def test_relu_backward() -> Tuple[bool, str]:
    """Test ReLU backward."""
    try:
        x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = x.relu()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        expected = np.array([0, 0, 0, 1, 1])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "ReLU gradient OK"
    except Exception as e:
        return False, str(e)


def test_relu_chain() -> Tuple[bool, str]:
    """Test ReLU in chain rule."""
    try:
        x = Tensor([-1.0, 1.0, 2.0])
        y = (x * 2).relu().sum()  # [-2, 2, 4] -> [0, 2, 4] -> 6
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        # dy/dx = 2 * (input > 0) = [0, 2, 2]
        expected = np.array([0, 2, 2])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "ReLU chain rule OK"
    except Exception as e:
        return False, str(e)


def test_leaky_relu_forward() -> Tuple[bool, str]:
    """Test Leaky ReLU forward."""
    try:
        x = Tensor([-2, -1, 0, 1, 2])
        y = x.leaky_relu(negative_slope=0.1)
        
        if y is None or y.data is None:
            return False, "Returned None"
        expected = np.array([-0.2, -0.1, 0, 1, 2])
        if not np.allclose(y.data, expected):
            return False, f"values = {y.data}"
        return True, "Leaky ReLU forward OK"
    except Exception as e:
        return False, str(e)


def test_leaky_relu_backward() -> Tuple[bool, str]:
    """Test Leaky ReLU backward."""
    try:
        x = Tensor([-2.0, 0.0, 2.0])
        y = x.leaky_relu(negative_slope=0.1)
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        expected = np.array([0.1, 0.1, 1])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        return True, "Leaky ReLU gradient OK"
    except Exception as e:
        return False, str(e)


def test_maximum_two_tensors() -> Tuple[bool, str]:
    """Test element-wise maximum."""
    try:
        a = Tensor([1, 4, 3])
        b = Tensor([2, 2, 5])
        c = a.maximum(b)
        
        if c is None or c.data is None:
            return False, "Returned None"
        expected = np.array([2, 4, 5])
        if not np.allclose(c.data, expected):
            return False, f"values = {c.data}"
        return True, "max(a,b) element-wise OK"
    except Exception as e:
        return False, str(e)


def test_maximum_gradient() -> Tuple[bool, str]:
    """Test element-wise maximum gradient."""
    try:
        a = Tensor([1.0, 4.0, 3.0])
        b = Tensor([2.0, 2.0, 5.0])
        c = a.maximum(b)
        c.backward()
        
        if a.grad is None or b.grad is None:
            return False, "grad is None"
        if not np.allclose(a.grad, [0, 1, 0]):
            return False, f"a.grad = {a.grad}"
        if not np.allclose(b.grad, [1, 0, 1]):
            return False, f"b.grad = {b.grad}"
        return True, "maximum gradient OK"
    except Exception as e:
        return False, str(e)


def test_minimum_two_tensors() -> Tuple[bool, str]:
    """Test element-wise minimum."""
    try:
        a = Tensor([1, 4, 3])
        b = Tensor([2, 2, 5])
        c = a.minimum(b)
        
        if c is None or c.data is None:
            return False, "Returned None"
        expected = np.array([1, 2, 3])
        if not np.allclose(c.data, expected):
            return False, f"values = {c.data}"
        return True, "min(a,b) element-wise OK"
    except Exception as e:
        return False, str(e)


def test_minimum_gradient() -> Tuple[bool, str]:
    """Test element-wise minimum gradient."""
    try:
        a = Tensor([1.0, 4.0, 3.0])
        b = Tensor([2.0, 2.0, 5.0])
        c = a.minimum(b)
        c.backward()
        
        if a.grad is None or b.grad is None:
            return False, "grad is None"
        if not np.allclose(a.grad, [1, 0, 1]):
            return False, f"a.grad = {a.grad}"
        if not np.allclose(b.grad, [0, 1, 0]):
            return False, f"b.grad = {b.grad}"
        return True, "minimum gradient OK"
    except Exception as e:
        return False, str(e)


def test_argmax_all() -> Tuple[bool, str]:
    """Test argmax of all elements."""
    try:
        x = Tensor([[1, 5, 3], [4, 2, 6]])
        idx = x.argmax()
        
        if idx is None:
            return False, "Returned None"
        if idx != 5:
            return False, f"argmax = {idx}"
        return True, "argmax = 5 (flattened)"
    except Exception as e:
        return False, str(e)


def test_argmax_axis() -> Tuple[bool, str]:
    """Test argmax along axis."""
    try:
        x = Tensor([[1, 5, 3], [4, 2, 6]])
        
        idx0 = x.argmax(axis=0)
        if idx0 is None or not np.array_equal(idx0, [1, 0, 1]):
            return False, f"argmax(axis=0) = {idx0}"
        
        idx1 = x.argmax(axis=1)
        if idx1 is None or not np.array_equal(idx1, [1, 2]):
            return False, f"argmax(axis=1) = {idx1}"
        
        return True, "argmax axis OK"
    except Exception as e:
        return False, str(e)


def test_argmin() -> Tuple[bool, str]:
    """Test argmin operation."""
    try:
        x = Tensor([[1, 5, 3], [4, 2, 6]])
        idx = x.argmin()
        
        if idx is None:
            return False, "Returned None"
        if idx != 0:
            return False, f"argmin = {idx}"
        return True, "argmin = 0"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test against PyTorch reference."""
    try:
        import torch
        
        np_x = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        
        # Our max
        x = Tensor(np_x)
        y = x.max()
        y.backward()
        
        # PyTorch max
        tx = torch.tensor(np_x, requires_grad=True)
        ty = tx.max()
        ty.backward()
        
        if not np.allclose(y.data, ty.item()):
            return False, "Forward mismatch"
        if not np.allclose(x.grad, tx.grad.numpy()):
            return False, "Gradient mismatch"
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def test_3d_max() -> Tuple[bool, str]:
    """Test max on 3D tensor."""
    try:
        x = Tensor(np.arange(24).reshape(2, 3, 4).astype(float))
        y = x.max(axis=1)  # Max along middle axis
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (2, 4):
            return False, f"shape = {y.shape}"
        return True, "3D max OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("max_all_forward", test_max_all_forward),
        ("max_all_backward", test_max_all_backward),
        ("max_axis0_forward", test_max_axis0_forward),
        ("max_axis1_forward", test_max_axis1_forward),
        ("max_axis0_backward", test_max_axis0_backward),
        ("max_axis1_backward", test_max_axis1_backward),
        ("max_keepdims", test_max_keepdims),
        ("max_ties_equal_split", test_max_ties_equal_split),
        ("min_all_forward", test_min_all_forward),
        ("min_backward", test_min_backward),
        ("min_axis", test_min_axis),
        ("clamp_forward", test_clamp_forward),
        ("clamp_backward", test_clamp_backward),
        ("clamp_min_only", test_clamp_min_only),
        ("relu_forward", test_relu_forward),
        ("relu_backward", test_relu_backward),
        ("relu_chain", test_relu_chain),
        ("leaky_relu_forward", test_leaky_relu_forward),
        ("leaky_relu_backward", test_leaky_relu_backward),
        ("maximum_two_tensors", test_maximum_two_tensors),
        ("maximum_gradient", test_maximum_gradient),
        ("minimum_two_tensors", test_minimum_two_tensors),
        ("minimum_gradient", test_minimum_gradient),
        ("argmax_all", test_argmax_all),
        ("argmax_axis", test_argmax_axis),
        ("argmin", test_argmin),
        ("against_pytorch", test_against_pytorch),
        ("3d_max", test_3d_max),
    ]
    
    print(f"\n{'='*50}\nDay 18: Max and Min Operations - Tests\n{'='*50}")
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
