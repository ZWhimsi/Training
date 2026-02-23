"""Test Suite for Day 17: Reshape and View Operations"""

import numpy as np
import sys
from typing import Tuple

try:
    from day17 import (
        Tensor,
        test_reshape_basic,
        test_reshape_gradient,
        test_flatten,
        test_flatten_partial,
        test_squeeze,
        test_unsqueeze,
        test_expand,
        test_permute
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_reshape_2d_to_1d() -> Tuple[bool, str]:
    """Test reshape from 2D to 1D."""
    try:
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x.reshape(6)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (6,):
            return False, f"shape = {y.shape}"
        if not np.allclose(y.data, [1, 2, 3, 4, 5, 6]):
            return False, "Values incorrect"
        return True, "(2,3) -> (6,)"
    except Exception as e:
        return False, str(e)


def test_reshape_1d_to_2d() -> Tuple[bool, str]:
    """Test reshape from 1D to 2D."""
    try:
        x = Tensor([1, 2, 3, 4, 5, 6])
        y = x.reshape(2, 3)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (2, 3):
            return False, f"shape = {y.shape}"
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        if not np.allclose(y.data, expected):
            return False, "Values incorrect"
        return True, "(6,) -> (2,3)"
    except Exception as e:
        return False, str(e)


def test_reshape_gradient_simple() -> Tuple[bool, str]:
    """Test reshape gradient preserves shape."""
    try:
        x = Tensor(np.arange(12).reshape(3, 4).astype(float))
        y = x.reshape(4, 3)
        loss = y.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "grad is None"
        if x.grad.shape != (3, 4):
            return False, f"grad shape = {x.grad.shape}"
        if not np.allclose(x.grad, np.ones((3, 4))):
            return False, "grad values incorrect"
        return True, "Gradient shape preserved"
    except Exception as e:
        return False, str(e)


def test_reshape_infer_dimension() -> Tuple[bool, str]:
    """Test reshape with -1 for inferred dimension."""
    try:
        x = Tensor(np.arange(24))
        y = x.reshape(4, -1)  # Should be (4, 6)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (4, 6):
            return False, f"shape = {y.shape}"
        return True, "Inferred -1 as 6"
    except Exception as e:
        return False, str(e)


def test_reshape_3d() -> Tuple[bool, str]:
    """Test 3D reshape."""
    try:
        x = Tensor(np.arange(24).reshape(2, 3, 4))
        y = x.reshape(3, 8)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (3, 8):
            return False, f"shape = {y.shape}"
        return True, "(2,3,4) -> (3,8)"
    except Exception as e:
        return False, str(e)


def test_flatten_all() -> Tuple[bool, str]:
    """Test flatten all dimensions."""
    try:
        x = Tensor(np.arange(24).reshape(2, 3, 4))
        y = x.flatten()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (24,):
            return False, f"shape = {y.shape}"
        if not np.allclose(y.data, np.arange(24)):
            return False, "Values incorrect"
        return True, "(2,3,4) -> (24,)"
    except Exception as e:
        return False, str(e)


def test_flatten_partial_batch() -> Tuple[bool, str]:
    """Test flatten keeping batch dimension."""
    try:
        x = Tensor(np.arange(24).reshape(2, 3, 4))
        y = x.flatten(start_dim=1)  # Keep batch, flatten rest
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (2, 12):
            return False, f"shape = {y.shape}"
        return True, "(2,3,4) -> (2,12)"
    except Exception as e:
        return False, str(e)


def test_flatten_gradient() -> Tuple[bool, str]:
    """Test flatten gradient."""
    try:
        x = Tensor(np.ones((2, 3, 4)))
        y = x.flatten()
        loss = y.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "grad is None"
        if x.grad.shape != (2, 3, 4):
            return False, f"grad shape = {x.grad.shape}"
        return True, "Flatten gradient OK"
    except Exception as e:
        return False, str(e)


def test_squeeze_all() -> Tuple[bool, str]:
    """Test squeeze all size-1 dimensions."""
    try:
        x = Tensor(np.arange(6).reshape(1, 2, 1, 3, 1))
        y = x.squeeze()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (2, 3):
            return False, f"shape = {y.shape}"
        return True, "(1,2,1,3,1) -> (2,3)"
    except Exception as e:
        return False, str(e)


def test_squeeze_specific_axis() -> Tuple[bool, str]:
    """Test squeeze specific axis."""
    try:
        x = Tensor(np.arange(6).reshape(1, 2, 1, 3))
        y = x.squeeze(axis=2)  # Only squeeze axis 2
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (1, 2, 3):
            return False, f"shape = {y.shape}"
        return True, "Squeeze axis 2 only"
    except Exception as e:
        return False, str(e)


def test_squeeze_gradient() -> Tuple[bool, str]:
    """Test squeeze gradient."""
    try:
        x = Tensor(np.ones((1, 3, 1, 4)))
        y = x.squeeze()
        loss = y.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "grad is None"
        if x.grad.shape != (1, 3, 1, 4):
            return False, f"grad shape = {x.grad.shape}"
        return True, "Squeeze gradient OK"
    except Exception as e:
        return False, str(e)


def test_unsqueeze_dim0() -> Tuple[bool, str]:
    """Test unsqueeze at dimension 0."""
    try:
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x.unsqueeze(0)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (1, 2, 3):
            return False, f"shape = {y.shape}"
        return True, "(2,3) -> (1,2,3)"
    except Exception as e:
        return False, str(e)


def test_unsqueeze_dim_neg() -> Tuple[bool, str]:
    """Test unsqueeze at negative dimension."""
    try:
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x.unsqueeze(-1)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (2, 3, 1):
            return False, f"shape = {y.shape}"
        return True, "(2,3) -> (2,3,1)"
    except Exception as e:
        return False, str(e)


def test_unsqueeze_gradient() -> Tuple[bool, str]:
    """Test unsqueeze gradient."""
    try:
        x = Tensor([1.0, 2.0, 3.0])
        y = x.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        loss = y.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "grad is None"
        if x.grad.shape != (3,):
            return False, f"grad shape = {x.grad.shape}"
        return True, "Unsqueeze gradient OK"
    except Exception as e:
        return False, str(e)


def test_expand_row() -> Tuple[bool, str]:
    """Test expand row vector."""
    try:
        x = Tensor([[1, 2, 3]])  # (1, 3)
        y = x.expand(4, 3)  # (4, 3)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (4, 3):
            return False, f"shape = {y.shape}"
        expected = np.array([[1, 2, 3]] * 4)
        if not np.allclose(y.data, expected):
            return False, "Values incorrect"
        return True, "(1,3) -> (4,3)"
    except Exception as e:
        return False, str(e)


def test_expand_col() -> Tuple[bool, str]:
    """Test expand column vector."""
    try:
        x = Tensor([[1], [2], [3]])  # (3, 1)
        y = x.expand(3, 4)  # (3, 4)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (3, 4):
            return False, f"shape = {y.shape}"
        expected = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
        if not np.allclose(y.data, expected):
            return False, "Values incorrect"
        return True, "(3,1) -> (3,4)"
    except Exception as e:
        return False, str(e)


def test_expand_gradient() -> Tuple[bool, str]:
    """Test expand gradient sums correctly."""
    try:
        x = Tensor([[1.0], [2.0]])  # (2, 1)
        y = x.expand(2, 5)  # (2, 5)
        loss = y.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "grad is None"
        if x.grad.shape != (2, 1):
            return False, f"grad shape = {x.grad.shape}"
        # Each element was expanded 5 times, so gradient is 5
        if not np.allclose(x.grad, [[5], [5]]):
            return False, f"grad = {x.grad}"
        return True, "Expand gradient sums"
    except Exception as e:
        return False, str(e)


def test_permute_simple() -> Tuple[bool, str]:
    """Test simple permute."""
    try:
        x = Tensor(np.arange(6).reshape(2, 3))
        y = x.permute(1, 0)  # Transpose
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (3, 2):
            return False, f"shape = {y.shape}"
        return True, "(2,3) permute(1,0) = (3,2)"
    except Exception as e:
        return False, str(e)


def test_permute_3d() -> Tuple[bool, str]:
    """Test 3D permute."""
    try:
        x = Tensor(np.arange(24).reshape(2, 3, 4))
        y = x.permute(2, 0, 1)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (4, 2, 3):
            return False, f"shape = {y.shape}"
        return True, "(2,3,4) permute(2,0,1) = (4,2,3)"
    except Exception as e:
        return False, str(e)


def test_permute_gradient() -> Tuple[bool, str]:
    """Test permute gradient."""
    try:
        x = Tensor(np.ones((2, 3, 4)))
        y = x.permute(1, 2, 0)
        loss = y.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "grad is None"
        if x.grad.shape != (2, 3, 4):
            return False, f"grad shape = {x.grad.shape}"
        return True, "Permute gradient OK"
    except Exception as e:
        return False, str(e)


def test_view_alias() -> Tuple[bool, str]:
    """Test view as reshape alias."""
    try:
        x = Tensor(np.arange(12))
        y = x.view(3, 4)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (3, 4):
            return False, f"shape = {y.shape}"
        return True, "view works as reshape"
    except Exception as e:
        return False, str(e)


def test_chained_operations() -> Tuple[bool, str]:
    """Test chained reshape operations."""
    try:
        x = Tensor(np.arange(24).reshape(2, 3, 4).astype(float))
        y = x.flatten().reshape(4, 6).permute(1, 0)  # (6, 4)
        loss = y.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "grad is None"
        if x.grad.shape != (2, 3, 4):
            return False, f"grad shape = {x.grad.shape}"
        if not np.allclose(x.grad, np.ones((2, 3, 4))):
            return False, "grad values incorrect"
        return True, "Chained ops gradient OK"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test against PyTorch reference."""
    try:
        import torch
        
        np_x = np.arange(12).reshape(3, 4).astype(float)
        
        # Our implementation
        x = Tensor(np_x)
        y = x.reshape(4, 3)
        z = (y * 2).sum()
        z.backward()
        
        # PyTorch
        tx = torch.tensor(np_x, requires_grad=True)
        ty = tx.reshape(4, 3)
        tz = (ty * 2).sum()
        tz.backward()
        
        if not np.allclose(y.data, ty.detach().numpy()):
            return False, "Forward mismatch"
        if not np.allclose(x.grad, tx.grad.numpy()):
            return False, "Gradient mismatch"
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def test_squeeze_unsqueeze_inverse() -> Tuple[bool, str]:
    """Test squeeze and unsqueeze are inverses."""
    try:
        x = Tensor(np.arange(6).reshape(2, 3))
        y = x.unsqueeze(0).unsqueeze(2)  # (1, 2, 1, 3)
        z = y.squeeze()  # Back to (2, 3)
        
        if z is None or z.data is None:
            return False, "Returned None"
        if z.shape != (2, 3):
            return False, f"shape = {z.shape}"
        if not np.allclose(z.data, x.data):
            return False, "Values changed"
        return True, "squeeze(unsqueeze(x)) = x"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("reshape_2d_to_1d", test_reshape_2d_to_1d),
        ("reshape_1d_to_2d", test_reshape_1d_to_2d),
        ("reshape_gradient_simple", test_reshape_gradient_simple),
        ("reshape_infer_dimension", test_reshape_infer_dimension),
        ("reshape_3d", test_reshape_3d),
        ("flatten_all", test_flatten_all),
        ("flatten_partial_batch", test_flatten_partial_batch),
        ("flatten_gradient", test_flatten_gradient),
        ("squeeze_all", test_squeeze_all),
        ("squeeze_specific_axis", test_squeeze_specific_axis),
        ("squeeze_gradient", test_squeeze_gradient),
        ("unsqueeze_dim0", test_unsqueeze_dim0),
        ("unsqueeze_dim_neg", test_unsqueeze_dim_neg),
        ("unsqueeze_gradient", test_unsqueeze_gradient),
        ("expand_row", test_expand_row),
        ("expand_col", test_expand_col),
        ("expand_gradient", test_expand_gradient),
        ("permute_simple", test_permute_simple),
        ("permute_3d", test_permute_3d),
        ("permute_gradient", test_permute_gradient),
        ("view_alias", test_view_alias),
        ("chained_operations", test_chained_operations),
        ("against_pytorch", test_against_pytorch),
        ("squeeze_unsqueeze_inverse", test_squeeze_unsqueeze_inverse),
    ]
    
    print(f"\n{'='*50}\nDay 17: Reshape and View Operations - Tests\n{'='*50}")
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
