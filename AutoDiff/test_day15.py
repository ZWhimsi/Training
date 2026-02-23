"""Test Suite for Day 15: Sum and Mean Reductions"""

import numpy as np
import sys
from typing import Tuple

try:
    from day15 import (
        Tensor,
        test_sum_all,
        test_sum_axis,
        test_sum_keepdims,
        test_mean_all,
        test_mean_axis,
        test_chain_rule_reduction,
        test_loss_function
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_sum_all_forward() -> Tuple[bool, str]:
    """Test sum of all elements forward."""
    try:
        x = Tensor([[1, 2], [3, 4]])
        y = x.sum()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, 10):
            return False, f"sum = {y.data}, expected 10"
        return True, "sum([[1,2],[3,4]]) = 10"
    except Exception as e:
        return False, str(e)


def test_sum_all_backward() -> Tuple[bool, str]:
    """Test sum of all elements backward."""
    try:
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = x.sum()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        if not np.allclose(x.grad, np.ones((2, 2))):
            return False, f"grad = {x.grad}"
        return True, "d(sum)/dx = 1"
    except Exception as e:
        return False, str(e)


def test_sum_axis0_forward() -> Tuple[bool, str]:
    """Test sum along axis 0."""
    try:
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x.sum(axis=0)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (3,):
            return False, f"shape = {y.shape}"
        if not np.allclose(y.data, [5, 7, 9]):
            return False, f"result = {y.data}"
        return True, "sum(axis=0) = [5,7,9]"
    except Exception as e:
        return False, str(e)


def test_sum_axis1_forward() -> Tuple[bool, str]:
    """Test sum along axis 1."""
    try:
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x.sum(axis=1)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (2,):
            return False, f"shape = {y.shape}"
        if not np.allclose(y.data, [6, 15]):
            return False, f"result = {y.data}"
        return True, "sum(axis=1) = [6,15]"
    except Exception as e:
        return False, str(e)


def test_sum_axis_backward() -> Tuple[bool, str]:
    """Test sum along axis backward."""
    try:
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = x.sum(axis=0)
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        if not np.allclose(x.grad, np.ones((2, 2))):
            return False, f"grad = {x.grad}"
        return True, "axis sum gradient OK"
    except Exception as e:
        return False, str(e)


def test_sum_keepdims() -> Tuple[bool, str]:
    """Test sum with keepdims."""
    try:
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        y = x.sum(axis=1, keepdims=True)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (2, 1):
            return False, f"shape = {y.shape}, expected (2,1)"
        if not np.allclose(y.data, [[6], [15]]):
            return False, f"values = {y.data}"
        return True, "keepdims=True works"
    except Exception as e:
        return False, str(e)


def test_mean_all_forward() -> Tuple[bool, str]:
    """Test mean of all elements."""
    try:
        x = Tensor([[1, 2], [3, 4]])
        y = x.mean()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, 2.5):
            return False, f"mean = {y.data}, expected 2.5"
        return True, "mean([[1,2],[3,4]]) = 2.5"
    except Exception as e:
        return False, str(e)


def test_mean_all_backward() -> Tuple[bool, str]:
    """Test mean backward pass."""
    try:
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        y = x.mean()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        # Each element contributes 1/4 to the mean
        if not np.allclose(x.grad, np.full((2, 2), 0.25)):
            return False, f"grad = {x.grad}"
        return True, "d(mean)/dx = 1/n"
    except Exception as e:
        return False, str(e)


def test_mean_axis_forward() -> Tuple[bool, str]:
    """Test mean along axis."""
    try:
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = x.mean(axis=0)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, [2.5, 3.5, 4.5]):
            return False, f"mean = {y.data}"
        return True, "mean(axis=0) = [2.5,3.5,4.5]"
    except Exception as e:
        return False, str(e)


def test_mean_axis_backward() -> Tuple[bool, str]:
    """Test mean along axis backward."""
    try:
        x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        y = x.mean(axis=0)  # Mean over 3 rows
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        # Each element contributes 1/3
        if not np.allclose(x.grad, np.full((3, 2), 1/3)):
            return False, f"grad = {x.grad}"
        return True, "axis mean gradient = 1/n"
    except Exception as e:
        return False, str(e)


def test_sum_chain_rule() -> Tuple[bool, str]:
    """Test chain rule with sum."""
    try:
        x = Tensor([1.0, 2.0, 3.0])
        y = (x ** 2).sum()  # 1 + 4 + 9 = 14
        y.backward()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, 14):
            return False, f"sum = {y.data}"
        # d(sum(x^2))/dx = 2x
        if not np.allclose(x.grad, [2, 4, 6]):
            return False, f"grad = {x.grad}"
        return True, "d(sum(x²))/dx = 2x"
    except Exception as e:
        return False, str(e)


def test_mean_chain_rule() -> Tuple[bool, str]:
    """Test chain rule with mean."""
    try:
        x = Tensor([1.0, 2.0, 3.0])
        y = (x ** 2).mean()  # (1 + 4 + 9) / 3 = 14/3
        y.backward()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, 14/3):
            return False, f"mean = {y.data}"
        # d(mean(x^2))/dx = 2x/n = [2/3, 4/3, 6/3]
        if not np.allclose(x.grad, [2/3, 4/3, 2]):
            return False, f"grad = {x.grad}"
        return True, "d(mean(x²))/dx = 2x/n"
    except Exception as e:
        return False, str(e)


def test_mse_loss() -> Tuple[bool, str]:
    """Test MSE loss computation."""
    try:
        pred = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.0, 2.0, 3.0])  # Perfect prediction
        
        mse = ((pred - target) ** 2).mean()
        
        if mse is None or mse.data is None:
            return False, "Returned None"
        if not np.allclose(mse.data, 0):
            return False, f"MSE = {mse.data}, expected 0"
        
        mse.backward()
        if not np.allclose(pred.grad, [0, 0, 0]):
            return False, f"grad = {pred.grad}"
        return True, "MSE=0 for perfect prediction"
    except Exception as e:
        return False, str(e)


def test_mse_gradient() -> Tuple[bool, str]:
    """Test MSE loss gradient."""
    try:
        pred = Tensor([2.0, 3.0, 4.0])
        target = Tensor([1.0, 2.0, 3.0])  # Errors of [1, 1, 1]
        
        mse = ((pred - target) ** 2).mean()  # MSE = 1
        
        if not np.allclose(mse.data, 1):
            return False, f"MSE = {mse.data}, expected 1"
        
        mse.backward()
        # d(MSE)/d(pred) = 2*(pred-target)/n = 2*[1,1,1]/3 = [2/3, 2/3, 2/3]
        expected_grad = np.array([2/3, 2/3, 2/3])
        if not np.allclose(pred.grad, expected_grad):
            return False, f"grad = {pred.grad}, expected {expected_grad}"
        return True, "MSE gradient correct"
    except Exception as e:
        return False, str(e)


def test_sum_3d() -> Tuple[bool, str]:
    """Test sum on 3D tensor."""
    try:
        x = Tensor(np.ones((2, 3, 4)))
        y = x.sum(axis=1)  # Sum along middle axis -> (2, 4)
        
        if y is None or y.data is None:
            return False, "Returned None"
        if y.shape != (2, 4):
            return False, f"shape = {y.shape}"
        if not np.allclose(y.data, 3 * np.ones((2, 4))):
            return False, f"values wrong"
        
        y.backward()
        if not np.allclose(x.grad, np.ones((2, 3, 4))):
            return False, f"grad shape/values wrong"
        return True, "3D sum works"
    except Exception as e:
        return False, str(e)


def test_max_forward() -> Tuple[bool, str]:
    """Test max forward pass."""
    try:
        x = Tensor([[1, 5, 3], [4, 2, 6]])
        y = x.max()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, 6):
            return False, f"max = {y.data}"
        return True, "max = 6"
    except Exception as e:
        return False, str(e)


def test_max_backward() -> Tuple[bool, str]:
    """Test max backward pass."""
    try:
        x = Tensor([1.0, 5.0, 3.0])
        y = x.max()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        # Only max element (5 at index 1) should have gradient
        if not np.allclose(x.grad, [0, 1, 0]):
            return False, f"grad = {x.grad}"
        return True, "Only max gets gradient"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test against PyTorch reference."""
    try:
        import torch
        
        np_x = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Our implementation
        x = Tensor(np_x)
        y = (x ** 2).sum()
        y.backward()
        
        # PyTorch
        tx = torch.tensor(np_x, requires_grad=True)
        ty = (tx ** 2).sum()
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


def test_variance() -> Tuple[bool, str]:
    """Test variance computation."""
    try:
        x = Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        var = x.var()
        
        if var is None or var.data is None:
            return False, "Returned None"
        
        # var = mean((x - mean)^2) = mean of [4, 1, 0, 1, 4] = 2
        expected_var = 2.0
        if not np.allclose(var.data, expected_var):
            return False, f"var = {var.data}, expected {expected_var}"
        return True, "var([1,2,3,4,5]) = 2"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("sum_all_forward", test_sum_all_forward),
        ("sum_all_backward", test_sum_all_backward),
        ("sum_axis0_forward", test_sum_axis0_forward),
        ("sum_axis1_forward", test_sum_axis1_forward),
        ("sum_axis_backward", test_sum_axis_backward),
        ("sum_keepdims", test_sum_keepdims),
        ("mean_all_forward", test_mean_all_forward),
        ("mean_all_backward", test_mean_all_backward),
        ("mean_axis_forward", test_mean_axis_forward),
        ("mean_axis_backward", test_mean_axis_backward),
        ("sum_chain_rule", test_sum_chain_rule),
        ("mean_chain_rule", test_mean_chain_rule),
        ("mse_loss", test_mse_loss),
        ("mse_gradient", test_mse_gradient),
        ("sum_3d", test_sum_3d),
        ("max_forward", test_max_forward),
        ("max_backward", test_max_backward),
        ("against_pytorch", test_against_pytorch),
        ("variance", test_variance),
    ]
    
    print(f"\n{'='*50}\nDay 15: Sum and Mean Reductions - Tests\n{'='*50}")
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
