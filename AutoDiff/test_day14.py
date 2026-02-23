"""Test Suite for Day 14: Tensor Addition/Multiplication with Gradients"""

import numpy as np
import sys
from typing import Tuple

try:
    from day14 import (
        Tensor,
        test_subtraction,
        test_division,
        test_power,
        test_exp_log,
        test_complex_expression,
        test_broadcast_division
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_negation_forward() -> Tuple[bool, str]:
    """Test negation forward pass."""
    try:
        x = Tensor([1.0, -2.0, 3.0])
        y = -x
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, [-1, 2, -3]):
            return False, f"Got {y.data}"
        return True, "-[1,-2,3] = [-1,2,-3]"
    except Exception as e:
        return False, str(e)


def test_negation_backward() -> Tuple[bool, str]:
    """Test negation backward pass."""
    try:
        x = Tensor([1.0, 2.0])
        y = -x
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        if not np.allclose(x.grad, [-1, -1]):
            return False, f"Got {x.grad}"
        return True, "d(-x)/dx = -1"
    except Exception as e:
        return False, str(e)


def test_sub_forward() -> Tuple[bool, str]:
    """Test subtraction forward pass."""
    try:
        a = Tensor([5.0, 6.0])
        b = Tensor([1.0, 2.0])
        c = a - b
        
        if c is None or c.data is None:
            return False, "Returned None"
        if not np.allclose(c.data, [4, 4]):
            return False, f"Got {c.data}"
        return True, "[5,6] - [1,2] = [4,4]"
    except Exception as e:
        return False, str(e)


def test_sub_backward() -> Tuple[bool, str]:
    """Test subtraction backward pass."""
    try:
        a = Tensor([5.0, 6.0])
        b = Tensor([1.0, 2.0])
        c = a - b
        c.backward()
        
        if a.grad is None or b.grad is None:
            return False, "grad is None"
        if not np.allclose(a.grad, [1, 1]):
            return False, f"a.grad = {a.grad}"
        if not np.allclose(b.grad, [-1, -1]):
            return False, f"b.grad = {b.grad}"
        return True, "da=1, db=-1"
    except Exception as e:
        return False, str(e)


def test_rsub() -> Tuple[bool, str]:
    """Test reverse subtraction."""
    try:
        x = Tensor([1.0, 2.0])
        y = 5 - x
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, [4, 3]):
            return False, f"Got {y.data}"
        
        y.backward()
        if not np.allclose(x.grad, [-1, -1]):
            return False, f"grad = {x.grad}"
        return True, "5 - [1,2] = [4,3]"
    except Exception as e:
        return False, str(e)


def test_div_forward() -> Tuple[bool, str]:
    """Test division forward pass."""
    try:
        a = Tensor([6.0, 8.0])
        b = Tensor([2.0, 4.0])
        c = a / b
        
        if c is None or c.data is None:
            return False, "Returned None"
        if not np.allclose(c.data, [3, 2]):
            return False, f"Got {c.data}"
        return True, "[6,8] / [2,4] = [3,2]"
    except Exception as e:
        return False, str(e)


def test_div_backward() -> Tuple[bool, str]:
    """Test division backward pass."""
    try:
        a = Tensor([6.0])
        b = Tensor([2.0])
        c = a / b
        c.backward()
        
        if a.grad is None or b.grad is None:
            return False, "grad is None"
        # dc/da = 1/b = 0.5
        # dc/db = -a/b² = -6/4 = -1.5
        if not np.allclose(a.grad, [0.5]):
            return False, f"a.grad = {a.grad}, expected 0.5"
        if not np.allclose(b.grad, [-1.5]):
            return False, f"b.grad = {b.grad}, expected -1.5"
        return True, "da=1/b, db=-a/b²"
    except Exception as e:
        return False, str(e)


def test_rtruediv() -> Tuple[bool, str]:
    """Test reverse division."""
    try:
        x = Tensor([2.0, 4.0])
        y = 8 / x  # [4, 2]
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, [4, 2]):
            return False, f"Got {y.data}"
        
        y.backward()
        # d(8/x)/dx = -8/x² = [-2, -0.5]
        if not np.allclose(x.grad, [-2, -0.5]):
            return False, f"grad = {x.grad}"
        return True, "8 / [2,4] = [4,2]"
    except Exception as e:
        return False, str(e)


def test_pow_forward() -> Tuple[bool, str]:
    """Test power forward pass."""
    try:
        x = Tensor([2.0, 3.0])
        y = x ** 2
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, [4, 9]):
            return False, f"Got {y.data}"
        return True, "[2,3]² = [4,9]"
    except Exception as e:
        return False, str(e)


def test_pow_backward() -> Tuple[bool, str]:
    """Test power backward pass."""
    try:
        x = Tensor([2.0, 3.0])
        y = x ** 2
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        # d(x²)/dx = 2x = [4, 6]
        if not np.allclose(x.grad, [4, 6]):
            return False, f"Got {x.grad}"
        return True, "d(x²)/dx = 2x"
    except Exception as e:
        return False, str(e)


def test_pow_negative() -> Tuple[bool, str]:
    """Test power with negative exponent."""
    try:
        x = Tensor([2.0, 4.0])
        y = x ** -1  # [0.5, 0.25]
        y.backward()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, [0.5, 0.25]):
            return False, f"Got {y.data}"
        
        # d(x^-1)/dx = -x^-2 = [-0.25, -0.0625]
        if not np.allclose(x.grad, [-0.25, -0.0625]):
            return False, f"grad = {x.grad}"
        return True, "x^-1 works"
    except Exception as e:
        return False, str(e)


def test_exp_forward() -> Tuple[bool, str]:
    """Test exp forward pass."""
    try:
        x = Tensor([0.0, 1.0])
        y = x.exp()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, [1, np.e]):
            return False, f"Got {y.data}"
        return True, "exp([0,1]) = [1,e]"
    except Exception as e:
        return False, str(e)


def test_exp_backward() -> Tuple[bool, str]:
    """Test exp backward pass."""
    try:
        x = Tensor([0.0, 1.0])
        y = x.exp()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        # d(e^x)/dx = e^x
        if not np.allclose(x.grad, [1, np.e]):
            return False, f"Got {x.grad}"
        return True, "d(e^x)/dx = e^x"
    except Exception as e:
        return False, str(e)


def test_log_forward() -> Tuple[bool, str]:
    """Test log forward pass."""
    try:
        x = Tensor([1.0, np.e])
        y = x.log()
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, [0, 1]):
            return False, f"Got {y.data}"
        return True, "log([1,e]) = [0,1]"
    except Exception as e:
        return False, str(e)


def test_log_backward() -> Tuple[bool, str]:
    """Test log backward pass."""
    try:
        x = Tensor([1.0, 2.0])
        y = x.log()
        y.backward()
        
        if x.grad is None:
            return False, "grad is None"
        # d(ln(x))/dx = 1/x
        if not np.allclose(x.grad, [1, 0.5]):
            return False, f"Got {x.grad}"
        return True, "d(ln(x))/dx = 1/x"
    except Exception as e:
        return False, str(e)


def test_exp_log_chain() -> Tuple[bool, str]:
    """Test exp(log(x)) = x identity."""
    try:
        x = Tensor([2.0, 3.0])
        y = x.log().exp()  # Should be [2, 3]
        
        if y is None or y.data is None:
            return False, "Returned None"
        if not np.allclose(y.data, [2, 3]):
            return False, f"Got {y.data}"
        
        y.backward()
        # d/dx(exp(log(x))) = 1
        if not np.allclose(x.grad, [1, 1]):
            return False, f"grad = {x.grad}"
        return True, "exp(log(x)) = x"
    except Exception as e:
        return False, str(e)


def test_complex_expr() -> Tuple[bool, str]:
    """Test complex expression."""
    try:
        # f(x, y) = (x + y) * (x - y) = x² - y²
        x = Tensor([3.0])
        y = Tensor([2.0])
        f = (x + y) * (x - y)  # 5 * 1 = 5
        
        if f is None or f.data is None:
            return False, "Returned None"
        if not np.allclose(f.data, [5.0]):
            return False, f"f = {f.data}"
        
        f.backward()
        # df/dx = 2x = 6, df/dy = -2y = -4
        if not np.allclose(x.grad, [6.0]):
            return False, f"x.grad = {x.grad}"
        if not np.allclose(y.grad, [-4.0]):
            return False, f"y.grad = {y.grad}"
        return True, "(x+y)(x-y) gradient OK"
    except Exception as e:
        return False, str(e)


def test_sub_broadcast() -> Tuple[bool, str]:
    """Test subtraction with broadcasting."""
    try:
        a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        b = Tensor([1, 1, 1])                # (3,)
        c = a - b
        
        if c is None or c.data is None:
            return False, "Returned None"
        expected = np.array([[0, 1, 2], [3, 4, 5]])
        if not np.allclose(c.data, expected):
            return False, f"Got {c.data}"
        
        c.backward()
        # db = -1 summed over axis 0 = [-2, -2, -2]
        if not np.allclose(b.grad, [-2, -2, -2]):
            return False, f"b.grad = {b.grad}"
        return True, "Broadcast subtraction OK"
    except Exception as e:
        return False, str(e)


def test_div_broadcast() -> Tuple[bool, str]:
    """Test division with broadcasting."""
    try:
        a = Tensor([[2.0, 4.0], [6.0, 8.0]])  # (2, 2)
        b = Tensor([2.0, 2.0])                 # (2,)
        c = a / b
        
        if c is None or c.data is None:
            return False, "Returned None"
        expected = np.array([[1, 2], [3, 4]])
        if not np.allclose(c.data, expected):
            return False, f"Got {c.data}"
        
        c.backward()
        # da = 1/b = [0.5, 0.5]
        if not np.allclose(a.grad, 0.5 * np.ones((2, 2))):
            return False, f"a.grad = {a.grad}"
        
        return True, "Broadcast division OK"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test against PyTorch reference."""
    try:
        import torch
        
        # Complex expression: y = exp(x) / (x + 1)
        np_x = np.array([1.0, 2.0])
        
        # Our implementation
        x = Tensor(np_x)
        y = x.exp() / (x + 1)
        y.backward()
        
        # PyTorch reference
        tx = torch.tensor(np_x, requires_grad=True)
        ty = torch.exp(tx) / (tx + 1)
        ty.backward(torch.ones_like(ty))
        
        if not np.allclose(y.data, ty.detach().numpy()):
            return False, "Forward mismatch"
        if not np.allclose(x.grad, tx.grad.numpy(), rtol=1e-4):
            return False, "Gradient mismatch"
        
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("negation_forward", test_negation_forward),
        ("negation_backward", test_negation_backward),
        ("sub_forward", test_sub_forward),
        ("sub_backward", test_sub_backward),
        ("rsub", test_rsub),
        ("div_forward", test_div_forward),
        ("div_backward", test_div_backward),
        ("rtruediv", test_rtruediv),
        ("pow_forward", test_pow_forward),
        ("pow_backward", test_pow_backward),
        ("pow_negative", test_pow_negative),
        ("exp_forward", test_exp_forward),
        ("exp_backward", test_exp_backward),
        ("log_forward", test_log_forward),
        ("log_backward", test_log_backward),
        ("exp_log_chain", test_exp_log_chain),
        ("complex_expr", test_complex_expr),
        ("sub_broadcast", test_sub_broadcast),
        ("div_broadcast", test_div_broadcast),
        ("against_pytorch", test_against_pytorch),
    ]
    
    print(f"\n{'='*50}\nDay 14: Tensor Operations - Tests\n{'='*50}")
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
