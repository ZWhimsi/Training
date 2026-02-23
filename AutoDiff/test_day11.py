"""Test Suite for Day 11: Backward Pass"""

import math
import sys
from typing import Tuple

try:
    from day11 import Value, gradient_check
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_simple_backward() -> Tuple[bool, str]:
    try:
        x = Value(3.0)
        y = x ** 2
        y.backward()
        
        # d/dx(x^2) = 2x = 6 at x=3
        if abs(x.grad - 6.0) > 1e-5:
            return False, f"x.grad = {x.grad}, expected 6.0"
        return True, "d/dx(x^2)|_{x=3} = 6"
    except Exception as e:
        return False, str(e)


def test_addition_backward() -> Tuple[bool, str]:
    try:
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        
        # d(a+b)/da = 1, d(a+b)/db = 1
        if abs(a.grad - 1.0) > 1e-5 or abs(b.grad - 1.0) > 1e-5:
            return False, f"Grads: a={a.grad}, b={b.grad}, expected 1, 1"
        return True, "d(a+b)/da = d(a+b)/db = 1"
    except Exception as e:
        return False, str(e)


def test_multiplication_backward() -> Tuple[bool, str]:
    try:
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        
        # d(a*b)/da = b = 3, d(a*b)/db = a = 2
        if abs(a.grad - 3.0) > 1e-5:
            return False, f"a.grad = {a.grad}, expected 3.0"
        if abs(b.grad - 2.0) > 1e-5:
            return False, f"b.grad = {b.grad}, expected 2.0"
        return True, "d(a*b)/da = b, d(a*b)/db = a"
    except Exception as e:
        return False, str(e)


def test_chain_rule() -> Tuple[bool, str]:
    try:
        x = Value(2.0)
        y = (x * 2 + 1) ** 2  # (2*2+1)^2 = 25
        y.backward()
        
        # y = (2x + 1)^2
        # dy/dx = 2 * (2x + 1) * 2 = 4 * (2*2+1) = 20
        if abs(y.data - 25.0) > 1e-5:
            return False, f"y.data = {y.data}, expected 25"
        if abs(x.grad - 20.0) > 1e-5:
            return False, f"x.grad = {x.grad}, expected 20"
        return True, "Chain rule: dy/dx = 20"
    except Exception as e:
        return False, str(e)


def test_relu_backward() -> Tuple[bool, str]:
    try:
        # Positive input
        x1 = Value(3.0)
        y1 = x1.relu()
        y1.backward()
        
        if abs(y1.data - 3.0) > 1e-5:
            return False, f"relu(3) = {y1.data}, expected 3"
        if abs(x1.grad - 1.0) > 1e-5:
            return False, f"d relu(3)/dx = {x1.grad}, expected 1"
        
        # Negative input
        x2 = Value(-3.0)
        y2 = x2.relu()
        y2.backward()
        
        if abs(y2.data - 0.0) > 1e-5:
            return False, f"relu(-3) = {y2.data}, expected 0"
        if abs(x2.grad - 0.0) > 1e-5:
            return False, f"d relu(-3)/dx = {x2.grad}, expected 0"
        
        return True, "ReLU: pos→1, neg→0"
    except Exception as e:
        return False, str(e)


def test_exp_backward() -> Tuple[bool, str]:
    try:
        x = Value(1.0)
        y = x.exp()
        y.backward()
        
        expected = math.exp(1.0)
        if abs(y.data - expected) > 1e-5:
            return False, f"exp(1) = {y.data}, expected {expected}"
        if abs(x.grad - expected) > 1e-5:
            return False, f"d exp(1)/dx = {x.grad}, expected {expected}"
        return True, f"exp(1) = e, d/dx = e"
    except Exception as e:
        return False, str(e)


def test_tanh_backward() -> Tuple[bool, str]:
    try:
        x = Value(0.5)
        y = x.tanh()
        y.backward()
        
        expected_y = math.tanh(0.5)
        expected_grad = 1 - expected_y ** 2
        
        if abs(y.data - expected_y) > 1e-5:
            return False, f"tanh(0.5) = {y.data}, expected {expected_y}"
        if abs(x.grad - expected_grad) > 1e-4:
            return False, f"d tanh/dx = {x.grad}, expected {expected_grad}"
        return True, "tanh and grad correct"
    except Exception as e:
        return False, str(e)


def test_gradient_accumulation() -> Tuple[bool, str]:
    try:
        # When a value is used multiple times, gradients should accumulate
        x = Value(2.0)
        y = x + x  # x used twice
        y.backward()
        
        # d(x + x)/dx = 2
        if abs(x.grad - 2.0) > 1e-5:
            return False, f"d(x+x)/dx = {x.grad}, expected 2"
        
        # Reset and test multiplication
        x2 = Value(3.0)
        y2 = x2 * x2  # x^2
        y2.backward()
        
        # d(x^2)/dx = 2x = 6
        if abs(x2.grad - 6.0) > 1e-5:
            return False, f"d(x*x)/dx = {x2.grad}, expected 6"
        
        return True, "Gradient accumulation works"
    except Exception as e:
        return False, str(e)


def test_complex_expression() -> Tuple[bool, str]:
    try:
        # Test a complex expression
        x = Value(2.0)
        y = Value(3.0)
        z = (x * y + x ** 2).relu()
        z.backward()
        
        # z = relu(x*y + x^2) = relu(6 + 4) = 10
        # dz/dx = (y + 2x) * (1 if input>0 else 0) = (3 + 4) * 1 = 7
        # dz/dy = x * 1 = 2
        
        if abs(z.data - 10.0) > 1e-5:
            return False, f"z = {z.data}, expected 10"
        if abs(x.grad - 7.0) > 1e-5:
            return False, f"dz/dx = {x.grad}, expected 7"
        if abs(y.grad - 2.0) > 1e-5:
            return False, f"dz/dy = {y.grad}, expected 2"
        
        return True, "Complex expression OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("simple_backward", test_simple_backward),
        ("addition_backward", test_addition_backward),
        ("multiplication_backward", test_multiplication_backward),
        ("chain_rule", test_chain_rule),
        ("relu_backward", test_relu_backward),
        ("exp_backward", test_exp_backward),
        ("tanh_backward", test_tanh_backward),
        ("gradient_accumulation", test_gradient_accumulation),
        ("complex_expression", test_complex_expression),
    ]
    
    print(f"\n{'='*50}\nDay 11: Backward Pass - Tests\n{'='*50}")
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
