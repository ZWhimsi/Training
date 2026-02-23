"""Test Suite for Day 8: Division and Subtraction Backward"""

import sys
from typing import Tuple

try:
    from day08 import (
        Value,
        verify_subtraction_gradients,
        verify_division_gradients,
        chain_rule_division
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_negation_forward() -> Tuple[bool, str]:
    """Test negation forward pass."""
    try:
        x = Value(5.0)
        y = -x
        
        if y is None:
            return False, "Negation returned None"
        if abs(y.data - (-5.0)) > 1e-6:
            return False, f"-5 = {y.data}, expected -5.0"
        return True, "-5 = -5.0"
    except Exception as e:
        return False, str(e)


def test_negation_backward() -> Tuple[bool, str]:
    """Test negation backward pass."""
    try:
        x = Value(5.0)
        y = -x
        y.backward()
        
        if abs(x.grad - (-1.0)) > 1e-6:
            return False, f"d(-x)/dx = {x.grad}, expected -1.0"
        return True, "d(-x)/dx = -1"
    except Exception as e:
        return False, str(e)


def test_subtraction_forward() -> Tuple[bool, str]:
    """Test subtraction forward pass."""
    try:
        a = Value(7.0)
        b = Value(3.0)
        c = a - b
        
        if c is None:
            return False, "Subtraction returned None"
        if abs(c.data - 4.0) > 1e-6:
            return False, f"7 - 3 = {c.data}, expected 4.0"
        return True, "7 - 3 = 4"
    except Exception as e:
        return False, str(e)


def test_subtraction_backward() -> Tuple[bool, str]:
    """Test subtraction backward pass."""
    try:
        a = Value(7.0)
        b = Value(3.0)
        c = a - b
        c.backward()
        
        if abs(a.grad - 1.0) > 1e-6:
            return False, f"d(a-b)/da = {a.grad}, expected 1.0"
        if abs(b.grad - (-1.0)) > 1e-6:
            return False, f"d(a-b)/db = {b.grad}, expected -1.0"
        return True, "d/da=1, d/db=-1"
    except Exception as e:
        return False, str(e)


def test_rsub() -> Tuple[bool, str]:
    """Test reverse subtraction (number - Value)."""
    try:
        x = Value(3.0)
        y = 10 - x  # Should be 7
        
        if y is None:
            return False, "Reverse subtraction returned None"
        if abs(y.data - 7.0) > 1e-6:
            return False, f"10 - 3 = {y.data}, expected 7.0"
        
        y.backward()
        if abs(x.grad - (-1.0)) > 1e-6:
            return False, f"d(10-x)/dx = {x.grad}, expected -1.0"
        return True, "10 - x works"
    except Exception as e:
        return False, str(e)


def test_division_forward() -> Tuple[bool, str]:
    """Test division forward pass."""
    try:
        a = Value(10.0)
        b = Value(2.0)
        c = a / b
        
        if c is None:
            return False, "Division returned None"
        if abs(c.data - 5.0) > 1e-6:
            return False, f"10 / 2 = {c.data}, expected 5.0"
        return True, "10 / 2 = 5"
    except Exception as e:
        return False, str(e)


def test_division_backward() -> Tuple[bool, str]:
    """Test division backward pass."""
    try:
        a = Value(6.0)
        b = Value(2.0)
        c = a / b  # = 3
        c.backward()
        
        # d(a/b)/da = 1/b = 0.5
        # d(a/b)/db = -a/b² = -6/4 = -1.5
        if abs(a.grad - 0.5) > 1e-6:
            return False, f"d(a/b)/da = {a.grad}, expected 0.5"
        if abs(b.grad - (-1.5)) > 1e-6:
            return False, f"d(a/b)/db = {b.grad}, expected -1.5"
        return True, "d/da=1/b, d/db=-a/b²"
    except Exception as e:
        return False, str(e)


def test_rtruediv() -> Tuple[bool, str]:
    """Test reverse division (number / Value)."""
    try:
        x = Value(4.0)
        y = 12 / x  # Should be 3
        
        if y is None:
            return False, "Reverse division returned None"
        if abs(y.data - 3.0) > 1e-6:
            return False, f"12 / 4 = {y.data}, expected 3.0"
        
        y.backward()
        # d(12/x)/dx = -12/x² = -12/16 = -0.75
        if abs(x.grad - (-0.75)) > 1e-6:
            return False, f"d(12/x)/dx = {x.grad}, expected -0.75"
        return True, "12 / x works"
    except Exception as e:
        return False, str(e)


def test_chain_rule_subtraction() -> Tuple[bool, str]:
    """Test chain rule with subtraction."""
    try:
        x = Value(2.0)
        # f(x) = (x^2 - x) at x=2
        # f(2) = 4 - 2 = 2
        # f'(x) = 2x - 1 = 3 at x=2
        y = x ** 2 - x
        y.backward()
        
        if abs(y.data - 2.0) > 1e-6:
            return False, f"x²-x at x=2 = {y.data}, expected 2"
        if abs(x.grad - 3.0) > 1e-6:
            return False, f"d(x²-x)/dx = {x.grad}, expected 3"
        return True, "f(x)=x²-x, f'(2)=3"
    except Exception as e:
        return False, str(e)


def test_chain_rule_division() -> Tuple[bool, str]:
    """Test chain rule with division."""
    try:
        result = chain_rule_division()
        
        if result['value'] is None:
            return False, "Result is None"
        if abs(result['value'] - 2.0) > 1e-6:
            return False, f"(x+1)/(x-1) = {result['value']}, expected 2"
        if abs(result['gradient'] - (-0.5)) > 1e-4:
            return False, f"gradient = {result['gradient']}, expected -0.5"
        return True, "(x+1)/(x-1), f'(3)=-0.5"
    except Exception as e:
        return False, str(e)


def test_complex_expression() -> Tuple[bool, str]:
    """Test complex expression with division and subtraction."""
    try:
        # f(x) = (2x - 1) / x at x = 2
        # f(2) = 3/2 = 1.5
        # f'(x) = [2x - (2x-1)] / x² = 1/x²
        # f'(2) = 1/4 = 0.25
        x = Value(2.0)
        y = (2 * x - 1) / x
        y.backward()
        
        if abs(y.data - 1.5) > 1e-6:
            return False, f"f(2) = {y.data}, expected 1.5"
        if abs(x.grad - 0.25) > 1e-4:
            return False, f"f'(2) = {x.grad}, expected 0.25"
        return True, "f(x)=(2x-1)/x"
    except Exception as e:
        return False, str(e)


def test_numerical_gradient_check() -> Tuple[bool, str]:
    """Verify against numerical gradient."""
    try:
        def f(val):
            x = Value(val)
            return ((x + 1) / (x - 1)).data
        
        x_val = 3.0
        h = 1e-5
        numerical = (f(x_val + h) - f(x_val - h)) / (2 * h)
        
        x = Value(x_val)
        y = (x + 1) / (x - 1)
        y.backward()
        analytical = x.grad
        
        if abs(analytical - numerical) > 1e-4:
            return False, f"Analytical={analytical}, numerical={numerical}"
        return True, "Matches numerical gradient"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("negation_forward", test_negation_forward),
        ("negation_backward", test_negation_backward),
        ("subtraction_forward", test_subtraction_forward),
        ("subtraction_backward", test_subtraction_backward),
        ("rsub", test_rsub),
        ("division_forward", test_division_forward),
        ("division_backward", test_division_backward),
        ("rtruediv", test_rtruediv),
        ("chain_rule_subtraction", test_chain_rule_subtraction),
        ("chain_rule_division", test_chain_rule_division),
        ("complex_expression", test_complex_expression),
        ("numerical_gradient_check", test_numerical_gradient_check),
    ]
    
    print(f"\n{'='*50}\nDay 8: Division & Subtraction - Tests\n{'='*50}")
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
