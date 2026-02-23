"""Test Suite for Day 6: The Value Class"""

import sys
from typing import Tuple

try:
    from day06 import Value
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_value_creation() -> Tuple[bool, str]:
    try:
        v = Value(3.0)
        if v.data != 3.0:
            return False, f"data should be 3.0, got {v.data}"
        if v.grad != 0.0:
            return False, f"grad should be 0.0, got {v.grad}"
        return True, "Value creation OK"
    except Exception as e:
        return False, str(e)


def test_addition() -> Tuple[bool, str]:
    try:
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        
        if c.data != 5.0:
            return False, f"2 + 3 = {c.data}, expected 5"
        if a not in c._prev and b not in c._prev:
            return False, "Children not tracked"
        return True, "2 + 3 = 5"
    except Exception as e:
        return False, str(e)


def test_addition_scalar() -> Tuple[bool, str]:
    try:
        a = Value(2.0)
        c = a + 5
        
        if c.data != 7.0:
            return False, f"2 + 5 = {c.data}, expected 7"
        
        d = 5 + a  # radd
        if d.data != 7.0:
            return False, "radd failed"
        
        return True, "Scalar addition OK"
    except Exception as e:
        return False, str(e)


def test_multiplication() -> Tuple[bool, str]:
    try:
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        
        if c.data != 6.0:
            return False, f"2 * 3 = {c.data}, expected 6"
        return True, "2 * 3 = 6"
    except Exception as e:
        return False, str(e)


def test_negation() -> Tuple[bool, str]:
    try:
        a = Value(5.0)
        b = -a
        
        if b.data != -5.0:
            return False, f"-5 = {b.data}"
        return True, "-5 = -5"
    except Exception as e:
        return False, str(e)


def test_subtraction() -> Tuple[bool, str]:
    try:
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        
        if c.data != 2.0:
            return False, f"5 - 3 = {c.data}, expected 2"
        return True, "5 - 3 = 2"
    except Exception as e:
        return False, str(e)


def test_power() -> Tuple[bool, str]:
    try:
        a = Value(3.0)
        b = a ** 2
        
        if b.data != 9.0:
            return False, f"3^2 = {b.data}, expected 9"
        
        c = a ** 3
        if c.data != 27.0:
            return False, f"3^3 = {c.data}, expected 27"
        
        return True, "3^2=9, 3^3=27"
    except Exception as e:
        return False, str(e)


def test_division() -> Tuple[bool, str]:
    try:
        a = Value(6.0)
        b = Value(2.0)
        c = a / b
        
        if abs(c.data - 3.0) > 1e-6:
            return False, f"6 / 2 = {c.data}, expected 3"
        return True, "6 / 2 = 3"
    except Exception as e:
        return False, str(e)


def test_complex_expression() -> Tuple[bool, str]:
    try:
        a = Value(2.0)
        b = Value(3.0)
        c = a * b + a ** 2  # 2*3 + 2^2 = 6 + 4 = 10
        
        if c.data != 10.0:
            return False, f"2*3 + 2^2 = {c.data}, expected 10"
        return True, "2*3 + 2^2 = 10"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("creation", test_value_creation),
        ("addition", test_addition),
        ("addition_scalar", test_addition_scalar),
        ("multiplication", test_multiplication),
        ("negation", test_negation),
        ("subtraction", test_subtraction),
        ("power", test_power),
        ("division", test_division),
        ("complex_expr", test_complex_expression),
    ]
    
    print(f"\n{'='*50}\nDay 6: Value Class - Tests\n{'='*50}")
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
