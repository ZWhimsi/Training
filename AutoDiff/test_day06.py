"""Test Suite for Day 6: The Value Class"""

import pytest

from day06 import Value


def test_value_creation():
    v = Value(3.0)
    assert v.data == 3.0, f"data should be 3.0, got {v.data}"
    assert v.grad == 0.0, f"grad should be 0.0, got {v.grad}"


def test_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    
    assert c.data == 5.0, f"2 + 3 = {c.data}, expected 5"
    assert a in c._prev and b in c._prev, "Children not tracked"


def test_addition_scalar():
    a = Value(2.0)
    c = a + 5
    
    assert c.data == 7.0, f"2 + 5 = {c.data}, expected 7"
    
    d = 5 + a  # radd
    assert d.data == 7.0, "radd failed"


def test_multiplication():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    
    assert c.data == 6.0, f"2 * 3 = {c.data}, expected 6"


def test_negation():
    a = Value(5.0)
    b = -a
    
    assert b.data == -5.0, f"-5 = {b.data}"


def test_subtraction():
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    
    assert c.data == 2.0, f"5 - 3 = {c.data}, expected 2"


def test_power():
    a = Value(3.0)
    b = a ** 2
    
    assert b.data == 9.0, f"3^2 = {b.data}, expected 9"
    
    c = a ** 3
    assert c.data == 27.0, f"3^3 = {c.data}, expected 27"


def test_division():
    a = Value(6.0)
    b = Value(2.0)
    c = a / b
    
    assert abs(c.data - 3.0) <= 1e-6, f"6 / 2 = {c.data}, expected 3"


def test_complex_expression():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a ** 2  # 2*3 + 2^2 = 6 + 4 = 10
    
    assert c.data == 10.0, f"2*3 + 2^2 = {c.data}, expected 10"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
