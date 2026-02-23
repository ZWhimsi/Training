"""Test Suite for Day 2: Tensor Operations"""

import torch
import sys
from typing import Tuple

try:
    from day02 import arithmetic_ops, matrix_ops, broadcasting_examples, reductions, math_functions
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_arithmetic() -> Tuple[bool, str]:
    try:
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([2.0, 2.0, 2.0])
        result = arithmetic_ops(a, b)
        
        if result['add'] is None or not torch.equal(result['add'], torch.tensor([3., 4., 5.])):
            return False, "Addition wrong"
        if result['mul'] is None or not torch.equal(result['mul'], torch.tensor([2., 4., 6.])):
            return False, "Multiplication wrong"
        if result['pow'] is None or not torch.allclose(result['pow'], torch.tensor([1., 4., 9.])):
            return False, "Power wrong"
        return True, "All arithmetic ops OK"
    except Exception as e:
        return False, str(e)


def test_matrix_ops() -> Tuple[bool, str]:
    try:
        A = torch.randn(3, 4)
        B = torch.randn(4, 5)
        result = matrix_ops(A, B)
        
        if result['matmul'] is None or result['matmul'].shape != (3, 5):
            return False, "Matmul shape wrong"
        if result['transpose'] is None or result['transpose'].shape != (4, 3):
            return False, "Transpose shape wrong"
        if result['inner_product'] is None:
            return False, "Inner product is None"
        return True, "Matrix ops OK"
    except Exception as e:
        return False, str(e)


def test_broadcasting() -> Tuple[bool, str]:
    try:
        result = broadcasting_examples()
        
        if result['scalar_add'] is None:
            return False, "scalar_add is None"
        if not torch.equal(result['scalar_add'], torch.tensor([[6., 7.], [8., 9.]])):
            return False, "scalar_add wrong"
        
        if result['row_broadcast'] is None or result['row_broadcast'].shape != (3, 4):
            return False, "row_broadcast wrong"
        
        if result['outer_product'] is None or result['outer_product'].shape != (3, 4):
            return False, "outer_product shape wrong"
        
        return True, "Broadcasting OK"
    except Exception as e:
        return False, str(e)


def test_reductions() -> Tuple[bool, str]:
    try:
        t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        result = reductions(t)
        
        if result['total_sum'] is None or result['total_sum'] != 21:
            return False, f"total_sum: expected 21, got {result['total_sum']}"
        if result['row_sum'] is None or not torch.equal(result['row_sum'], torch.tensor([5., 7., 9.])):
            return False, "row_sum wrong"
        if result['col_sum'] is None or not torch.equal(result['col_sum'], torch.tensor([6., 15.])):
            return False, "col_sum wrong"
        if result['max_val'] is None or result['max_val'] != 6:
            return False, "max_val wrong"
        
        return True, "Reductions OK"
    except Exception as e:
        return False, str(e)


def test_math_functions() -> Tuple[bool, str]:
    try:
        t = torch.tensor([0., 1., 2.])
        result = math_functions(t)
        
        if result['exp'] is None:
            return False, "exp is None"
        if not torch.allclose(result['exp'], torch.exp(t)):
            return False, "exp wrong"
        
        if result['sin'] is None:
            return False, "sin is None"
        
        if result['clamped'] is None:
            return False, "clamped is None"
        
        return True, "Math functions OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("arithmetic", test_arithmetic),
        ("matrix_ops", test_matrix_ops),
        ("broadcasting", test_broadcasting),
        ("reductions", test_reductions),
        ("math_functions", test_math_functions),
    ]
    
    print(f"\n{'='*50}\nDay 2: Tensor Operations - Tests\n{'='*50}")
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
