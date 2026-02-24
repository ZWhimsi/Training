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
        
        # Check add: [1,2,3] + [2,2,2] = [3,4,5]
        if result['add'] is None:
            return False, "add is None"
        expected_add = torch.tensor([3., 4., 5.])
        if not torch.equal(result['add'], expected_add):
            return False, f"add: got {result['add']}, expected {expected_add}"
        
        # Check sub: [1,2,3] - [2,2,2] = [-1,0,1]
        if result['sub'] is None:
            return False, "sub is None"
        expected_sub = torch.tensor([-1., 0., 1.])
        if not torch.equal(result['sub'], expected_sub):
            return False, f"sub: got {result['sub']}, expected {expected_sub}"
        
        # Check mul: [1,2,3] * [2,2,2] = [2,4,6]
        if result['mul'] is None:
            return False, "mul is None"
        expected_mul = torch.tensor([2., 4., 6.])
        if not torch.equal(result['mul'], expected_mul):
            return False, f"mul: got {result['mul']}, expected {expected_mul}"
        
        # Check div: [1,2,3] / [2,2,2] = [0.5,1,1.5]
        if result['div'] is None:
            return False, "div is None"
        expected_div = torch.tensor([0.5, 1., 1.5])
        if not torch.allclose(result['div'], expected_div):
            return False, f"div: got {result['div']}, expected {expected_div}"
        
        # Check pow: [1,2,3] ** [2,2,2] = [1,4,9]
        if result['pow'] is None:
            return False, "pow is None"
        expected_pow = torch.tensor([1., 4., 9.])
        if not torch.allclose(result['pow'], expected_pow):
            return False, f"pow: got {result['pow']}, expected {expected_pow}"
        
        return True, "All arithmetic ops OK"
    except Exception as e:
        return False, str(e)


def test_matrix_ops() -> Tuple[bool, str]:
    try:
        # Use fixed values for reproducible testing
        A = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])  # 3x2
        B = torch.tensor([[1., 2., 3.], [4., 5., 6.]])     # 2x3
        result = matrix_ops(A, B)
        
        # Check matmul: A @ B
        if result['matmul'] is None:
            return False, "matmul is None"
        expected_matmul = torch.tensor([[9., 12., 15.], [19., 26., 33.], [29., 40., 51.]])
        if not torch.allclose(result['matmul'], expected_matmul):
            return False, f"matmul: got {result['matmul']}, expected {expected_matmul}"
        
        # Check transpose: A.T should be 2x3
        if result['transpose'] is None:
            return False, "transpose is None"
        expected_transpose = torch.tensor([[1., 3., 5.], [2., 4., 6.]])
        if not torch.equal(result['transpose'], expected_transpose):
            return False, f"transpose: got {result['transpose']}, expected {expected_transpose}"
        
        # Check inner_product: sum of squares of A = 1+4+9+16+25+36 = 91
        if result['inner_product'] is None:
            return False, "inner_product is None"
        expected_inner = torch.tensor(91.)
        if not torch.allclose(result['inner_product'], expected_inner):
            return False, f"inner_product: got {result['inner_product']}, expected {expected_inner}"
        
        return True, "Matrix ops OK"
    except Exception as e:
        return False, str(e)


def test_broadcasting() -> Tuple[bool, str]:
    try:
        result = broadcasting_examples()
        
        # scalar_add: [[1,2],[3,4]] + 5 = [[6,7],[8,9]]
        if result['scalar_add'] is None:
            return False, "scalar_add is None"
        expected_scalar = torch.tensor([[6., 7.], [8., 9.]])
        if not torch.equal(result['scalar_add'], expected_scalar):
            return False, f"scalar_add: got {result['scalar_add']}, expected {expected_scalar}"
        
        # row_broadcast: ones(3,4) + [1,2,3,4] = [[2,3,4,5],[2,3,4,5],[2,3,4,5]]
        if result['row_broadcast'] is None:
            return False, "row_broadcast is None"
        expected_row = torch.tensor([[2., 3., 4., 5.], [2., 3., 4., 5.], [2., 3., 4., 5.]])
        if not torch.equal(result['row_broadcast'], expected_row):
            return False, f"row_broadcast: got {result['row_broadcast']}, expected {expected_row}"
        
        # col_broadcast: ones(3,4) + [[1],[2],[3]] = [[2,2,2,2],[3,3,3,3],[4,4,4,4]]
        if result['col_broadcast'] is None:
            return False, "col_broadcast is None"
        expected_col = torch.tensor([[2., 2., 2., 2.], [3., 3., 3., 3.], [4., 4., 4., 4.]])
        if not torch.equal(result['col_broadcast'], expected_col):
            return False, f"col_broadcast: got {result['col_broadcast']}, expected {expected_col}"
        
        # outer_product: [[1],[2],[3]] * [[1,2,3,4]] = [[1,2,3,4],[2,4,6,8],[3,6,9,12]]
        if result['outer_product'] is None:
            return False, "outer_product is None"
        expected_outer = torch.tensor([[1., 2., 3., 4.], [2., 4., 6., 8.], [3., 6., 9., 12.]])
        if not torch.equal(result['outer_product'], expected_outer):
            return False, f"outer_product: got {result['outer_product']}, expected {expected_outer}"
        
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
        
        # exp: exp([0,1,2]) = [1, e, e^2]
        if result['exp'] is None:
            return False, "exp is None"
        expected_exp = torch.exp(t)
        if not torch.allclose(result['exp'], expected_exp):
            return False, f"exp: got {result['exp']}, expected {expected_exp}"
        
        # log: log(|t| + eps)
        if result['log'] is None:
            return False, "log is None"
        expected_log = torch.log(torch.abs(t) + 1e-8)
        if not torch.allclose(result['log'], expected_log):
            return False, f"log: got {result['log']}, expected {expected_log}"
        
        # sqrt: sqrt(|t|) = [0, 1, sqrt(2)]
        if result['sqrt'] is None:
            return False, "sqrt is None"
        expected_sqrt = torch.sqrt(torch.abs(t))
        if not torch.allclose(result['sqrt'], expected_sqrt):
            return False, f"sqrt: got {result['sqrt']}, expected {expected_sqrt}"
        
        # sin: sin([0,1,2])
        if result['sin'] is None:
            return False, "sin is None"
        expected_sin = torch.sin(t)
        if not torch.allclose(result['sin'], expected_sin):
            return False, f"sin: got {result['sin']}, expected {expected_sin}"
        
        # clamped: clamp to [-1,1], but t=[0,1,2] so result=[0,1,1]
        if result['clamped'] is None:
            return False, "clamped is None"
        expected_clamped = torch.tensor([0., 1., 1.])
        if not torch.allclose(result['clamped'], expected_clamped):
            return False, f"clamped: got {result['clamped']}, expected {expected_clamped}"
        
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
