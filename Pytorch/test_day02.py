"""Test Suite for Day 2: Tensor Operations"""

import torch
import pytest

from day02 import arithmetic_ops, matrix_ops, broadcasting_examples, reductions, math_functions


def test_arithmetic():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([2.0, 2.0, 2.0])
    result = arithmetic_ops(a, b)
    
    assert result['add'] is not None, "add is None"
    expected_add = torch.tensor([3., 4., 5.])
    assert torch.equal(result['add'], expected_add), f"add: got {result['add']}, expected {expected_add}"
    
    assert result['sub'] is not None, "sub is None"
    expected_sub = torch.tensor([-1., 0., 1.])
    assert torch.equal(result['sub'], expected_sub), f"sub: got {result['sub']}, expected {expected_sub}"
    
    assert result['mul'] is not None, "mul is None"
    expected_mul = torch.tensor([2., 4., 6.])
    assert torch.equal(result['mul'], expected_mul), f"mul: got {result['mul']}, expected {expected_mul}"
    
    assert result['div'] is not None, "div is None"
    expected_div = torch.tensor([0.5, 1., 1.5])
    assert torch.allclose(result['div'], expected_div), f"div: got {result['div']}, expected {expected_div}"
    
    assert result['pow'] is not None, "pow is None"
    expected_pow = torch.tensor([1., 4., 9.])
    assert torch.allclose(result['pow'], expected_pow), f"pow: got {result['pow']}, expected {expected_pow}"


def test_matrix_ops():
    A = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    B = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    result = matrix_ops(A, B)
    
    assert result['matmul'] is not None, "matmul is None"
    expected_matmul = torch.tensor([[9., 12., 15.], [19., 26., 33.], [29., 40., 51.]])
    assert torch.allclose(result['matmul'], expected_matmul), f"matmul: got {result['matmul']}, expected {expected_matmul}"
    
    assert result['transpose'] is not None, "transpose is None"
    expected_transpose = torch.tensor([[1., 3., 5.], [2., 4., 6.]])
    assert torch.equal(result['transpose'], expected_transpose), f"transpose: got {result['transpose']}, expected {expected_transpose}"
    
    assert result['inner_product'] is not None, "inner_product is None"
    expected_inner = torch.tensor(91.)
    assert torch.allclose(result['inner_product'], expected_inner), f"inner_product: got {result['inner_product']}, expected {expected_inner}"


def test_broadcasting():
    result = broadcasting_examples()
    
    assert result['scalar_add'] is not None, "scalar_add is None"
    expected_scalar = torch.tensor([[6., 7.], [8., 9.]])
    assert torch.equal(result['scalar_add'], expected_scalar), f"scalar_add: got {result['scalar_add']}, expected {expected_scalar}"
    
    assert result['row_broadcast'] is not None, "row_broadcast is None"
    expected_row = torch.tensor([[2., 3., 4., 5.], [2., 3., 4., 5.], [2., 3., 4., 5.]])
    assert torch.equal(result['row_broadcast'], expected_row), f"row_broadcast: got {result['row_broadcast']}, expected {expected_row}"
    
    assert result['col_broadcast'] is not None, "col_broadcast is None"
    expected_col = torch.tensor([[2., 2., 2., 2.], [3., 3., 3., 3.], [4., 4., 4., 4.]])
    assert torch.equal(result['col_broadcast'], expected_col), f"col_broadcast: got {result['col_broadcast']}, expected {expected_col}"
    
    assert result['outer_product'] is not None, "outer_product is None"
    expected_outer = torch.tensor([[1., 2., 3., 4.], [2., 4., 6., 8.], [3., 6., 9., 12.]])
    assert torch.equal(result['outer_product'], expected_outer), f"outer_product: got {result['outer_product']}, expected {expected_outer}"


def test_reductions():
    t = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    result = reductions(t)
    
    assert result['total_sum'] is not None and result['total_sum'] == 21, f"total_sum: expected 21, got {result['total_sum']}"
    assert result['row_sum'] is not None and torch.equal(result['row_sum'], torch.tensor([5., 7., 9.])), "row_sum wrong"
    assert result['col_sum'] is not None and torch.equal(result['col_sum'], torch.tensor([6., 15.])), "col_sum wrong"
    assert result['max_val'] is not None and result['max_val'] == 6, "max_val wrong"


def test_math_functions():
    t = torch.tensor([0., 1., 2.])
    result = math_functions(t)
    
    assert result['exp'] is not None, "exp is None"
    expected_exp = torch.exp(t)
    assert torch.allclose(result['exp'], expected_exp), f"exp: got {result['exp']}, expected {expected_exp}"
    
    assert result['log'] is not None, "log is None"
    expected_log = torch.log(torch.abs(t) + 1e-8)
    assert torch.allclose(result['log'], expected_log), f"log: got {result['log']}, expected {expected_log}"
    
    assert result['sqrt'] is not None, "sqrt is None"
    expected_sqrt = torch.sqrt(torch.abs(t))
    assert torch.allclose(result['sqrt'], expected_sqrt), f"sqrt: got {result['sqrt']}, expected {expected_sqrt}"
    
    assert result['sin'] is not None, "sin is None"
    expected_sin = torch.sin(t)
    assert torch.allclose(result['sin'], expected_sin), f"sin: got {result['sin']}, expected {expected_sin}"
    
    assert result['clamped'] is not None, "clamped is None"
    expected_clamped = torch.tensor([0., 1., 1.])
    assert torch.allclose(result['clamped'], expected_clamped), f"clamped: got {result['clamped']}, expected {expected_clamped}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
