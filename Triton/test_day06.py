"""
Test Suite for Day 6: Multi-dimensional Grids
=============================================
Run: python test_day06.py
"""

import torch
import sys
from typing import Tuple

try:
    from day06 import elementwise_2d, batch_matvec, batch_add_3d, sum_axis0, broadcast_add
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


# References
def ref_elementwise_2d(x, y): return x * y
def ref_batch_matvec(A, x): return torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
def ref_batch_add_3d(a, b): return a + b
def ref_sum_axis0(x): return x.sum(dim=0)
def ref_broadcast_add(x, bias): return x + bias


# Tests
def test_elementwise_2d() -> Tuple[bool, str]:
    try:
        x = torch.randn(64, 64, device='cuda')
        y = torch.randn(64, 64, device='cuda')
        result = elementwise_2d(x, y)
        expected = ref_elementwise_2d(x, y)
        if result is None: return False, "None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "64x64 elementwise OK"
    except Exception as e:
        return False, str(e)


def test_elementwise_2d_rect() -> Tuple[bool, str]:
    try:
        x = torch.randn(100, 200, device='cuda')
        y = torch.randn(100, 200, device='cuda')
        result = elementwise_2d(x, y)
        expected = ref_elementwise_2d(x, y)
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "100x200 rect OK"
    except Exception as e:
        return False, str(e)


def test_batch_matvec_basic() -> Tuple[bool, str]:
    try:
        B, M, N = 4, 8, 16
        A = torch.randn(B, M, N, device='cuda')
        x = torch.randn(B, N, device='cuda')
        result = batch_matvec(A, x)
        expected = ref_batch_matvec(A, x)
        if result is None: return False, "None"
        if result.shape != expected.shape:
            return False, f"Shape {result.shape} vs {expected.shape}"
        if not torch.allclose(result, expected, atol=1e-4):
            return False, "Values mismatch"
        return True, "4 batches of 8x16 @ 16 OK"
    except Exception as e:
        return False, str(e)


def test_batch_matvec_large() -> Tuple[bool, str]:
    try:
        B, M, N = 16, 32, 64
        A = torch.randn(B, M, N, device='cuda')
        x = torch.randn(B, N, device='cuda')
        result = batch_matvec(A, x)
        expected = ref_batch_matvec(A, x)
        if not torch.allclose(result, expected, atol=1e-4):
            return False, "Mismatch"
        return True, "16 batches 32x64 OK"
    except Exception as e:
        return False, str(e)


def test_batch_add_3d() -> Tuple[bool, str]:
    try:
        a = torch.randn(8, 32, 32, device='cuda')
        b = torch.randn(8, 32, 32, device='cuda')
        result = batch_add_3d(a, b)
        expected = ref_batch_add_3d(a, b)
        if result is None: return False, "None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "8x32x32 batch add OK"
    except Exception as e:
        return False, str(e)


def test_sum_axis0_basic() -> Tuple[bool, str]:
    try:
        x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device='cuda')
        result = sum_axis0(x)
        expected = ref_sum_axis0(x)  # [5, 7, 9]
        if result is None: return False, "None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, f"Expected {expected.tolist()}, got {result.tolist()}"
        return True, "Sum axis0: [5,7,9] OK"
    except Exception as e:
        return False, str(e)


def test_sum_axis0_large() -> Tuple[bool, str]:
    try:
        x = torch.randn(256, 128, device='cuda')
        result = sum_axis0(x)
        expected = ref_sum_axis0(x)
        if not torch.allclose(result, expected, atol=1e-4):
            return False, "Mismatch"
        return True, "256x128 sum axis0 OK"
    except Exception as e:
        return False, str(e)


def test_broadcast_add_basic() -> Tuple[bool, str]:
    try:
        x = torch.randn(4, 8, device='cuda')
        bias = torch.randn(8, device='cuda')
        result = broadcast_add(x, bias)
        expected = ref_broadcast_add(x, bias)
        if result is None: return False, "None"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "4x8 + bias OK"
    except Exception as e:
        return False, str(e)


def test_broadcast_add_large() -> Tuple[bool, str]:
    try:
        x = torch.randn(128, 256, device='cuda')
        bias = torch.randn(256, device='cuda')
        result = broadcast_add(x, bias)
        expected = ref_broadcast_add(x, bias)
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "128x256 + bias OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("Elementwise 2D", [
            ("basic", test_elementwise_2d),
            ("rect", test_elementwise_2d_rect),
        ]),
        ("Batch MatVec", [
            ("basic", test_batch_matvec_basic),
            ("large", test_batch_matvec_large),
        ]),
        ("Batch Add 3D", [
            ("basic", test_batch_add_3d),
        ]),
        ("Sum Axis 0", [
            ("basic", test_sum_axis0_basic),
            ("large", test_sum_axis0_large),
        ]),
        ("Broadcast Add", [
            ("basic", test_broadcast_add_basic),
            ("large", test_broadcast_add_large),
        ]),
    ]
    
    results = {}
    for name, ts in tests:
        print(f"\n{'='*50}\n{name}\n{'='*50}")
        for tname, tfn in ts:
            p, m = tfn()
            results[f"{name}_{tname}"] = (p, m)
            print(f"  [{'PASS' if p else 'FAIL'}] {tname}: {m}")
    
    passed = sum(1 for p, _ in results.values() if p)
    print(f"\n{'='*50}\nSummary: {passed}/{len(results)}\n{'='*50}")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)
    run_all_tests()
