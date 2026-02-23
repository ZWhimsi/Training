"""Test Suite for Day 8: Vector Operations"""

import torch
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day08 import gelu, silu, fused_linear_gelu, ema_update, polynomial
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_gelu() -> Tuple[bool, str]:
    try:
        x = torch.randn(10000, device='cuda')
        result = gelu(x)
        expected = F.gelu(x, approximate='tanh')
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch with torch GELU"
        return True, "GELU matches PyTorch"
    except Exception as e:
        return False, str(e)


def test_silu() -> Tuple[bool, str]:
    try:
        x = torch.randn(10000, device='cuda')
        result = silu(x)
        expected = F.silu(x)
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "SiLU matches PyTorch"
    except Exception as e:
        return False, str(e)


def test_fused_linear_gelu() -> Tuple[bool, str]:
    try:
        x = torch.randn(1000, device='cuda')
        w = torch.randn(1000, device='cuda')
        b = torch.randn(1000, device='cuda')
        result = fused_linear_gelu(x, w, b)
        expected = F.gelu(x * w + b, approximate='tanh')
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "Fused linear+GELU OK"
    except Exception as e:
        return False, str(e)


def test_ema() -> Tuple[bool, str]:
    try:
        running = torch.randn(1000, device='cuda')
        new = torch.randn(1000, device='cuda')
        alpha = 0.1
        result = ema_update(running, new, alpha)
        expected = (1 - alpha) * running + alpha * new
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "EMA update OK"
    except Exception as e:
        return False, str(e)


def test_polynomial() -> Tuple[bool, str]:
    try:
        x = torch.randn(1000, device='cuda')
        coeffs = [1.0, 2.0, 3.0, 4.0]  # 1 + 2x + 3x^2 + 4x^3
        result = polynomial(x, coeffs)
        expected = coeffs[0] + coeffs[1]*x + coeffs[2]*x**2 + coeffs[3]*x**3
        if not torch.allclose(result, expected, atol=1e-4):
            return False, "Mismatch"
        return True, "Polynomial eval OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("gelu", test_gelu),
        ("silu", test_silu),
        ("fused_linear_gelu", test_fused_linear_gelu),
        ("ema_update", test_ema),
        ("polynomial", test_polynomial),
    ]
    
    print(f"\n{'='*50}\nDay 8 Tests\n{'='*50}")
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
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)
    run_all_tests()
