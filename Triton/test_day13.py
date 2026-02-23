"""Test Suite for Day 13: Fused Softmax"""

import torch
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day13 import softmax_1d, softmax_2d, log_softmax_1d, softmax_temperature
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_softmax_1d() -> Tuple[bool, str]:
    try:
        x = torch.randn(128, device='cuda')
        result = softmax_1d(x)
        expected = F.softmax(x, dim=0)
        
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        if not torch.allclose(result.sum(), torch.tensor(1.0, device='cuda'), atol=1e-5):
            return False, "Doesn't sum to 1"
        return True, "1D softmax OK, sums to 1"
    except Exception as e:
        return False, str(e)


def test_softmax_1d_stability() -> Tuple[bool, str]:
    try:
        # Large values that would overflow naive exp
        x = torch.tensor([1000, 1001, 1002], dtype=torch.float32, device='cuda')
        result = softmax_1d(x)
        expected = F.softmax(x, dim=0)
        
        if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
            return False, "Contains NaN/Inf (not stable)"
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "Large values handled (stable)"
    except Exception as e:
        return False, str(e)


def test_softmax_2d() -> Tuple[bool, str]:
    try:
        x = torch.randn(32, 64, device='cuda')
        result = softmax_2d(x)
        expected = F.softmax(x, dim=1)
        
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        # Check each row sums to 1
        row_sums = result.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones(32, device='cuda'), atol=1e-5):
            return False, "Rows don't sum to 1"
        return True, "2D row softmax OK"
    except Exception as e:
        return False, str(e)


def test_log_softmax() -> Tuple[bool, str]:
    try:
        x = torch.randn(64, device='cuda')
        result = log_softmax_1d(x)
        expected = F.log_softmax(x, dim=0)
        
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        return True, "Log-softmax OK"
    except Exception as e:
        return False, str(e)


def test_softmax_temperature_high() -> Tuple[bool, str]:
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        result = softmax_temperature(x, temperature=10.0)
        expected = F.softmax(x / 10.0, dim=0)
        
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        # High temp should be more uniform
        if result.max() - result.min() > 0.2:
            return False, "Not uniform enough"
        return True, "High temp = uniform"
    except Exception as e:
        return False, str(e)


def test_softmax_temperature_low() -> Tuple[bool, str]:
    try:
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        result = softmax_temperature(x, temperature=0.1)
        expected = F.softmax(x / 0.1, dim=0)
        
        if not torch.allclose(result, expected, atol=1e-5):
            return False, "Mismatch"
        # Low temp should be peaked
        if result.max() < 0.95:
            return False, "Not peaked enough"
        return True, "Low temp = peaked"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("softmax_1d", test_softmax_1d),
        ("softmax_stability", test_softmax_1d_stability),
        ("softmax_2d", test_softmax_2d),
        ("log_softmax", test_log_softmax),
        ("temp_high", test_softmax_temperature_high),
        ("temp_low", test_softmax_temperature_low),
    ]
    
    print(f"\n{'='*50}\nDay 13: Fused Softmax - Tests\n{'='*50}")
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
