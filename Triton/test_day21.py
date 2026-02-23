"""Test Suite for Day 21: Matrix-Vector Products"""

import torch
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day21 import matvec, matvec_blocked, vecmat, batched_matvec
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_matvec() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        M, N = 256, 128
        A = torch.randn(M, N, device='cuda')
        x = torch.randn(N, device='cuda')
        
        result = matvec(A, x)
        expected = A @ x
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "matvec OK"
    except Exception as e:
        return False, str(e)


def test_matvec_blocked() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        M, N = 256, 2048  # Large N
        A = torch.randn(M, N, device='cuda')
        x = torch.randn(N, device='cuda')
        
        result = matvec_blocked(A, x)
        expected = A @ x
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "blocked matvec OK"
    except Exception as e:
        return False, str(e)


def test_vecmat() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        M, N = 128, 256
        v = torch.randn(M, device='cuda')
        A = torch.randn(M, N, device='cuda')
        
        result = vecmat(v, A)
        expected = v @ A
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "vecmat OK"
    except Exception as e:
        return False, str(e)


def test_batched_matvec() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        B, M, N = 8, 256, 128
        A = torch.randn(B, M, N, device='cuda')
        x = torch.randn(B, N, device='cuda')
        
        result = batched_matvec(A, x)
        expected = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
        
        if result.shape != expected.shape:
            return False, f"Shape: {result.shape}"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-4:
            return False, f"Error: {max_err:.6f}"
        return True, "batched matvec OK"
    except Exception as e:
        return False, str(e)


def test_various_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for M, N in [(64, 64), (128, 256), (100, 200)]:
            A = torch.randn(M, N, device='cuda')
            x = torch.randn(N, device='cuda')
            
            result = matvec(A, x)
            expected = A @ x
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-3:
                return False, f"Failed at {M}x{N}"
        
        return True, "various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("matvec", test_matvec),
        ("matvec_blocked", test_matvec_blocked),
        ("vecmat", test_vecmat),
        ("batched_matvec", test_batched_matvec),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 21: Matrix-Vector - Tests\n{'='*50}")
    
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        return
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    run_all_tests()
