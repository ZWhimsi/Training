"""Test Suite for Day 22: Batch Matrix Multiply"""

import torch
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day22 import batched_matmul, batched_matmul_bt, scaled_batched_matmul
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_batched_matmul() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        B, M, K, N = 8, 64, 32, 64
        A = torch.randn(B, M, K, device='cuda')
        B_mat = torch.randn(B, K, N, device='cuda')
        
        result = batched_matmul(A, B_mat)
        expected = torch.bmm(A, B_mat)
        
        if result.shape != expected.shape:
            return False, f"Shape: {result.shape}"
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "bmm OK"
    except Exception as e:
        return False, str(e)


def test_batched_matmul_bt() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        B, M, K, N = 8, 64, 32, 64
        A = torch.randn(B, M, K, device='cuda')
        B_mat = torch.randn(B, N, K, device='cuda')  # (B, N, K) to be transposed
        
        result = batched_matmul_bt(A, B_mat)
        expected = torch.bmm(A, B_mat.transpose(-2, -1))
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "bmm with transpose OK"
    except Exception as e:
        return False, str(e)


def test_scaled_matmul() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        B, M, K, N = 8, 64, 32, 64
        A = torch.randn(B, M, K, device='cuda')
        B_mat = torch.randn(B, K, N, device='cuda')
        scale = 0.125
        
        result = scaled_batched_matmul(A, B_mat, scale)
        expected = scale * torch.bmm(A, B_mat)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "scaled bmm OK"
    except Exception as e:
        return False, str(e)


def test_attention_pattern() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        # Simulate Q @ K^T / sqrt(d_k)
        B, seq_len, d_k = 4, 128, 64
        Q = torch.randn(B, seq_len, d_k, device='cuda')
        K = torch.randn(B, seq_len, d_k, device='cuda')
        scale = 1.0 / (d_k ** 0.5)
        
        result = batched_matmul_bt(Q, K)
        result = result * scale
        
        expected = torch.bmm(Q, K.transpose(-2, -1)) * scale
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "attention pattern OK"
    except Exception as e:
        return False, str(e)


def test_various_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for B, M, K, N in [(4, 32, 32, 32), (8, 64, 64, 64), (16, 100, 50, 100)]:
            A = torch.randn(B, M, K, device='cuda')
            B_mat = torch.randn(B, K, N, device='cuda')
            
            result = batched_matmul(A, B_mat)
            expected = torch.bmm(A, B_mat)
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-2:
                return False, f"Failed at ({B},{M},{K},{N})"
        
        return True, "various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("batched_matmul", test_batched_matmul),
        ("batched_matmul_bt", test_batched_matmul_bt),
        ("scaled_matmul", test_scaled_matmul),
        ("attention_pattern", test_attention_pattern),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 22: Batch Matmul - Tests\n{'='*50}")
    
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
