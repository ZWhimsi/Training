"""Test Suite for Day 19: Block Matrix Operations"""

import torch
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day19 import (extract_block, block_matrix_add, 
                           extract_block_diagonal, block_traces)
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_extract_block() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(128, 128, device='cuda')
        block = extract_block(x, 1, 2, 32, 32)
        expected = x[32:64, 64:96]
        
        max_err = (block - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "block extraction OK"
    except Exception as e:
        return False, str(e)


def test_block_add() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        a = torch.randn(100, 100, device='cuda')
        b = torch.randn(100, 100, device='cuda')
        
        result = block_matrix_add(a, b)
        expected = a + b
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "block add OK"
    except Exception as e:
        return False, str(e)


def test_block_diagonal() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        x = torch.randn(128, 128, device='cuda')
        block_size = 32
        
        result = extract_block_diagonal(x, block_size)
        
        # Check shape
        expected_shape = (4, 32, 32)
        if result.shape != expected_shape:
            return False, f"Shape: {result.shape} != {expected_shape}"
        
        # Check first diagonal block
        expected_block = x[:32, :32]
        max_err = (result[0] - expected_block).abs().max().item()
        if max_err > 1e-5:
            return False, f"Block 0 error: {max_err:.6f}"
        
        return True, "diagonal blocks OK"
    except Exception as e:
        return False, str(e)


def test_block_traces() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        # Create identity-like matrix
        x = torch.eye(128, device='cuda')
        block_size = 32
        
        result = block_traces(x, block_size)
        expected = torch.tensor([32.0] * 4, device='cuda')
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "traces OK"
    except Exception as e:
        return False, str(e)


def test_non_square() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        a = torch.randn(96, 128, device='cuda')
        b = torch.randn(96, 128, device='cuda')
        
        result = block_matrix_add(a, b, block_m=32, block_n=32)
        expected = a + b
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-5:
            return False, f"Error: {max_err:.6f}"
        return True, "non-square OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("extract_block", test_extract_block),
        ("block_add", test_block_add),
        ("block_diagonal", test_block_diagonal),
        ("block_traces", test_block_traces),
        ("non_square", test_non_square),
    ]
    
    print(f"\n{'='*50}\nDay 19: Block Operations - Tests\n{'='*50}")
    
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
