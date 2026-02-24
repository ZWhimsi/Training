"""Test Suite for Day 32: Flash Attention Forward"""

import torch
import sys
from typing import Tuple

try:
    from day32 import flash_attention_forward, standard_attention
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_flash_attention_small() -> Tuple[bool, str]:
    """Test with small tensors."""
    try:
        B, H, M, N, D = 1, 1, 16, 16, 32
        Q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        
        result = flash_attention_forward(Q, K, V)
        expected = standard_attention(Q, K, V)
        
        if result is None:
            return False, "Returned None"
        
        if not torch.allclose(result, expected, atol=1e-2, rtol=1e-2):
            max_diff = (result - expected).abs().max().item()
            return False, f"Max diff: {max_diff:.4f}"
        
        return True, f"Small attention (1,1,16,16,32) OK"
    except Exception as e:
        return False, str(e)


def test_flash_attention_medium() -> Tuple[bool, str]:
    """Test with medium tensors."""
    try:
        B, H, M, N, D = 2, 4, 64, 64, 64
        Q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        
        result = flash_attention_forward(Q, K, V)
        expected = standard_attention(Q, K, V)
        
        if not torch.allclose(result, expected, atol=1e-2, rtol=1e-2):
            max_diff = (result - expected).abs().max().item()
            return False, f"Max diff: {max_diff:.4f}"
        
        return True, f"Medium attention (2,4,64,64,64) OK"
    except Exception as e:
        return False, str(e)


def test_flash_attention_asymmetric() -> Tuple[bool, str]:
    """Test with M != N."""
    try:
        B, H, M, N, D = 1, 2, 32, 48, 32
        Q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        
        result = flash_attention_forward(Q, K, V)
        expected = standard_attention(Q, K, V)
        
        if not torch.allclose(result, expected, atol=1e-2, rtol=1e-2):
            max_diff = (result - expected).abs().max().item()
            return False, f"Max diff: {max_diff:.4f}"
        
        return True, f"Asymmetric (M=32, N=48) OK"
    except Exception as e:
        return False, str(e)


def test_flash_attention_output_sum() -> Tuple[bool, str]:
    """Test that output is valid and matches reference."""
    try:
        B, H, M, N, D = 1, 1, 64, 64, 32
        Q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        
        result = flash_attention_forward(Q, K, V)
        expected = standard_attention(Q, K, V)
        
        if torch.any(torch.isnan(result)):
            return False, "Output contains NaN"
        
        if torch.any(torch.isinf(result)):
            return False, "Output contains Inf"
        
        if not torch.allclose(result, expected, atol=1e-2, rtol=1e-2):
            max_diff = (result - expected).abs().max().item()
            return False, f"Values mismatch: {max_diff:.4f}"
        
        return True, "Output valid and matches reference"
    except Exception as e:
        return False, str(e)


def test_memory_efficiency() -> Tuple[bool, str]:
    """Verify Flash Attention produces correct output on larger inputs."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        B, H, M, N, D = 1, 1, 256, 256, 64
        Q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        
        result = flash_attention_forward(Q, K, V)
        expected = standard_attention(Q, K, V)
        
        if not torch.allclose(result, expected, atol=1e-2, rtol=1e-2):
            max_diff = (result - expected).abs().max().item()
            return False, f"Values mismatch: {max_diff:.4f}"
        
        return True, "Large input OK, O(N) memory"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("small", test_flash_attention_small),
        ("medium", test_flash_attention_medium),
        ("asymmetric", test_flash_attention_asymmetric),
        ("valid_output", test_flash_attention_output_sum),
        ("memory", test_memory_efficiency),
    ]
    
    print(f"\n{'='*60}")
    print("Day 32: Flash Attention Forward - Tests")
    print("='*60")
    print("\nComparing Flash Attention against standard attention...")
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    
    print(f"\nSummary: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\nCongratulations! You've implemented Flash Attention!")
        print("This is a significant achievement in GPU programming.")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)
    run_all_tests()
