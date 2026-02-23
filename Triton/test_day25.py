"""Test Suite for Day 25: Attention Output Computation"""

import torch
import math
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day25 import attention_output, mha_attention_output, full_attention, full_mha
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_attention_output() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len, head_dim = 64, 32
        weights = torch.softmax(torch.randn(seq_len, seq_len, device='cuda'), dim=-1)
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        result = attention_output(weights, V)
        expected = weights @ V
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "output OK"
    except Exception as e:
        return False, str(e)


def test_mha_output() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        n_heads, seq_len, head_dim = 8, 64, 32
        weights = torch.softmax(torch.randn(n_heads, seq_len, seq_len, device='cuda'), dim=-1)
        V = torch.randn(n_heads, seq_len, head_dim, device='cuda')
        
        result = mha_attention_output(weights, V)
        expected = torch.bmm(weights, V)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "MHA output OK"
    except Exception as e:
        return False, str(e)


def test_full_attention() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len, head_dim = 64, 32
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        result = full_attention(Q, K, V)
        
        if result is None:
            return False, "Returned None"
        
        # Manual reference
        scale = 1.0 / math.sqrt(head_dim)
        scores = (Q @ K.T) * scale
        weights = torch.softmax(scores, dim=-1)
        expected = weights @ V
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "full attention OK"
    except Exception as e:
        return False, str(e)


def test_full_mha() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        n_heads, seq_len, head_dim = 8, 64, 32
        Q = torch.randn(n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(n_heads, seq_len, head_dim, device='cuda')
        
        result = full_mha(Q, K, V)
        
        if result is None:
            return False, "Returned None"
        
        # Manual reference
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.bmm(Q, K.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        expected = torch.bmm(weights, V)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "full MHA OK"
    except Exception as e:
        return False, str(e)


def test_various_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for seq_len, head_dim in [(32, 32), (64, 64), (128, 64)]:
            weights = torch.softmax(torch.randn(seq_len, seq_len, device='cuda'), dim=-1)
            V = torch.randn(seq_len, head_dim, device='cuda')
            
            result = attention_output(weights, V)
            expected = weights @ V
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-2:
                return False, f"Failed at {seq_len}x{head_dim}"
        
        return True, "various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("attention_output", test_attention_output),
        ("mha_output", test_mha_output),
        ("full_attention", test_full_attention),
        ("full_mha", test_full_mha),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 25: Attention Output - Tests\n{'='*50}")
    
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
