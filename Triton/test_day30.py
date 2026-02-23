"""Test Suite for Day 30: Flash Attention Forward"""

import torch
import math
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day30 import flash_attention, flash_attention_forward
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_attention(Q, K, V, causal=False):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-2, -1)) * scale
    
    if causal:
        seq_len = Q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    return torch.softmax(scores, dim=-1) @ V


def test_flash_attention() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        result = flash_attention(Q, K, V, causal=False)
        expected = reference_attention(Q, K, V, causal=False)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-2:
            return False, f"Error: {max_err:.6f}"
        return True, "non-causal OK"
    except Exception as e:
        return False, str(e)


def test_causal() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        result = flash_attention(Q, K, V, causal=True)
        expected = reference_attention(Q, K, V, causal=True)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-2:
            return False, f"Error: {max_err:.6f}"
        return True, "causal OK"
    except Exception as e:
        return False, str(e)


def test_lse_stored() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 2, 4, 64, 32
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        output, L = flash_attention_forward(Q, K, V)
        
        if L.shape != (batch, n_heads, seq_len):
            return False, f"L shape: {L.shape}"
        
        if torch.isnan(L).any():
            return False, "NaN in L"
        
        return True, "LSE stored OK"
    except Exception as e:
        return False, str(e)


def test_numerical_stability() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 1, 4, 64, 32
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda') * 10
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda') * 10
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        result = flash_attention(Q, K, V)
        
        if torch.isnan(result).any():
            return False, "NaN in output"
        if torch.isinf(result).any():
            return False, "Inf in output"
        
        return True, "stable OK"
    except Exception as e:
        return False, str(e)


def test_various_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for batch, heads, seq, dim in [(1, 4, 64, 32), (2, 8, 128, 64), (1, 1, 256, 64)]:
            Q = torch.randn(batch, heads, seq, dim, device='cuda')
            K = torch.randn(batch, heads, seq, dim, device='cuda')
            V = torch.randn(batch, heads, seq, dim, device='cuda')
            
            result = flash_attention(Q, K, V)
            expected = reference_attention(Q, K, V)
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-2:
                return False, f"Failed at ({batch},{heads},{seq},{dim})"
        
        return True, "various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("flash_attention", test_flash_attention),
        ("causal", test_causal),
        ("lse_stored", test_lse_stored),
        ("numerical_stability", test_numerical_stability),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 30: Flash Attention Forward - Tests\n{'='*50}")
    
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
