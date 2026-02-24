"""Test Suite for Day 33: Flash Attention v2"""

import torch
import math
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day33 import flash_attention_v2
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


def test_basic() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        output, L, M = flash_attention_v2(Q, K, V)
        expected = reference_attention(Q, K, V)
        
        max_err = (output - expected).abs().max().item()
        if max_err > 1e-2:
            return False, f"Error: {max_err:.6f}"
        return True, "basic OK"
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
        
        output, _, _ = flash_attention_v2(Q, K, V, causal=True)
        expected = reference_attention(Q, K, V, causal=True)
        
        max_err = (output - expected).abs().max().item()
        if max_err > 1e-2:
            return False, f"Error: {max_err:.6f}"
        return True, "causal OK"
    except Exception as e:
        return False, str(e)


def test_stats_stored() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 2, 4, 64, 32
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        output, L, M = flash_attention_v2(Q, K, V)
        expected = reference_attention(Q, K, V)
        
        if L.shape != (batch, n_heads, seq_len):
            return False, f"L shape: {L.shape}"
        if M.shape != (batch, n_heads, seq_len):
            return False, f"M shape: {M.shape}"
        
        if torch.isnan(L).any() or torch.isnan(M).any():
            return False, "NaN in stats"
        
        if not torch.allclose(output, expected, atol=1e-2, rtol=1e-2):
            max_err = (output - expected).abs().max().item()
            return False, f"Output mismatch: {max_err:.4f}"
        
        return True, "stats and output OK"
    except Exception as e:
        return False, str(e)


def test_large_sequence() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 1, 4, 512, 64
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        output, _, _ = flash_attention_v2(Q, K, V)
        expected = reference_attention(Q, K, V)
        
        max_err = (output - expected).abs().max().item()
        if max_err > 0.05:
            return False, f"Error: {max_err:.6f}"
        return True, "large seq OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("basic", test_basic),
        ("causal", test_causal),
        ("stats_stored", test_stats_stored),
        ("large_sequence", test_large_sequence),
    ]
    
    print(f"\n{'='*50}\nDay 33: Flash Attention v2 - Tests\n{'='*50}")
    
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
