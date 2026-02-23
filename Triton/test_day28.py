"""Test Suite for Day 28: Causal Attention"""

import torch
import math
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day28 import causal_attention, mh_causal_attention
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_causal_attention(Q, K, V):
    """Reference causal attention."""
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-2, -1)) * scale
    
    # Create causal mask
    seq_len = Q.shape[-2]
    mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
    if Q.dim() == 3:
        mask = mask.unsqueeze(0)
    scores.masked_fill_(mask, float('-inf'))
    
    return torch.softmax(scores, dim=-1) @ V


def test_causal_attention() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len, head_dim = 64, 32
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        result = causal_attention(Q, K, V)
        expected = reference_causal_attention(Q, K, V)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "causal attention OK"
    except Exception as e:
        return False, str(e)


def test_mh_causal_attention() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        n_heads, seq_len, head_dim = 8, 64, 32
        Q = torch.randn(n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(n_heads, seq_len, head_dim, device='cuda')
        
        result = mh_causal_attention(Q, K, V)
        expected = reference_causal_attention(Q, K, V)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-3:
            return False, f"Error: {max_err:.6f}"
        return True, "multi-head causal OK"
    except Exception as e:
        return False, str(e)


def test_causality() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        seq_len, head_dim = 32, 16
        Q = torch.randn(seq_len, head_dim, device='cuda')
        K = torch.randn(seq_len, head_dim, device='cuda')
        V = torch.randn(seq_len, head_dim, device='cuda')
        
        result = causal_attention(Q, K, V)
        
        # Verify output[i] only depends on V[:i+1]
        # Change V[i+1:] and check output[:i+1] unchanged
        V_modified = V.clone()
        V_modified[seq_len//2:] = torch.randn_like(V_modified[seq_len//2:])
        result_modified = causal_attention(Q, K, V_modified)
        
        # First half should be unchanged
        first_half_err = (result[:seq_len//2] - result_modified[:seq_len//2]).abs().max().item()
        if first_half_err > 1e-5:
            return False, "Causality violated"
        
        return True, "causality verified"
    except Exception as e:
        return False, str(e)


def test_various_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for seq_len, head_dim in [(32, 32), (64, 64), (100, 32)]:
            Q = torch.randn(seq_len, head_dim, device='cuda')
            K = torch.randn(seq_len, head_dim, device='cuda')
            V = torch.randn(seq_len, head_dim, device='cuda')
            
            result = causal_attention(Q, K, V)
            expected = reference_causal_attention(Q, K, V)
            
            max_err = (result - expected).abs().max().item()
            if max_err > 1e-2:
                return False, f"Failed at {seq_len}x{head_dim}"
        
        return True, "various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("causal_attention", test_causal_attention),
        ("mh_causal_attention", test_mh_causal_attention),
        ("causality", test_causality),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 28: Causal Attention - Tests\n{'='*50}")
    
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
