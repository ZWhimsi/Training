"""Test Suite for Day 34: Flash Attention Backward dQ"""

import torch
import math
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day33 import flash_attention_v2
        from day34 import flash_attention_backward_dq
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_dq(Q, K, V, dO):
    """Reference dQ using autograd."""
    Q = Q.detach().requires_grad_(True)
    K = K.detach()
    V = V.detach()
    
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-2, -1)) * scale
    P = torch.softmax(scores, dim=-1)
    output = P @ V
    output.backward(dO)
    
    return Q.grad


def test_dq() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 1, 4, 64, 32
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        output, L, M = flash_attention_v2(Q, K, V)
        dO = torch.randn_like(output)
        
        dQ = flash_attention_backward_dq(Q, K, V, output, dO, L, M)
        ref_dQ = reference_dq(Q, K, V, dO)
        
        max_err = (dQ - ref_dQ).abs().max().item()
        if max_err > 1e-2:
            return False, f"Error: {max_err:.6f}"
        return True, f"dQ err={max_err:.4f}"
    except Exception as e:
        return False, str(e)


def test_numerical_stability() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 1, 2, 32, 16
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda') * 5
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda') * 5
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        output, L, M = flash_attention_v2(Q, K, V)
        dO = torch.randn_like(output)
        
        dQ = flash_attention_backward_dq(Q, K, V, output, dO, L, M)
        
        if torch.isnan(dQ).any():
            return False, "NaN in dQ"
        
        return True, "stable OK"
    except Exception as e:
        return False, str(e)


def test_various_sizes() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        for batch, heads, seq, dim in [(1, 2, 32, 16), (2, 4, 64, 32)]:
            Q = torch.randn(batch, heads, seq, dim, device='cuda')
            K = torch.randn(batch, heads, seq, dim, device='cuda')
            V = torch.randn(batch, heads, seq, dim, device='cuda')
            
            output, L, M = flash_attention_v2(Q, K, V)
            dO = torch.randn_like(output)
            
            dQ = flash_attention_backward_dq(Q, K, V, output, dO, L, M)
            ref_dQ = reference_dq(Q, K, V, dO)
            
            if (dQ - ref_dQ).abs().max() > 0.1:
                return False, f"Failed at ({batch},{heads},{seq},{dim})"
        
        return True, "various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("dq", test_dq),
        ("numerical_stability", test_numerical_stability),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 34: Flash Backward dQ - Tests\n{'='*50}")
    
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
