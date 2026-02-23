"""Test Suite for Day 31: Flash Attention Backward dV, dK"""

import torch
import math
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day30 import flash_attention_forward
        from day31 import flash_attention_backward_dv_dk
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def reference_backward(Q, K, V, dO):
    """Reference backward using autograd."""
    Q = Q.detach().requires_grad_(True)
    K = K.detach().requires_grad_(True)
    V = V.detach().requires_grad_(True)
    
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.transpose(-2, -1)) * scale
    P = torch.softmax(scores, dim=-1)
    output = P @ V
    output.backward(dO)
    
    return Q.grad, K.grad, V.grad


def test_dv_dk() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 1, 4, 64, 32
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        output, L = flash_attention_forward(Q, K, V)
        dO = torch.randn_like(output)
        
        dK, dV = flash_attention_backward_dv_dk(Q, K, V, output, dO, L)
        
        _, ref_dK, ref_dV = reference_backward(Q, K, V, dO)
        
        dk_err = (dK - ref_dK).abs().max().item()
        dv_err = (dV - ref_dV).abs().max().item()
        
        if dk_err > 1e-2:
            return False, f"dK error: {dk_err:.6f}"
        if dv_err > 1e-2:
            return False, f"dV error: {dv_err:.6f}"
        
        return True, f"dK err={dk_err:.4f}, dV err={dv_err:.4f}"
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
        
        output, L = flash_attention_forward(Q, K, V)
        dO = torch.randn_like(output)
        
        dK, dV = flash_attention_backward_dv_dk(Q, K, V, output, dO, L)
        
        if torch.isnan(dK).any() or torch.isnan(dV).any():
            return False, "NaN in gradients"
        
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
            
            output, L = flash_attention_forward(Q, K, V)
            dO = torch.randn_like(output)
            
            dK, dV = flash_attention_backward_dv_dk(Q, K, V, output, dO, L)
            _, ref_dK, ref_dV = reference_backward(Q, K, V, dO)
            
            if (dK - ref_dK).abs().max() > 0.1 or (dV - ref_dV).abs().max() > 0.1:
                return False, f"Failed at ({batch},{heads},{seq},{dim})"
        
        return True, "various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("dv_dk", test_dv_dk),
        ("numerical_stability", test_numerical_stability),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*50}\nDay 31: Flash Backward dV,dK - Tests\n{'='*50}")
    
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
