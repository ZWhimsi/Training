"""Test Suite for Day 35: Complete Flash Attention

FINAL TEST - Complete Flash Attention Implementation
"""

import torch
import math
import sys
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day35 import flash_attention, standard_attention, FlashAttentionFunction
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


def test_forward_correctness() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        result = flash_attention(Q, K, V)
        expected = standard_attention(Q, K, V)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-2:
            return False, f"Forward error: {max_err:.6f}"
        return True, f"forward OK (err={max_err:.4f})"
    except Exception as e:
        return False, str(e)


def test_causal_forward() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 2, 8, 128, 64
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        result = flash_attention(Q, K, V, causal=True)
        expected = standard_attention(Q, K, V, causal=True)
        
        max_err = (result - expected).abs().max().item()
        if max_err > 1e-2:
            return False, f"Causal error: {max_err:.6f}"
        return True, f"causal OK (err={max_err:.4f})"
    except Exception as e:
        return False, str(e)


def test_backward_runs() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 1, 4, 64, 32
        Q = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        K = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        V = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        
        output = flash_attention(Q, K, V)
        loss = output.sum()
        loss.backward()
        
        if Q.grad is None or K.grad is None or V.grad is None:
            return False, "Gradients not computed"
        
        if torch.isnan(Q.grad).any() or torch.isnan(K.grad).any() or torch.isnan(V.grad).any():
            return False, "NaN in gradients"
        
        return True, "backward runs OK"
    except Exception as e:
        return False, str(e)


def test_backward_correctness() -> Tuple[bool, str]:
    if not CUDA_AVAILABLE:
        return False, "CUDA required"
    try:
        batch, n_heads, seq_len, head_dim = 1, 4, 32, 16
        
        # Our implementation
        Q1 = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        K1 = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        V1 = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        
        dO = torch.randn(batch, n_heads, seq_len, head_dim, device='cuda')
        
        out1 = flash_attention(Q1, K1, V1)
        out1.backward(dO)
        
        # Reference
        Q2 = Q1.detach().clone().requires_grad_(True)
        K2 = K1.detach().clone().requires_grad_(True)
        V2 = V1.detach().clone().requires_grad_(True)
        
        out2 = standard_attention(Q2, K2, V2)
        out2.backward(dO)
        
        dq_err = (Q1.grad - Q2.grad).abs().max().item()
        dk_err = (K1.grad - K2.grad).abs().max().item()
        dv_err = (V1.grad - V2.grad).abs().max().item()
        
        max_err = max(dq_err, dk_err, dv_err)
        if max_err > 0.1:  # Relaxed tolerance
            return False, f"Grad error: dQ={dq_err:.4f}, dK={dk_err:.4f}, dV={dv_err:.4f}"
        
        return True, f"backward correct (max_err={max_err:.4f})"
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
            expected = standard_attention(Q, K, V)
            
            max_err = (result - expected).abs().max().item()
            if max_err > 0.05:
                return False, f"Failed at ({batch},{heads},{seq},{dim})"
        
        return True, "various sizes OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("forward_correctness", test_forward_correctness),
        ("causal_forward", test_causal_forward),
        ("backward_runs", test_backward_runs),
        ("backward_correctness", test_backward_correctness),
        ("various_sizes", test_various_sizes),
    ]
    
    print(f"\n{'='*60}")
    print("Day 35: FINAL TEST - Complete Flash Attention")
    print("=" * 60)
    
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        return
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    
    print(f"\n{'='*60}")
    print(f"FINAL SCORE: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 CONGRATULATIONS! 🎉")
        print("You have successfully completed the Triton track!")
        print("You built Flash Attention from scratch!")
    else:
        print("\nKeep going! Review the failing tests and try again.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
