"""Test Suite for Day 11: Batch Normalization"""

import torch
import torch.nn as nn
from typing import Tuple

try:
    from day11 import (batch_norm_forward, ManualBatchNorm1d, ManualBatchNorm2d,
                       demonstrate_train_eval_difference, ConvBNReLU, MLPWithBatchNorm)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_batch_norm_forward() -> Tuple[bool, str]:
    """Test manual batch norm forward pass against PyTorch."""
    try:
        torch.manual_seed(42)
        x = torch.randn(32, 64)
        gamma = torch.ones(64)
        beta = torch.zeros(64)
        
        out, mean, var = batch_norm_forward(x, gamma, beta)
        
        # Check if actually implemented
        if torch.allclose(out, x):
            return False, "Not implemented"
        
        # Compare with PyTorch
        bn = nn.BatchNorm1d(64, affine=False)
        bn.train()
        expected = nn.functional.batch_norm(
            x, None, None, gamma, beta, training=True, eps=1e-5
        )
        
        if not torch.allclose(out, expected, atol=1e-5):
            return False, f"Output mismatch: max diff {(out - expected).abs().max():.6f}"
        
        # Check mean/var are reasonable
        expected_mean = x.mean(dim=0)
        expected_var = x.var(dim=0, unbiased=False)
        
        if not torch.allclose(mean, expected_mean, atol=1e-5):
            return False, "Mean calculation incorrect"
        
        if not torch.allclose(var, expected_var, atol=1e-5):
            return False, "Variance calculation incorrect"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_manual_batch_norm_1d_train() -> Tuple[bool, str]:
    """Test ManualBatchNorm1d in training mode."""
    try:
        torch.manual_seed(42)
        
        manual_bn = ManualBatchNorm1d(32)
        pytorch_bn = nn.BatchNorm1d(32)
        
        # Copy parameters
        pytorch_bn.weight.data = manual_bn.gamma.data.clone()
        pytorch_bn.bias.data = manual_bn.beta.data.clone()
        pytorch_bn.running_mean.data = manual_bn.running_mean.data.clone()
        pytorch_bn.running_var.data = manual_bn.running_var.data.clone()
        
        manual_bn.train()
        pytorch_bn.train()
        
        x = torch.randn(16, 32)
        
        out_manual = manual_bn(x)
        out_pytorch = pytorch_bn(x)
        
        if torch.allclose(out_manual, x):
            return False, "Not implemented"
        
        if not torch.allclose(out_manual, out_pytorch, atol=1e-5):
            return False, f"Output mismatch: max diff {(out_manual - out_pytorch).abs().max():.6f}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_manual_batch_norm_1d_running_stats() -> Tuple[bool, str]:
    """Test that running statistics are updated correctly."""
    try:
        torch.manual_seed(42)
        
        manual_bn = ManualBatchNorm1d(32, momentum=0.1)
        manual_bn.train()
        
        initial_mean = manual_bn.running_mean.clone()
        initial_var = manual_bn.running_var.clone()
        
        # Run a few batches
        for _ in range(5):
            x = torch.randn(16, 32)
            _ = manual_bn(x)
        
        # Check running stats changed
        if torch.allclose(manual_bn.running_mean, initial_mean):
            return False, "Running mean not updated"
        
        if torch.allclose(manual_bn.running_var, initial_var):
            return False, "Running var not updated"
        
        return True, "OK (stats updated)"
    except Exception as e:
        return False, str(e)


def test_manual_batch_norm_2d() -> Tuple[bool, str]:
    """Test ManualBatchNorm2d against PyTorch."""
    try:
        torch.manual_seed(42)
        
        manual_bn = ManualBatchNorm2d(16)
        pytorch_bn = nn.BatchNorm2d(16)
        
        # Copy parameters
        pytorch_bn.weight.data = manual_bn.gamma.data.clone()
        pytorch_bn.bias.data = manual_bn.beta.data.clone()
        
        manual_bn.train()
        pytorch_bn.train()
        
        x = torch.randn(8, 16, 14, 14)  # (N, C, H, W)
        
        out_manual = manual_bn(x)
        out_pytorch = pytorch_bn(x)
        
        if torch.allclose(out_manual, x):
            return False, "Not implemented"
        
        if not torch.allclose(out_manual, out_pytorch, atol=1e-5):
            return False, f"Output mismatch: max diff {(out_manual - out_pytorch).abs().max():.6f}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_train_eval_difference() -> Tuple[bool, str]:
    """Test train vs eval mode produces different outputs."""
    try:
        torch.manual_seed(42)
        
        bn = nn.BatchNorm1d(32)
        bn.train()
        
        # Train on some data to get non-trivial running stats
        for _ in range(10):
            _ = bn(torch.randn(16, 32))
        
        x_train = torch.randn(8, 32) * 2 + 1  # Different distribution
        x_eval = x_train.clone()
        
        out_train, out_eval = demonstrate_train_eval_difference(bn, x_train, x_eval)
        
        if torch.allclose(out_train, x_train):
            return False, "Not implemented"
        
        # Outputs should be different (train uses batch stats, eval uses running stats)
        if torch.allclose(out_train, out_eval):
            return False, "Train and eval outputs are the same"
        
        return True, "OK (outputs differ)"
    except Exception as e:
        return False, str(e)


def test_conv_bn_relu() -> Tuple[bool, str]:
    """Test ConvBNReLU block."""
    try:
        block = ConvBNReLU(3, 64, kernel_size=3, padding=1)
        
        if block.conv is None:
            return False, "Not implemented"
        
        x = torch.randn(4, 3, 32, 32)
        out = block(x)
        
        # Check output shape
        if out.shape != torch.Size([4, 64, 32, 32]):
            return False, f"Expected shape [4, 64, 32, 32], got {list(out.shape)}"
        
        # Check ReLU applied (no negative values)
        if out.min() < 0:
            return False, "ReLU not applied (negative values found)"
        
        # Check conv has no bias (since BN handles it)
        if block.conv.bias is not None:
            return False, "Conv should have bias=False"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_mlp_with_batch_norm() -> Tuple[bool, str]:
    """Test MLP with batch normalization."""
    try:
        model = MLPWithBatchNorm(input_dim=64, hidden_dim=128, output_dim=10)
        
        if model.fc1 is None:
            return False, "Not implemented"
        
        x = torch.randn(32, 64)
        out = model(x)
        
        # Check output shape
        if out.shape != torch.Size([32, 10]):
            return False, f"Expected shape [32, 10], got {list(out.shape)}"
        
        # Verify batch norm layers exist
        if model.bn1 is None or model.bn2 is None:
            return False, "Missing batch norm layers"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_batch_norm_gradient_flow() -> Tuple[bool, str]:
    """Test that gradients flow through batch norm."""
    try:
        bn = ManualBatchNorm1d(32)
        bn.train()
        
        x = torch.randn(16, 32, requires_grad=True)
        out = bn(x)
        
        if torch.allclose(out, x):
            return False, "Not implemented"
        
        loss = out.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "No gradient for input"
        
        if bn.gamma.grad is None:
            return False, "No gradient for gamma"
        
        if bn.beta.grad is None:
            return False, "No gradient for beta"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("batch_norm_forward", test_batch_norm_forward),
        ("manual_bn_1d_train", test_manual_batch_norm_1d_train),
        ("manual_bn_1d_running_stats", test_manual_batch_norm_1d_running_stats),
        ("manual_bn_2d", test_manual_batch_norm_2d),
        ("train_eval_difference", test_train_eval_difference),
        ("conv_bn_relu", test_conv_bn_relu),
        ("mlp_with_batch_norm", test_mlp_with_batch_norm),
        ("gradient_flow", test_batch_norm_gradient_flow),
    ]
    
    print(f"\n{'='*50}\nDay 11: Batch Normalization - Tests\n{'='*50}")
    
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
