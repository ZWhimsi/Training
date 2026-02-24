"""Test Suite for Day 11: Batch Normalization"""

import torch
import pytest
import torch.nn as nn
try:
    from day11 import (batch_norm_forward, ManualBatchNorm1d, ManualBatchNorm2d,
                       demonstrate_train_eval_difference, ConvBNReLU, MLPWithBatchNorm)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_batch_norm_forward():
    """Test manual batch norm forward pass against PyTorch."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    x = torch.randn(32, 64)
    gamma = torch.ones(64)
    beta = torch.zeros(64)
    
    out, mean, var = batch_norm_forward(x, gamma, beta)
    
    assert not torch.allclose(out, x), "Not implemented"
    
    bn = nn.BatchNorm1d(64, affine=False)
    bn.train()
    expected = nn.functional.batch_norm(
        x, None, None, gamma, beta, training=True, eps=1e-5
    )
    
    assert torch.allclose(out, expected, atol=1e-5), f"Output mismatch: max diff {(out - expected).abs().max():.6f}"
    
    expected_mean = x.mean(dim=0)
    expected_var = x.var(dim=0, unbiased=False)
    
    assert torch.allclose(mean, expected_mean, atol=1e-5), "Mean calculation incorrect"
    assert torch.allclose(var, expected_var, atol=1e-5), "Variance calculation incorrect"

def test_manual_batch_norm_1d_train():
    """Test ManualBatchNorm1d in training mode."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    manual_bn = ManualBatchNorm1d(32)
    pytorch_bn = nn.BatchNorm1d(32)
    
    pytorch_bn.weight.data = manual_bn.gamma.data.clone()
    pytorch_bn.bias.data = manual_bn.beta.data.clone()
    pytorch_bn.running_mean.data = manual_bn.running_mean.data.clone()
    pytorch_bn.running_var.data = manual_bn.running_var.data.clone()
    
    manual_bn.train()
    pytorch_bn.train()
    
    x = torch.randn(16, 32)
    
    out_manual = manual_bn(x)
    out_pytorch = pytorch_bn(x)
    
    assert not torch.allclose(out_manual, x), "Not implemented"
    assert torch.allclose(out_manual, out_pytorch, atol=1e-5), f"Output mismatch: max diff {(out_manual - out_pytorch).abs().max():.6f}"

def test_manual_batch_norm_1d_running_stats():
    """Test that running statistics are updated correctly."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    manual_bn = ManualBatchNorm1d(32, momentum=0.1)
    manual_bn.train()
    
    initial_mean = manual_bn.running_mean.clone()
    initial_var = manual_bn.running_var.clone()
    
    for _ in range(5):
        x = torch.randn(16, 32)
        _ = manual_bn(x)
    
    assert not torch.allclose(manual_bn.running_mean, initial_mean), "Running mean not updated"
    assert not torch.allclose(manual_bn.running_var, initial_var), "Running var not updated"

def test_manual_batch_norm_2d():
    """Test ManualBatchNorm2d against PyTorch."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    manual_bn = ManualBatchNorm2d(16)
    pytorch_bn = nn.BatchNorm2d(16)
    
    pytorch_bn.weight.data = manual_bn.gamma.data.clone()
    pytorch_bn.bias.data = manual_bn.beta.data.clone()
    
    manual_bn.train()
    pytorch_bn.train()
    
    x = torch.randn(8, 16, 14, 14)
    
    out_manual = manual_bn(x)
    out_pytorch = pytorch_bn(x)
    
    assert not torch.allclose(out_manual, x), "Not implemented"
    assert torch.allclose(out_manual, out_pytorch, atol=1e-5), f"Output mismatch: max diff {(out_manual - out_pytorch).abs().max():.6f}"

def test_train_eval_difference():
    """Test train vs eval mode produces different outputs."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    
    bn = nn.BatchNorm1d(32)
    bn.train()
    
    for _ in range(10):
        _ = bn(torch.randn(16, 32))
    
    x_train = torch.randn(8, 32) * 2 + 1
    x_eval = x_train.clone()
    
    out_train, out_eval = demonstrate_train_eval_difference(bn, x_train, x_eval)
    
    assert not torch.allclose(out_train, x_train), "Not implemented"
    assert not torch.allclose(out_train, out_eval), "Train and eval outputs are the same"

def test_conv_bn_relu():
    """Test ConvBNReLU block."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    block = ConvBNReLU(3, 64, kernel_size=3, padding=1)
    
    assert block.conv is not None, "Not implemented"
    
    x = torch.randn(4, 3, 32, 32)
    block.train()
    out = block(x)
    
    assert out.shape == torch.Size([4, 64, 32, 32]), f"Expected shape [4, 64, 32, 32], got {list(out.shape)}"
    assert out.min() >= 0, "ReLU not applied (negative values found)"
    assert block.conv.bias is None, "Conv should have bias=False"
    
    with torch.no_grad():
        conv_out = block.conv(x)
        bn_out = block.bn(conv_out)
        expected = torch.relu(bn_out)
    
    assert torch.allclose(out, expected, atol=1e-5), f"Output doesn't match Conv->BN->ReLU: max diff {(out - expected).abs().max():.6f}"

def test_mlp_with_batch_norm():
    """Test MLP with batch normalization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = MLPWithBatchNorm(input_dim=64, hidden_dim=128, output_dim=10)
    
    assert model.fc1 is not None, "Not implemented"
    
    x = torch.randn(32, 64)
    model.train()
    out = model(x)
    
    assert out.shape == torch.Size([32, 10]), f"Expected shape [32, 10], got {list(out.shape)}"
    assert model.bn1 is not None and model.bn2 is not None, "Missing batch norm layers"
    
    with torch.no_grad():
        h1 = model.fc1(x)
        h1 = model.bn1(h1)
        h1 = model.relu(h1)
        h2 = model.fc2(h1)
        h2 = model.bn2(h2)
        h2 = model.relu(h2)
        expected = model.fc3(h2)
    
    assert torch.allclose(out, expected, atol=1e-5), f"Output doesn't match expected MLP computation: max diff {(out - expected).abs().max():.6f}"

def test_batch_norm_gradient_flow():
    """Test that gradients flow through batch norm."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    bn = ManualBatchNorm1d(32)
    pytorch_bn = nn.BatchNorm1d(32)
    
    pytorch_bn.weight.data = bn.gamma.data.clone()
    pytorch_bn.bias.data = bn.beta.data.clone()
    pytorch_bn.running_mean.data = bn.running_mean.data.clone()
    pytorch_bn.running_var.data = bn.running_var.data.clone()
    
    bn.train()
    pytorch_bn.train()
    
    x = torch.randn(16, 32, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)
    
    out = bn(x)
    out_ref = pytorch_bn(x_ref)
    
    assert not torch.allclose(out, x), "Not implemented"
    
    loss = out.sum()
    loss.backward()
    
    loss_ref = out_ref.sum()
    loss_ref.backward()
    
    assert x.grad is not None, "No gradient for input"
    assert bn.gamma.grad is not None, "No gradient for gamma"
    assert bn.beta.grad is not None, "No gradient for beta"
    
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5), f"Input gradient mismatch: max diff {(x.grad - x_ref.grad).abs().max():.6f}"
    assert torch.allclose(bn.gamma.grad, pytorch_bn.weight.grad, atol=1e-5), f"Gamma gradient mismatch: max diff {(bn.gamma.grad - pytorch_bn.weight.grad).abs().max():.6f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
