"""Test Suite for Day 33: Batch Normalization"""

import numpy as np
import pytest

try:
    from day33 import (
        Tensor,
        batchnorm1d_forward,
        batchnorm1d_backward,
        batchnorm2d_forward,
        batchnorm2d_backward,
        BatchNorm1d,
        BatchNorm2d,
        LayerNorm,
        Module
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_bn1d_forward_shape():
    """Test BatchNorm1d forward output shape."""
    np.random.seed(42)
    x = np.random.randn(32, 64)
    gamma = np.ones(64)
    beta = np.zeros(64)
    rm = np.zeros(64)
    rv = np.ones(64)
        
    result = batchnorm1d_forward(x, gamma, beta, rm, rv, training=True)
        
    assert not (result is None), "returned None"
        
    out, cache = result
        
    assert not (out.shape != x.shape), f"shape {out.shape}, expected {x.shape}"
        
    # Verify output is finite
    assert np.all(np.isfinite(out)), "output contains NaN or Inf"
        
    # Verify normalization: mean should be ~0, var should be ~1
    out_mean = np.mean(out, axis=0)
    out_var = np.var(out, axis=0)
    assert np.allclose(out_mean, 0, atol=1e-5), f"mean not ~0: max abs mean = {np.max(np.abs(out_mean))}"
    assert np.allclose(out_var, 1, atol=1e-4), f"var not ~1: max diff = {np.max(np.abs(out_var - 1))}"
        
def test_bn1d_forward_normalized():
    """Test BatchNorm1d produces normalized output."""
    np.random.seed(42)
    x = np.random.randn(100, 32) * 5 + 3
    gamma = np.ones(32)
    beta = np.zeros(32)
    rm = np.zeros(32)
    rv = np.ones(32)
        
    result = batchnorm1d_forward(x, gamma, beta, rm, rv, training=True)
        
    assert not (result is None), "returned None"
        
    out, cache = result
        
    mean = np.mean(out, axis=0)
    var = np.var(out, axis=0)
        
    assert np.allclose(mean, 0, atol=1e-5), f"mean not zero: {np.max(np.abs(mean))}"
        
    assert np.allclose(var, 1, atol=1e-4), f"var not one: {np.max(np.abs(var - 1))}"
        
def test_bn1d_forward_affine():
    """Test BatchNorm1d with non-trivial gamma and beta."""
    np.random.seed(42)
    x = np.random.randn(64, 16)
    gamma = np.array([2.0] * 16)
    beta = np.array([1.0] * 16)
    rm = np.zeros(16)
    rv = np.ones(16)
        
    result = batchnorm1d_forward(x, gamma, beta, rm, rv, training=True)
        
    assert not (result is None), "returned None"
        
    out, cache = result
        
    mean = np.mean(out, axis=0)
    std = np.std(out, axis=0)
        
    assert np.allclose(mean, 1.0, atol=1e-4), f"mean not 1: {mean[0]}"
        
    assert np.allclose(std, 2.0, atol=1e-3), f"std not 2: {std[0]}"
        
def test_bn1d_backward_shape():
    """Test BatchNorm1d backward output shapes."""
    np.random.seed(42)
    x = np.random.randn(16, 8)
    gamma = np.ones(8)
    beta = np.zeros(8)
    rm = np.zeros(8)
    rv = np.ones(8)
        
    fwd_result = batchnorm1d_forward(x, gamma, beta, rm, rv, training=True)
    assert not (fwd_result is None), "forward returned None"
        
    out, cache = fwd_result
    dy = np.random.randn(*out.shape)
        
    bwd_result = batchnorm1d_backward(dy, cache)
    assert not (bwd_result is None), "backward returned None"
        
    dx, dgamma, dbeta = bwd_result
        
    assert not (dx.shape != x.shape), f"dx shape {dx.shape}"
    assert not (dgamma.shape != gamma.shape), f"dgamma shape {dgamma.shape}"
    assert not (dbeta.shape != beta.shape), f"dbeta shape {dbeta.shape}"
        
    # Verify gradients are finite and not all zeros
    assert np.all(np.isfinite(dx)), "dx contains NaN or Inf"
    assert not (np.all(dx == 0)), "dx is all zeros"
        
    # dbeta should equal sum of dy
    expected_dbeta = np.sum(dy, axis=0)
    assert np.allclose(dbeta, expected_dbeta, rtol=1e-5), f"dbeta mismatch: {dbeta[0]} vs {expected_dbeta[0]}"
        
def test_bn1d_numerical_gradient():
    """Numerical gradient check for BatchNorm1d."""
    np.random.seed(42)
        
    bn = BatchNorm1d(8)
    assert not (bn.gamma is None or bn.running_mean is None), "BatchNorm1d not initialized"
        
    x = Tensor(np.random.randn(4, 8))
    bn.train()
        
    y = bn(x)
    assert not (y is None), "forward returned None"
        
    y.sum().backward()
    analytic_grad = x.grad.copy()
        
    eps = 1e-5
    numeric_grad = np.zeros_like(x.data)
        
    for idx in np.ndindex(x.data.shape):
        bn.running_mean = np.zeros(8)
        bn.running_var = np.ones(8)
            
        x_plus = Tensor(x.data.copy())
        x_plus.data[idx] += eps
        y_plus = bn(x_plus)
        loss_plus = y_plus.sum().data if y_plus else 0
            
        bn.running_mean = np.zeros(8)
        bn.running_var = np.ones(8)
            
        x_minus = Tensor(x.data.copy())
        x_minus.data[idx] -= eps
        y_minus = bn(x_minus)
        loss_minus = y_minus.sum().data if y_minus else 0
            
        numeric_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        
    if not np.allclose(analytic_grad, numeric_grad, rtol=1e-2, atol=1e-5):
        max_diff = np.max(np.abs(analytic_grad - numeric_grad))
        assert False, f"Max diff: {max_diff}"
        
def test_bn1d_module_init():
    """Test BatchNorm1d module initialization."""
    bn = BatchNorm1d(64)
        
    assert not (bn.gamma is None), "gamma is None"
    assert not (bn.beta is None), "beta is None"
    assert not (bn.running_mean is None), "running_mean is None"
    assert not (bn.running_var is None), "running_var is None"
        
    assert not (bn.gamma.shape != (64,)), f"gamma shape {bn.gamma.shape}"
        
    # Verify initialization values
    assert np.allclose(bn.gamma.data, 1.0), f"gamma not initialized to 1: {bn.gamma.data[0]}"
    assert np.allclose(bn.beta.data, 0.0), f"beta not initialized to 0: {bn.beta.data[0]}"
    assert np.allclose(bn.running_mean, 0.0), f"running_mean not 0: {bn.running_mean[0]}"
    assert np.allclose(bn.running_var, 1.0), f"running_var not 1: {bn.running_var[0]}"
        
def test_bn1d_module_forward():
    """Test BatchNorm1d module forward."""
    np.random.seed(42)
    bn = BatchNorm1d(32)
        
    assert not (bn.gamma is None), "not initialized"
        
    bn.train()
    x = Tensor(np.random.randn(16, 32))
    y = bn(x)
        
    assert not (y is None), "returned None"
        
    assert not (y.shape != (16, 32)), f"shape {y.shape}"
        
    # Verify normalization
    out_mean = np.mean(y.data, axis=0)
    out_var = np.var(y.data, axis=0)
    assert np.allclose(out_mean, 0, atol=1e-5), f"mean not ~0: {np.max(np.abs(out_mean))}"
    assert np.allclose(out_var, 1, atol=1e-4), f"var not ~1: {np.max(np.abs(out_var - 1))}"
        
def test_bn1d_module_backward():
    """Test BatchNorm1d module backward."""
    np.random.seed(42)
    bn = BatchNorm1d(16)
        
    assert not (bn.gamma is None), "not initialized"
        
    bn.train()
    x = Tensor(np.random.randn(8, 16))
    y = bn(x)
        
    assert not (y is None), "forward returned None"
        
    y.sum().backward()
        
    assert not (np.all(bn.gamma.grad == 0)), "gamma grad is zero"
    assert not (np.all(x.grad == 0)), "x grad is zero"
        
    # For sum loss, dbeta = batch_size = 8
    expected_dbeta = 8.0
    assert np.allclose(bn.beta.grad, expected_dbeta), f"beta grad {bn.beta.grad[0]}, expected {expected_dbeta}"
        
    # Verify gradients are finite
    assert np.all(np.isfinite(x.grad)), "x grad contains NaN or Inf"
        
def test_bn1d_running_stats_update():
    """Test that running statistics are updated."""
    np.random.seed(42)
    bn = BatchNorm1d(16, momentum=0.1)
        
    assert not (bn.running_mean is None), "not initialized"
        
    init_mean = bn.running_mean.copy()
    init_var = bn.running_var.copy()
        
    bn.train()
    for _ in range(10):
        x = Tensor(np.random.randn(32, 16) * 5 + 3)
        y = bn(x)
        
    assert not (np.allclose(bn.running_mean, init_mean)), "running_mean not updated"
    assert not (np.allclose(bn.running_var, init_var)), "running_var not updated"
        
def test_bn1d_eval_mode():
    """Test BatchNorm1d eval mode uses running stats."""
    np.random.seed(42)
    bn = BatchNorm1d(16)
        
    assert not (bn.running_mean is None), "not initialized"
        
    bn.train()
    for _ in range(20):
        x = Tensor(np.random.randn(32, 16))
        y = bn(x)
        
    # Store running stats
    running_mean_copy = bn.running_mean.copy()
    running_var_copy = bn.running_var.copy()
        
    bn.eval()
        
    x1 = Tensor(np.random.randn(1, 16))
    y1 = bn(x1)
        
    assert not (y1 is None), "eval forward returned None"
        
    # Verify running stats weren't updated in eval mode
    assert np.allclose(bn.running_mean, running_mean_copy), "running_mean changed in eval mode"
    assert np.allclose(bn.running_var, running_var_copy), "running_var changed in eval mode"
        
    # Verify eval output uses running stats (manually compute expected)
    expected_y1 = (x1.data - running_mean_copy) / np.sqrt(running_var_copy + bn.eps)
    expected_y1 = bn.gamma.data * expected_y1 + bn.beta.data
    assert np.allclose(y1.data, expected_y1, rtol=1e-5), "eval output doesn't match running stats computation"
        
def test_bn2d_forward_shape():
    """Test BatchNorm2d forward output shape."""
    np.random.seed(42)
    x = np.random.randn(8, 16, 4, 4)
    gamma = np.ones(16)
    beta = np.zeros(16)
    rm = np.zeros(16)
    rv = np.ones(16)
        
    result = batchnorm2d_forward(x, gamma, beta, rm, rv, training=True)
        
    assert not (result is None), "returned None"
        
    out, cache = result
        
    assert not (out.shape != x.shape), f"shape {out.shape}"
        
    # Verify normalization per channel
    mean_per_channel = np.mean(out, axis=(0, 2, 3))
    var_per_channel = np.var(out, axis=(0, 2, 3))
    assert np.allclose(mean_per_channel, 0, atol=1e-5), f"channel means not ~0: max={np.max(np.abs(mean_per_channel))}"
    assert np.allclose(var_per_channel, 1, atol=1e-4), f"channel vars not ~1: max diff={np.max(np.abs(var_per_channel - 1))}"
        
def test_bn2d_forward_normalized():
    """Test BatchNorm2d produces normalized output."""
    np.random.seed(42)
    x = np.random.randn(32, 8, 4, 4) * 3 + 2
    gamma = np.ones(8)
    beta = np.zeros(8)
    rm = np.zeros(8)
    rv = np.ones(8)
        
    result = batchnorm2d_forward(x, gamma, beta, rm, rv, training=True)
        
    assert not (result is None), "returned None"
        
    out, cache = result
        
    mean = np.mean(out, axis=(0, 2, 3))
    var = np.var(out, axis=(0, 2, 3))
        
    assert np.allclose(mean, 0, atol=1e-5), f"mean not zero: {np.max(np.abs(mean))}"
        
    assert np.allclose(var, 1, atol=1e-4), f"var not one"
        
def test_bn2d_backward_shape():
    """Test BatchNorm2d backward output shapes."""
    np.random.seed(42)
    x = np.random.randn(4, 8, 4, 4)
    gamma = np.ones(8)
    beta = np.zeros(8)
    rm = np.zeros(8)
    rv = np.ones(8)
        
    fwd_result = batchnorm2d_forward(x, gamma, beta, rm, rv, training=True)
    assert not (fwd_result is None), "forward returned None"
        
    out, cache = fwd_result
    dy = np.random.randn(*out.shape)
        
    bwd_result = batchnorm2d_backward(dy, cache)
    assert not (bwd_result is None), "backward returned None"
        
    dx, dgamma, dbeta = bwd_result
        
    assert not (dx.shape != x.shape), f"dx shape {dx.shape}"
    assert not (dgamma.shape != (8,)), f"dgamma shape {dgamma.shape}"
    assert not (dbeta.shape != (8,)), f"dbeta shape {dbeta.shape}"
        
    # dbeta should equal sum over (N, H, W)
    expected_dbeta = np.sum(dy, axis=(0, 2, 3))
    assert np.allclose(dbeta, expected_dbeta, rtol=1e-5), f"dbeta mismatch: {dbeta[0]} vs {expected_dbeta[0]}"
        
    # Verify gradients are finite
    assert np.all(np.isfinite(dx)), "dx contains NaN or Inf"
        
def test_bn2d_module_init():
    """Test BatchNorm2d module initialization."""
    bn = BatchNorm2d(64)
        
    assert not (bn.gamma is None), "gamma is None"
    assert not (bn.running_mean is None), "running_mean is None"
        
    # Verify initialization values
    assert np.allclose(bn.gamma.data, 1.0), f"gamma not 1: {bn.gamma.data[0]}"
    assert np.allclose(bn.beta.data, 0.0), f"beta not 0: {bn.beta.data[0]}"
    assert np.allclose(bn.running_mean, 0.0), f"running_mean not 0: {bn.running_mean[0]}"
    assert np.allclose(bn.running_var, 1.0), f"running_var not 1: {bn.running_var[0]}"
        
def test_bn2d_module_forward():
    """Test BatchNorm2d module forward."""
    np.random.seed(42)
    bn = BatchNorm2d(16)
        
    assert not (bn.gamma is None), "not initialized"
        
    bn.train()
    x = Tensor(np.random.randn(4, 16, 8, 8))
    y = bn(x)
        
    assert not (y is None), "returned None"
        
    assert not (y.shape != (4, 16, 8, 8)), f"shape {y.shape}"
        
    # Verify normalization per channel
    mean_per_channel = np.mean(y.data, axis=(0, 2, 3))
    var_per_channel = np.var(y.data, axis=(0, 2, 3))
    assert np.allclose(mean_per_channel, 0, atol=1e-5), f"channel means not ~0"
    assert np.allclose(var_per_channel, 1, atol=1e-4), f"channel vars not ~1"
        
def test_bn2d_module_backward():
    """Test BatchNorm2d module backward."""
    np.random.seed(42)
    bn = BatchNorm2d(8)
        
    assert not (bn.gamma is None), "not initialized"
        
    bn.train()
    x = Tensor(np.random.randn(2, 8, 4, 4))
    y = bn(x)
        
    assert not (y is None), "forward returned None"
        
    y.sum().backward()
        
    assert not (np.all(bn.gamma.grad == 0)), "gamma grad is zero"
    assert not (np.all(x.grad == 0)), "x grad is zero"
        
    # For sum loss, dbeta = N * H * W = 2 * 4 * 4 = 32
    expected_dbeta = 2 * 4 * 4
    assert np.allclose(bn.beta.grad, expected_dbeta), f"beta grad {bn.beta.grad[0]}, expected {expected_dbeta}"
        
    # Verify gradients are finite
    assert np.all(np.isfinite(x.grad)), "x grad contains NaN or Inf"
        
def test_bn_parameters():
    """Test BatchNorm parameters method."""
    bn1 = BatchNorm1d(32, affine=True)
    bn2 = BatchNorm1d(32, affine=False)
        
    params1 = bn1.parameters()
    params2 = bn2.parameters()
        
    assert not (len(params1) != 2), f"affine=True has {len(params1)} params"
        
    assert not (len(params2) != 0), f"affine=False has {len(params2)} params"
        
def test_layernorm_forward():
    """Test LayerNorm forward."""
    np.random.seed(42)
    ln = LayerNorm(16)
    x = Tensor(np.random.randn(4, 16))
    y = ln(x)
        
    assert not (y is None), "returned None"
        
    assert not (y.shape != (4, 16)), f"shape {y.shape}"
        
    # Verify normalization per sample (not per feature like batchnorm)
    for i in range(4):
        sample_mean = np.mean(y.data[i])
        sample_var = np.var(y.data[i])
        assert np.isclose(sample_mean, 0, atol=1e-5), f"sample {i} mean not ~0: {sample_mean}"
        assert np.isclose(sample_var, 1, atol=1e-4), f"sample {i} var not ~1: {sample_var}"
        
def test_layernorm_backward():
    """Test LayerNorm backward."""
    np.random.seed(42)
    ln = LayerNorm(16)
    x = Tensor(np.random.randn(4, 16))
    y = ln(x)
        
    assert not (y is None), "forward returned None"
        
    y.sum().backward()
        
    assert not (np.all(x.grad == 0)), "x grad is zero"
    assert not (np.all(ln.gamma.grad == 0)), "gamma grad is zero"
        
    # For sum loss, dbeta = num_samples = 4
    expected_dbeta = 4.0
    assert np.allclose(ln.beta.grad, expected_dbeta), f"beta grad {ln.beta.grad[0]}, expected {expected_dbeta}"
        
    # Verify gradients are finite
    assert np.all(np.isfinite(x.grad)), "x grad contains NaN or Inf"
        
def test_bn_against_pytorch():
    """Test BatchNorm against PyTorch."""
    import torch
    import torch.nn as nn
        
    np.random.seed(42)
    torch.manual_seed(42)
        
    our_bn = BatchNorm2d(8)
    assert not (our_bn.gamma is None), "not initialized"
        
    torch_bn = nn.BatchNorm2d(8, momentum=0.1)
    torch_bn.weight.data = torch.tensor(our_bn.gamma.data.copy())
    torch_bn.bias.data = torch.tensor(our_bn.beta.data.copy())
    torch_bn.running_mean.data = torch.tensor(our_bn.running_mean.copy())
    torch_bn.running_var.data = torch.tensor(our_bn.running_var.copy())
        
    x_np = np.random.randn(4, 8, 4, 4)
        
    our_bn.train()
    torch_bn.train()
        
    our_x = Tensor(x_np.copy())
    our_y = our_bn(our_x)
        
    torch_x = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
    torch_bn = torch_bn.double()
    torch_y = torch_bn(torch_x)
        
    assert not (our_y is None), "our forward returned None"
        
    if not np.allclose(our_y.data, torch_y.detach().numpy(), rtol=1e-4, atol=1e-6):
        max_diff = np.max(np.abs(our_y.data - torch_y.detach().numpy()))
        assert False, f"Forward mismatch, max diff: {max_diff}"
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])