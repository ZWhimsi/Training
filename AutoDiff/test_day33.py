"""Test Suite for Day 33: Batch Normalization"""

import numpy as np
import sys
from typing import Tuple

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


def test_bn1d_forward_shape() -> Tuple[bool, str]:
    """Test BatchNorm1d forward output shape."""
    try:
        x = np.random.randn(32, 64)
        gamma = np.ones(64)
        beta = np.zeros(64)
        rm = np.zeros(64)
        rv = np.ones(64)
        
        result = batchnorm1d_forward(x, gamma, beta, rm, rv, training=True)
        
        if result is None:
            return False, "returned None"
        
        out, cache = result
        
        if out.shape != x.shape:
            return False, f"shape {out.shape}, expected {x.shape}"
        
        return True, "Output shape correct"
    except Exception as e:
        return False, str(e)


def test_bn1d_forward_normalized() -> Tuple[bool, str]:
    """Test BatchNorm1d produces normalized output."""
    try:
        np.random.seed(42)
        x = np.random.randn(100, 32) * 5 + 3
        gamma = np.ones(32)
        beta = np.zeros(32)
        rm = np.zeros(32)
        rv = np.ones(32)
        
        result = batchnorm1d_forward(x, gamma, beta, rm, rv, training=True)
        
        if result is None:
            return False, "returned None"
        
        out, cache = result
        
        mean = np.mean(out, axis=0)
        var = np.var(out, axis=0)
        
        if not np.allclose(mean, 0, atol=1e-5):
            return False, f"mean not zero: {np.max(np.abs(mean))}"
        
        if not np.allclose(var, 1, atol=1e-4):
            return False, f"var not one: {np.max(np.abs(var - 1))}"
        
        return True, "Output is normalized"
    except Exception as e:
        return False, str(e)


def test_bn1d_forward_affine() -> Tuple[bool, str]:
    """Test BatchNorm1d with non-trivial gamma and beta."""
    try:
        np.random.seed(42)
        x = np.random.randn(64, 16)
        gamma = np.array([2.0] * 16)
        beta = np.array([1.0] * 16)
        rm = np.zeros(16)
        rv = np.ones(16)
        
        result = batchnorm1d_forward(x, gamma, beta, rm, rv, training=True)
        
        if result is None:
            return False, "returned None"
        
        out, cache = result
        
        mean = np.mean(out, axis=0)
        std = np.std(out, axis=0)
        
        if not np.allclose(mean, 1.0, atol=1e-4):
            return False, f"mean not 1: {mean[0]}"
        
        if not np.allclose(std, 2.0, atol=1e-3):
            return False, f"std not 2: {std[0]}"
        
        return True, "Affine transform works"
    except Exception as e:
        return False, str(e)


def test_bn1d_backward_shape() -> Tuple[bool, str]:
    """Test BatchNorm1d backward output shapes."""
    try:
        np.random.seed(42)
        x = np.random.randn(16, 8)
        gamma = np.ones(8)
        beta = np.zeros(8)
        rm = np.zeros(8)
        rv = np.ones(8)
        
        fwd_result = batchnorm1d_forward(x, gamma, beta, rm, rv, training=True)
        if fwd_result is None:
            return False, "forward returned None"
        
        out, cache = fwd_result
        dy = np.random.randn(*out.shape)
        
        bwd_result = batchnorm1d_backward(dy, cache)
        if bwd_result is None:
            return False, "backward returned None"
        
        dx, dgamma, dbeta = bwd_result
        
        if dx.shape != x.shape:
            return False, f"dx shape {dx.shape}"
        if dgamma.shape != gamma.shape:
            return False, f"dgamma shape {dgamma.shape}"
        if dbeta.shape != beta.shape:
            return False, f"dbeta shape {dbeta.shape}"
        
        return True, "Gradient shapes correct"
    except Exception as e:
        return False, str(e)


def test_bn1d_numerical_gradient() -> Tuple[bool, str]:
    """Numerical gradient check for BatchNorm1d."""
    try:
        np.random.seed(42)
        
        bn = BatchNorm1d(8)
        if bn.gamma is None or bn.running_mean is None:
            return False, "BatchNorm1d not initialized"
        
        x = Tensor(np.random.randn(4, 8))
        bn.train()
        
        y = bn(x)
        if y is None:
            return False, "forward returned None"
        
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
            return False, f"Max diff: {max_diff}"
        
        return True, "Numerical gradient check passed"
    except Exception as e:
        return False, str(e)


def test_bn1d_module_init() -> Tuple[bool, str]:
    """Test BatchNorm1d module initialization."""
    try:
        bn = BatchNorm1d(64)
        
        if bn.gamma is None:
            return False, "gamma is None"
        if bn.beta is None:
            return False, "beta is None"
        if bn.running_mean is None:
            return False, "running_mean is None"
        if bn.running_var is None:
            return False, "running_var is None"
        
        if bn.gamma.shape != (64,):
            return False, f"gamma shape {bn.gamma.shape}"
        
        return True, "Initialization correct"
    except Exception as e:
        return False, str(e)


def test_bn1d_module_forward() -> Tuple[bool, str]:
    """Test BatchNorm1d module forward."""
    try:
        np.random.seed(42)
        bn = BatchNorm1d(32)
        
        if bn.gamma is None:
            return False, "not initialized"
        
        bn.train()
        x = Tensor(np.random.randn(16, 32))
        y = bn(x)
        
        if y is None:
            return False, "returned None"
        
        if y.shape != (16, 32):
            return False, f"shape {y.shape}"
        
        return True, "Forward works"
    except Exception as e:
        return False, str(e)


def test_bn1d_module_backward() -> Tuple[bool, str]:
    """Test BatchNorm1d module backward."""
    try:
        np.random.seed(42)
        bn = BatchNorm1d(16)
        
        if bn.gamma is None:
            return False, "not initialized"
        
        bn.train()
        x = Tensor(np.random.randn(8, 16))
        y = bn(x)
        
        if y is None:
            return False, "forward returned None"
        
        y.sum().backward()
        
        if np.all(bn.gamma.grad == 0):
            return False, "gamma grad is zero"
        if np.all(x.grad == 0):
            return False, "x grad is zero"
        
        return True, "Backward works"
    except Exception as e:
        return False, str(e)


def test_bn1d_running_stats_update() -> Tuple[bool, str]:
    """Test that running statistics are updated."""
    try:
        np.random.seed(42)
        bn = BatchNorm1d(16, momentum=0.1)
        
        if bn.running_mean is None:
            return False, "not initialized"
        
        init_mean = bn.running_mean.copy()
        init_var = bn.running_var.copy()
        
        bn.train()
        for _ in range(10):
            x = Tensor(np.random.randn(32, 16) * 5 + 3)
            y = bn(x)
        
        if np.allclose(bn.running_mean, init_mean):
            return False, "running_mean not updated"
        if np.allclose(bn.running_var, init_var):
            return False, "running_var not updated"
        
        return True, "Running stats updated"
    except Exception as e:
        return False, str(e)


def test_bn1d_eval_mode() -> Tuple[bool, str]:
    """Test BatchNorm1d eval mode uses running stats."""
    try:
        np.random.seed(42)
        bn = BatchNorm1d(16)
        
        if bn.running_mean is None:
            return False, "not initialized"
        
        bn.train()
        for _ in range(20):
            x = Tensor(np.random.randn(32, 16))
            y = bn(x)
        
        bn.eval()
        
        x1 = Tensor(np.random.randn(1, 16))
        y1 = bn(x1)
        
        if y1 is None:
            return False, "eval forward returned None"
        
        x2 = Tensor(np.random.randn(64, 16))
        y2 = bn(x2)
        
        if y2 is None:
            return False, "eval forward returned None"
        
        return True, "Eval mode works"
    except Exception as e:
        return False, str(e)


def test_bn2d_forward_shape() -> Tuple[bool, str]:
    """Test BatchNorm2d forward output shape."""
    try:
        x = np.random.randn(8, 16, 4, 4)
        gamma = np.ones(16)
        beta = np.zeros(16)
        rm = np.zeros(16)
        rv = np.ones(16)
        
        result = batchnorm2d_forward(x, gamma, beta, rm, rv, training=True)
        
        if result is None:
            return False, "returned None"
        
        out, cache = result
        
        if out.shape != x.shape:
            return False, f"shape {out.shape}"
        
        return True, "Output shape correct"
    except Exception as e:
        return False, str(e)


def test_bn2d_forward_normalized() -> Tuple[bool, str]:
    """Test BatchNorm2d produces normalized output."""
    try:
        np.random.seed(42)
        x = np.random.randn(32, 8, 4, 4) * 3 + 2
        gamma = np.ones(8)
        beta = np.zeros(8)
        rm = np.zeros(8)
        rv = np.ones(8)
        
        result = batchnorm2d_forward(x, gamma, beta, rm, rv, training=True)
        
        if result is None:
            return False, "returned None"
        
        out, cache = result
        
        mean = np.mean(out, axis=(0, 2, 3))
        var = np.var(out, axis=(0, 2, 3))
        
        if not np.allclose(mean, 0, atol=1e-5):
            return False, f"mean not zero: {np.max(np.abs(mean))}"
        
        if not np.allclose(var, 1, atol=1e-4):
            return False, f"var not one"
        
        return True, "Output is normalized"
    except Exception as e:
        return False, str(e)


def test_bn2d_backward_shape() -> Tuple[bool, str]:
    """Test BatchNorm2d backward output shapes."""
    try:
        np.random.seed(42)
        x = np.random.randn(4, 8, 4, 4)
        gamma = np.ones(8)
        beta = np.zeros(8)
        rm = np.zeros(8)
        rv = np.ones(8)
        
        fwd_result = batchnorm2d_forward(x, gamma, beta, rm, rv, training=True)
        if fwd_result is None:
            return False, "forward returned None"
        
        out, cache = fwd_result
        dy = np.random.randn(*out.shape)
        
        bwd_result = batchnorm2d_backward(dy, cache)
        if bwd_result is None:
            return False, "backward returned None"
        
        dx, dgamma, dbeta = bwd_result
        
        if dx.shape != x.shape:
            return False, f"dx shape {dx.shape}"
        if dgamma.shape != (8,):
            return False, f"dgamma shape {dgamma.shape}"
        if dbeta.shape != (8,):
            return False, f"dbeta shape {dbeta.shape}"
        
        return True, "Gradient shapes correct"
    except Exception as e:
        return False, str(e)


def test_bn2d_module_init() -> Tuple[bool, str]:
    """Test BatchNorm2d module initialization."""
    try:
        bn = BatchNorm2d(64)
        
        if bn.gamma is None:
            return False, "gamma is None"
        if bn.running_mean is None:
            return False, "running_mean is None"
        
        return True, "Initialization correct"
    except Exception as e:
        return False, str(e)


def test_bn2d_module_forward() -> Tuple[bool, str]:
    """Test BatchNorm2d module forward."""
    try:
        np.random.seed(42)
        bn = BatchNorm2d(16)
        
        if bn.gamma is None:
            return False, "not initialized"
        
        bn.train()
        x = Tensor(np.random.randn(4, 16, 8, 8))
        y = bn(x)
        
        if y is None:
            return False, "returned None"
        
        if y.shape != (4, 16, 8, 8):
            return False, f"shape {y.shape}"
        
        return True, "Forward works"
    except Exception as e:
        return False, str(e)


def test_bn2d_module_backward() -> Tuple[bool, str]:
    """Test BatchNorm2d module backward."""
    try:
        np.random.seed(42)
        bn = BatchNorm2d(8)
        
        if bn.gamma is None:
            return False, "not initialized"
        
        bn.train()
        x = Tensor(np.random.randn(2, 8, 4, 4))
        y = bn(x)
        
        if y is None:
            return False, "forward returned None"
        
        y.sum().backward()
        
        if np.all(bn.gamma.grad == 0):
            return False, "gamma grad is zero"
        if np.all(x.grad == 0):
            return False, "x grad is zero"
        
        return True, "Backward works"
    except Exception as e:
        return False, str(e)


def test_bn_parameters() -> Tuple[bool, str]:
    """Test BatchNorm parameters method."""
    try:
        bn1 = BatchNorm1d(32, affine=True)
        bn2 = BatchNorm1d(32, affine=False)
        
        params1 = bn1.parameters()
        params2 = bn2.parameters()
        
        if len(params1) != 2:
            return False, f"affine=True has {len(params1)} params"
        
        if len(params2) != 0:
            return False, f"affine=False has {len(params2)} params"
        
        return True, "Parameters correct"
    except Exception as e:
        return False, str(e)


def test_layernorm_forward() -> Tuple[bool, str]:
    """Test LayerNorm forward."""
    try:
        ln = LayerNorm(16)
        x = Tensor(np.random.randn(4, 16))
        y = ln(x)
        
        if y is None:
            return False, "returned None"
        
        if y.shape != (4, 16):
            return False, f"shape {y.shape}"
        
        return True, "LayerNorm forward works"
    except Exception as e:
        return False, str(e)


def test_layernorm_backward() -> Tuple[bool, str]:
    """Test LayerNorm backward."""
    try:
        ln = LayerNorm(16)
        x = Tensor(np.random.randn(4, 16))
        y = ln(x)
        
        if y is None:
            return False, "forward returned None"
        
        y.sum().backward()
        
        if np.all(x.grad == 0):
            return False, "x grad is zero"
        if np.all(ln.gamma.grad == 0):
            return False, "gamma grad is zero"
        
        return True, "LayerNorm backward works"
    except Exception as e:
        return False, str(e)


def test_bn_against_pytorch() -> Tuple[bool, str]:
    """Test BatchNorm against PyTorch."""
    try:
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        our_bn = BatchNorm2d(8)
        if our_bn.gamma is None:
            return False, "not initialized"
        
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
        
        if our_y is None:
            return False, "our forward returned None"
        
        if not np.allclose(our_y.data, torch_y.detach().numpy(), rtol=1e-4, atol=1e-6):
            max_diff = np.max(np.abs(our_y.data - torch_y.detach().numpy()))
            return False, f"Forward mismatch, max diff: {max_diff}"
        
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("bn1d_forward_shape", test_bn1d_forward_shape),
        ("bn1d_forward_normalized", test_bn1d_forward_normalized),
        ("bn1d_forward_affine", test_bn1d_forward_affine),
        ("bn1d_backward_shape", test_bn1d_backward_shape),
        ("bn1d_numerical_gradient", test_bn1d_numerical_gradient),
        ("bn1d_module_init", test_bn1d_module_init),
        ("bn1d_module_forward", test_bn1d_module_forward),
        ("bn1d_module_backward", test_bn1d_module_backward),
        ("bn1d_running_stats_update", test_bn1d_running_stats_update),
        ("bn1d_eval_mode", test_bn1d_eval_mode),
        ("bn2d_forward_shape", test_bn2d_forward_shape),
        ("bn2d_forward_normalized", test_bn2d_forward_normalized),
        ("bn2d_backward_shape", test_bn2d_backward_shape),
        ("bn2d_module_init", test_bn2d_module_init),
        ("bn2d_module_forward", test_bn2d_module_forward),
        ("bn2d_module_backward", test_bn2d_module_backward),
        ("bn_parameters", test_bn_parameters),
        ("layernorm_forward", test_layernorm_forward),
        ("layernorm_backward", test_layernorm_backward),
        ("bn_against_pytorch", test_bn_against_pytorch),
    ]
    
    print(f"\n{'='*60}")
    print("Day 33: Batch Normalization - Tests")
    print(f"{'='*60}")
    
    passed = 0
    for name, fn in tests:
        try:
            p, m = fn()
        except Exception as e:
            p, m = False, str(e)
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    
    print(f"\nSummary: {passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    success = run_all_tests()
    sys.exit(0 if success else 1)
