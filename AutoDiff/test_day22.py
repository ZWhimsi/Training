"""Test Suite for Day 22: Linear Layer Module"""

import numpy as np
import sys
from typing import Tuple

try:
    from day22 import (
        Tensor,
        Linear,
        LinearKaiming,
        LinearNoBias,
        MLP,
        numerical_gradient,
        test_linear_forward,
        test_linear_backward,
        test_xavier_init,
        test_kaiming_init,
        test_mlp_forward,
        test_mlp_backward,
        test_gradient_check
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_linear_weight_shape() -> Tuple[bool, str]:
    """Test linear layer weight shape."""
    try:
        layer = Linear(10, 5)
        
        if layer.weight is None:
            return False, "weight is None"
        if layer.weight.shape != (5, 10):
            return False, f"shape = {layer.weight.shape}, expected (5, 10)"
        
        return True, "Weight shape (out, in) correct"
    except Exception as e:
        return False, str(e)


def test_linear_bias_shape() -> Tuple[bool, str]:
    """Test linear layer bias shape."""
    try:
        layer = Linear(10, 5)
        
        if layer.bias is None:
            return False, "bias is None"
        if layer.bias.shape != (5,):
            return False, f"shape = {layer.bias.shape}, expected (5,)"
        
        return True, "Bias shape (out,) correct"
    except Exception as e:
        return False, str(e)


def test_linear_no_bias() -> Tuple[bool, str]:
    """Test linear layer without bias."""
    try:
        layer = Linear(10, 5, bias=False)
        
        if layer.weight is None:
            return False, "weight is None"
        
        params = layer.parameters()
        if len(params) != 1:
            return False, f"Expected 1 param, got {len(params)}"
        
        return True, "No bias mode works"
    except Exception as e:
        return False, str(e)


def test_linear_forward_batch() -> Tuple[bool, str]:
    """Test linear forward with batch."""
    try:
        np.random.seed(42)
        layer = Linear(4, 3)
        
        if layer.weight is None:
            return False, "weight is None"
        
        x = Tensor(np.random.randn(8, 4))
        y = layer(x)
        
        if y is None:
            return False, "forward returned None"
        if y.shape != (8, 3):
            return False, f"shape = {y.shape}, expected (8, 3)"
        
        return True, "Batch forward shape correct"
    except Exception as e:
        return False, str(e)


def test_linear_forward_single() -> Tuple[bool, str]:
    """Test linear forward with single sample (1D input)."""
    try:
        np.random.seed(42)
        layer = Linear(4, 3)
        
        if layer.weight is None:
            return False, "weight is None"
        
        x = Tensor(np.random.randn(4))
        y = layer(x)
        
        if y is None:
            return False, "forward returned None"
        
        return True, "1D input handled"
    except Exception as e:
        return False, str(e)


def test_linear_forward_values() -> Tuple[bool, str]:
    """Test linear forward computes correct values."""
    try:
        layer = Linear(3, 2)
        
        if layer.weight is None or layer.bias is None:
            return False, "weight or bias is None"
        
        layer.weight.data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        layer.bias.data = np.array([0.1, 0.2], dtype=np.float64)
        
        x = Tensor([[1, 0, 1]])  # (1, 3)
        y = layer(x)
        
        if y is None:
            return False, "forward returned None"
        
        expected = np.array([[1*1 + 2*0 + 3*1 + 0.1, 4*1 + 5*0 + 6*1 + 0.2]])
        if not np.allclose(y.data, expected):
            return False, f"values = {y.data}, expected {expected}"
        
        return True, "Forward values correct"
    except Exception as e:
        return False, str(e)


def test_linear_backward_weight() -> Tuple[bool, str]:
    """Test weight gradient computation."""
    try:
        np.random.seed(42)
        layer = Linear(3, 2)
        
        if layer.weight is None:
            return False, "weight is None"
        
        x = Tensor([[1.0, 2.0, 3.0]])
        y = layer(x)
        
        if y is None:
            return False, "forward returned None"
        
        loss = y.sum()
        loss.backward()
        
        if np.all(layer.weight.grad == 0):
            return False, "weight grad is all zeros"
        
        return True, "Weight gradient computed"
    except Exception as e:
        return False, str(e)


def test_linear_backward_bias() -> Tuple[bool, str]:
    """Test bias gradient computation."""
    try:
        np.random.seed(42)
        layer = Linear(3, 2)
        
        if layer.bias is None:
            return False, "bias is None"
        
        x = Tensor([[1.0, 2.0, 3.0]])
        y = layer(x)
        
        if y is None:
            return False, "forward returned None"
        
        loss = y.sum()
        loss.backward()
        
        if not np.allclose(layer.bias.grad, [1, 1]):
            return False, f"bias grad = {layer.bias.grad}, expected [1, 1]"
        
        return True, "Bias gradient = sum of upstream"
    except Exception as e:
        return False, str(e)


def test_linear_backward_input() -> Tuple[bool, str]:
    """Test input gradient computation."""
    try:
        np.random.seed(42)
        layer = Linear(3, 2)
        
        if layer.weight is None:
            return False, "weight is None"
        
        x = Tensor([[1.0, 2.0, 3.0]])
        y = layer(x)
        
        if y is None:
            return False, "forward returned None"
        
        loss = y.sum()
        loss.backward()
        
        if np.all(x.grad == 0):
            return False, "input grad is all zeros"
        
        return True, "Input gradient computed"
    except Exception as e:
        return False, str(e)


def test_xavier_variance() -> Tuple[bool, str]:
    """Test Xavier initialization has correct variance."""
    try:
        np.random.seed(42)
        variances = []
        
        for _ in range(20):
            layer = Linear(100, 50)
            if layer.weight is not None:
                variances.append(np.var(layer.weight.data))
        
        if not variances:
            return False, "No weights created"
        
        mean_var = np.mean(variances)
        expected = 2.0 / (100 + 50)
        
        if abs(mean_var - expected) > 0.005:
            return False, f"variance = {mean_var:.4f}, expected {expected:.4f}"
        
        return True, f"Xavier variance ~= {expected:.4f}"
    except Exception as e:
        return False, str(e)


def test_kaiming_variance() -> Tuple[bool, str]:
    """Test Kaiming initialization has correct variance."""
    try:
        np.random.seed(42)
        variances = []
        
        for _ in range(20):
            layer = LinearKaiming(100, 50)
            if layer.weight is not None:
                variances.append(np.var(layer.weight.data))
        
        if not variances:
            return False, "No weights created"
        
        mean_var = np.mean(variances)
        expected = 2.0 / 100
        
        if abs(mean_var - expected) > 0.008:
            return False, f"variance = {mean_var:.4f}, expected {expected:.4f}"
        
        return True, f"Kaiming variance ~= {expected:.4f}"
    except Exception as e:
        return False, str(e)


def test_linear_no_bias_forward() -> Tuple[bool, str]:
    """Test LinearNoBias forward."""
    try:
        np.random.seed(42)
        layer = LinearNoBias(4, 3)
        
        if layer.weight is None:
            return False, "weight is None"
        
        x = Tensor(np.random.randn(2, 4))
        y = layer(x)
        
        if y is None:
            return False, "forward returned None"
        if y.shape != (2, 3):
            return False, f"shape = {y.shape}"
        
        expected = x.data @ layer.weight.data.T
        if not np.allclose(y.data, expected):
            return False, "Values don't match xW^T"
        
        return True, "LinearNoBias forward correct"
    except Exception as e:
        return False, str(e)


def test_mlp_construction() -> Tuple[bool, str]:
    """Test MLP layer construction."""
    try:
        mlp = MLP(784, [256, 128], 10)
        
        if not mlp.layers:
            return False, "No layers created"
        if len(mlp.layers) != 3:
            return False, f"Expected 3 layers, got {len(mlp.layers)}"
        
        return True, "MLP layers constructed"
    except Exception as e:
        return False, str(e)


def test_mlp_layer_sizes() -> Tuple[bool, str]:
    """Test MLP layer sizes."""
    try:
        mlp = MLP(784, [256, 128], 10)
        
        if not mlp.layers or len(mlp.layers) < 3:
            return False, "Layers not constructed"
        
        if mlp.layers[0].in_features != 784 or mlp.layers[0].out_features != 256:
            return False, f"Layer 0 size wrong"
        if mlp.layers[1].in_features != 256 or mlp.layers[1].out_features != 128:
            return False, f"Layer 1 size wrong"
        if mlp.layers[2].in_features != 128 or mlp.layers[2].out_features != 10:
            return False, f"Layer 2 size wrong"
        
        return True, "MLP layer sizes correct"
    except Exception as e:
        return False, str(e)


def test_mlp_forward() -> Tuple[bool, str]:
    """Test MLP forward pass."""
    try:
        np.random.seed(42)
        mlp = MLP(10, [20, 15], 5)
        
        x = Tensor(np.random.randn(4, 10))
        y = mlp(x)
        
        if y is None:
            return False, "forward returned None"
        if y.shape != (4, 5):
            return False, f"shape = {y.shape}, expected (4, 5)"
        
        return True, "MLP forward shape correct"
    except Exception as e:
        return False, str(e)


def test_mlp_relu_applied() -> Tuple[bool, str]:
    """Test that ReLU is applied in hidden layers."""
    try:
        np.random.seed(123)
        mlp = MLP(10, [20], 5)
        
        x = Tensor(np.random.randn(4, 10) * 3)
        y = mlp(x)
        
        if y is None:
            return False, "forward returned None"
        
        return True, "MLP with ReLU runs"
    except Exception as e:
        return False, str(e)


def test_mlp_backward() -> Tuple[bool, str]:
    """Test MLP backward pass."""
    try:
        np.random.seed(42)
        mlp = MLP(10, [20], 5)
        mlp.zero_grad()
        
        x = Tensor(np.random.randn(4, 10))
        y = mlp(x)
        
        if y is None:
            return False, "forward returned None"
        
        loss = y.sum()
        loss.backward()
        
        params = mlp.parameters()
        if not params:
            return False, "No parameters"
        
        all_have_grad = all(np.any(p.grad != 0) for p in params)
        if not all_have_grad:
            return False, "Some params have zero gradient"
        
        return True, "All MLP params have gradients"
    except Exception as e:
        return False, str(e)


def test_mlp_parameters() -> Tuple[bool, str]:
    """Test MLP parameter collection."""
    try:
        mlp = MLP(10, [20, 15], 5)
        params = mlp.parameters()
        
        expected_count = 3 * 2
        if len(params) != expected_count:
            return False, f"Expected {expected_count} params, got {len(params)}"
        
        return True, f"MLP has {expected_count} parameters"
    except Exception as e:
        return False, str(e)


def test_gradient_numerical() -> Tuple[bool, str]:
    """Test numerical gradient computation."""
    try:
        np.random.seed(42)
        layer = Linear(3, 2)
        
        if layer.weight is None:
            return False, "weight is None"
        
        x = Tensor([[1.0, 2.0, 3.0]])
        
        def f(inp):
            out = layer.forward(inp)
            return out.sum() if out is not None else Tensor(0)
        
        num_grad = numerical_gradient(f, x)
        
        if np.all(num_grad == 0):
            return False, "Numerical gradient is all zeros"
        
        y = layer(x)
        if y is None:
            return False, "forward returned None"
        
        loss = y.sum()
        loss.backward()
        
        if not np.allclose(x.grad, num_grad, rtol=1e-4, atol=1e-6):
            return False, f"Gradients don't match: {x.grad} vs {num_grad}"
        
        return True, "Numerical gradient matches analytical"
    except Exception as e:
        return False, str(e)


def test_zero_grad() -> Tuple[bool, str]:
    """Test zero_grad functionality."""
    try:
        np.random.seed(42)
        layer = Linear(3, 2)
        
        if layer.weight is None:
            return False, "weight is None"
        
        x = Tensor([[1.0, 2.0, 3.0]])
        y = layer(x)
        if y is not None:
            y.sum().backward()
        
        if np.all(layer.weight.grad == 0):
            return False, "Gradient wasn't computed"
        
        layer.zero_grad()
        
        if not np.all(layer.weight.grad == 0):
            return False, "zero_grad didn't reset weight grad"
        if layer.bias is not None and not np.all(layer.bias.grad == 0):
            return False, "zero_grad didn't reset bias grad"
        
        return True, "zero_grad works"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test against PyTorch reference."""
    try:
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        in_feat, out_feat = 4, 3
        batch_size = 2
        
        our_layer = Linear(in_feat, out_feat)
        if our_layer.weight is None or our_layer.bias is None:
            return False, "weight or bias is None"
        
        torch_layer = nn.Linear(in_feat, out_feat)
        torch_layer.weight.data = torch.tensor(our_layer.weight.data.copy())
        torch_layer.bias.data = torch.tensor(our_layer.bias.data.copy())
        
        x_np = np.random.randn(batch_size, in_feat)
        
        our_x = Tensor(x_np.copy())
        our_y = our_layer(our_x)
        if our_y is None:
            return False, "our forward returned None"
        our_y.sum().backward()
        
        torch_x = torch.tensor(x_np, requires_grad=True)
        torch_y = torch_layer(torch_x)
        torch_y.sum().backward()
        
        if not np.allclose(our_y.data, torch_y.detach().numpy(), rtol=1e-5):
            return False, "Forward mismatch"
        
        if not np.allclose(our_layer.weight.grad, torch_layer.weight.grad.numpy(), rtol=1e-5):
            return False, "Weight gradient mismatch"
        
        if not np.allclose(our_layer.bias.grad, torch_layer.bias.grad.numpy(), rtol=1e-5):
            return False, "Bias gradient mismatch"
        
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("linear_weight_shape", test_linear_weight_shape),
        ("linear_bias_shape", test_linear_bias_shape),
        ("linear_no_bias", test_linear_no_bias),
        ("linear_forward_batch", test_linear_forward_batch),
        ("linear_forward_single", test_linear_forward_single),
        ("linear_forward_values", test_linear_forward_values),
        ("linear_backward_weight", test_linear_backward_weight),
        ("linear_backward_bias", test_linear_backward_bias),
        ("linear_backward_input", test_linear_backward_input),
        ("xavier_variance", test_xavier_variance),
        ("kaiming_variance", test_kaiming_variance),
        ("linear_no_bias_forward", test_linear_no_bias_forward),
        ("mlp_construction", test_mlp_construction),
        ("mlp_layer_sizes", test_mlp_layer_sizes),
        ("mlp_forward", test_mlp_forward),
        ("mlp_relu_applied", test_mlp_relu_applied),
        ("mlp_backward", test_mlp_backward),
        ("mlp_parameters", test_mlp_parameters),
        ("gradient_numerical", test_gradient_numerical),
        ("zero_grad", test_zero_grad),
        ("against_pytorch", test_against_pytorch),
    ]
    
    print(f"\n{'='*60}")
    print("Day 22: Linear Layer Module - Tests")
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
