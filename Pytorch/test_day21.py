"""Test Suite for Day 21: Weight Initialization Strategies"""

import torch
import torch.nn as nn
import math
import sys
from typing import Tuple

try:
    from day21 import (xavier_uniform_, xavier_normal_,
                       kaiming_uniform_, kaiming_normal_,
                       orthogonal_, zeros_, ones_, constant_,
                       normal_, truncated_normal_,
                       init_transformer_weights, init_gpt_weights,
                       SimpleTransformerBlock, apply_custom_init, smart_init)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_xavier_uniform_shape() -> Tuple[bool, str]:
    """Test Xavier uniform preserves tensor shape."""
    try:
        tensor = torch.empty(128, 64)
        result = xavier_uniform_(tensor)
        
        if result is None:
            return False, "Function returned None"
        if result.shape != (128, 64):
            return False, f"Shape changed: {result.shape}"
        
        return True, f"Shape preserved: {result.shape}"
    except Exception as e:
        return False, str(e)


def test_xavier_uniform_bounds() -> Tuple[bool, str]:
    """Test Xavier uniform has correct bounds."""
    try:
        fan_in, fan_out = 64, 128
        tensor = torch.empty(fan_out, fan_in)
        result = xavier_uniform_(tensor)
        
        if result is None:
            return False, "Function returned None"
        
        # Expected bound
        expected_bound = math.sqrt(6.0 / (fan_in + fan_out))
        
        # Check values are within bounds (with small tolerance)
        if result.max().item() > expected_bound * 1.01:
            return False, f"Max {result.max().item():.4f} > bound {expected_bound:.4f}"
        if result.min().item() < -expected_bound * 1.01:
            return False, f"Min {result.min().item():.4f} < -bound {-expected_bound:.4f}"
        
        return True, f"Values in [-{expected_bound:.4f}, {expected_bound:.4f}]"
    except Exception as e:
        return False, str(e)


def test_xavier_normal_statistics() -> Tuple[bool, str]:
    """Test Xavier normal has correct mean and std."""
    try:
        fan_in, fan_out = 256, 256
        tensor = torch.empty(fan_out, fan_in)
        result = xavier_normal_(tensor)
        
        if result is None:
            return False, "Function returned None"
        
        expected_std = math.sqrt(2.0 / (fan_in + fan_out))
        actual_std = result.std().item()
        
        # Mean should be close to 0
        if abs(result.mean().item()) > 0.1:
            return False, f"Mean not ~0: {result.mean().item():.4f}"
        
        # Std should be close to expected
        if abs(actual_std - expected_std) > 0.05:
            return False, f"Std {actual_std:.4f} != expected {expected_std:.4f}"
        
        return True, f"Mean~0, Std~{expected_std:.4f}"
    except Exception as e:
        return False, str(e)


def test_xavier_vs_pytorch() -> Tuple[bool, str]:
    """Compare Xavier with PyTorch's implementation."""
    try:
        torch.manual_seed(42)
        tensor1 = torch.empty(64, 64)
        xavier_uniform_(tensor1)
        
        torch.manual_seed(42)
        tensor2 = torch.empty(64, 64)
        nn.init.xavier_uniform_(tensor2)
        
        # Should have similar statistics (not identical due to implementation)
        if abs(tensor1.std().item() - tensor2.std().item()) > 0.05:
            return False, "Std differs significantly from PyTorch"
        
        return True, "Similar to PyTorch implementation"
    except Exception as e:
        return False, str(e)


def test_kaiming_uniform_shape() -> Tuple[bool, str]:
    """Test Kaiming uniform preserves shape."""
    try:
        tensor = torch.empty(128, 64)
        result = kaiming_uniform_(tensor)
        
        if result is None:
            return False, "Function returned None"
        if result.shape != (128, 64):
            return False, f"Shape changed: {result.shape}"
        
        return True, f"Shape preserved: {result.shape}"
    except Exception as e:
        return False, str(e)


def test_kaiming_normal_statistics() -> Tuple[bool, str]:
    """Test Kaiming normal has correct statistics for ReLU."""
    try:
        fan_in = 256
        tensor = torch.empty(128, fan_in)
        result = kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')
        
        if result is None:
            return False, "Function returned None"
        
        # For ReLU: std = sqrt(2/fan_in)
        expected_std = math.sqrt(2.0 / fan_in)
        actual_std = result.std().item()
        
        if abs(result.mean().item()) > 0.1:
            return False, f"Mean not ~0: {result.mean().item():.4f}"
        
        if abs(actual_std - expected_std) > 0.05:
            return False, f"Std {actual_std:.4f} != expected {expected_std:.4f}"
        
        return True, f"Kaiming normal: std~{expected_std:.4f}"
    except Exception as e:
        return False, str(e)


def test_kaiming_fan_out() -> Tuple[bool, str]:
    """Test Kaiming with fan_out mode."""
    try:
        fan_out = 256
        tensor = torch.empty(fan_out, 128)
        result = kaiming_normal_(tensor, mode='fan_out', nonlinearity='relu')
        
        if result is None:
            return False, "Function returned None"
        
        expected_std = math.sqrt(2.0 / fan_out)
        actual_std = result.std().item()
        
        if abs(actual_std - expected_std) > 0.05:
            return False, f"Std {actual_std:.4f} != expected {expected_std:.4f}"
        
        return True, f"Fan_out mode: std~{expected_std:.4f}"
    except Exception as e:
        return False, str(e)


def test_kaiming_leaky_relu() -> Tuple[bool, str]:
    """Test Kaiming for leaky ReLU."""
    try:
        fan_in = 256
        a = 0.2  # Negative slope
        tensor = torch.empty(128, fan_in)
        result = kaiming_normal_(tensor, a=a, mode='fan_in', nonlinearity='leaky_relu')
        
        if result is None:
            return False, "Function returned None"
        
        # For Leaky ReLU: gain = sqrt(2 / (1 + a^2))
        gain = math.sqrt(2.0 / (1 + a ** 2))
        expected_std = gain / math.sqrt(fan_in)
        actual_std = result.std().item()
        
        if abs(actual_std - expected_std) > 0.05:
            return False, f"Std {actual_std:.4f} != expected {expected_std:.4f}"
        
        return True, f"Leaky ReLU (a={a}): std~{expected_std:.4f}"
    except Exception as e:
        return False, str(e)


def test_orthogonal_shape() -> Tuple[bool, str]:
    """Test orthogonal initialization preserves shape."""
    try:
        tensor = torch.empty(64, 64)
        result = orthogonal_(tensor)
        
        if result is None:
            return False, "Function returned None"
        if result.shape != (64, 64):
            return False, f"Shape changed: {result.shape}"
        
        return True, f"Shape preserved: {result.shape}"
    except Exception as e:
        return False, str(e)


def test_orthogonal_is_orthogonal() -> Tuple[bool, str]:
    """Test that orthogonal init produces orthogonal matrix."""
    try:
        n = 64
        tensor = torch.empty(n, n)
        result = orthogonal_(tensor)
        
        if result is None:
            return False, "Function returned None"
        
        # W @ W^T should be identity
        identity = torch.eye(n)
        product = torch.mm(result, result.t())
        error = (product - identity).abs().mean().item()
        
        if error > 0.01:
            return False, f"Not orthogonal: error={error:.6f}"
        
        return True, f"Orthogonality error: {error:.6f}"
    except Exception as e:
        return False, str(e)


def test_orthogonal_non_square() -> Tuple[bool, str]:
    """Test orthogonal init with non-square matrix."""
    try:
        tensor = torch.empty(128, 64)  # More rows than cols
        result = orthogonal_(tensor)
        
        if result is None:
            return False, "Function returned None"
        if result.shape != (128, 64):
            return False, f"Shape changed: {result.shape}"
        
        # Columns should be orthonormal
        # W^T @ W should be identity(64x64)
        product = torch.mm(result.t(), result)
        identity = torch.eye(64)
        error = (product - identity).abs().mean().item()
        
        if error > 0.01:
            return False, f"Columns not orthonormal: error={error:.6f}"
        
        return True, "Non-square orthogonal works"
    except Exception as e:
        return False, str(e)


def test_zeros_init() -> Tuple[bool, str]:
    """Test zeros initialization."""
    try:
        tensor = torch.randn(10, 10)  # Start with random
        result = zeros_(tensor)
        
        if not torch.allclose(result, torch.zeros_like(result)):
            return False, "Not all zeros"
        
        return True, "All values are 0"
    except Exception as e:
        return False, str(e)


def test_ones_init() -> Tuple[bool, str]:
    """Test ones initialization."""
    try:
        tensor = torch.randn(10, 10)
        result = ones_(tensor)
        
        if not torch.allclose(result, torch.ones_like(result)):
            return False, "Not all ones"
        
        return True, "All values are 1"
    except Exception as e:
        return False, str(e)


def test_constant_init() -> Tuple[bool, str]:
    """Test constant initialization."""
    try:
        tensor = torch.randn(10, 10)
        value = 3.14
        result = constant_(tensor, value)
        
        expected = torch.full_like(result, value)
        if not torch.allclose(result, expected):
            return False, f"Not all {value}"
        
        return True, f"All values are {value}"
    except Exception as e:
        return False, str(e)


def test_normal_init() -> Tuple[bool, str]:
    """Test normal initialization."""
    try:
        tensor = torch.empty(1000, 1000)
        mean, std = 2.0, 0.5
        result = normal_(tensor, mean, std)
        
        actual_mean = result.mean().item()
        actual_std = result.std().item()
        
        if abs(actual_mean - mean) > 0.1:
            return False, f"Mean {actual_mean:.4f} != {mean}"
        if abs(actual_std - std) > 0.05:
            return False, f"Std {actual_std:.4f} != {std}"
        
        return True, f"N({mean}, {std})"
    except Exception as e:
        return False, str(e)


def test_truncated_normal() -> Tuple[bool, str]:
    """Test truncated normal initialization."""
    try:
        tensor = torch.empty(1000, 1000)
        mean, std = 0.0, 1.0
        a, b = -2.0, 2.0
        
        result = truncated_normal_(tensor, mean, std, a, b)
        
        if result is None:
            return False, "Function returned None"
        
        # Values should be within bounds
        min_bound = mean + a * std
        max_bound = mean + b * std
        
        if result.min().item() < min_bound - 0.01:
            return False, f"Min {result.min().item():.4f} < {min_bound}"
        if result.max().item() > max_bound + 0.01:
            return False, f"Max {result.max().item():.4f} > {max_bound}"
        
        return True, f"Values in [{min_bound}, {max_bound}]"
    except Exception as e:
        return False, str(e)


def test_init_transformer_weights() -> Tuple[bool, str]:
    """Test Transformer-specific initialization."""
    try:
        d_model = 64
        
        # Test on linear layer
        linear = nn.Linear(d_model, d_model)
        init_transformer_weights(linear, d_model)
        
        # Bias should be zeros
        if linear.bias is not None:
            if not torch.allclose(linear.bias, torch.zeros_like(linear.bias)):
                return False, "Bias not zeros"
        
        return True, "Transformer init applied"
    except Exception as e:
        return False, str(e)


def test_smart_init() -> Tuple[bool, str]:
    """Test smart initialization on a model."""
    try:
        block = SimpleTransformerBlock(d_model=64, num_heads=4)
        apply_custom_init(block, smart_init)
        
        # Check that all 2D params have reasonable values
        for name, param in block.named_parameters():
            if param.dim() >= 2:
                std = param.std().item()
                if std > 1.0 or std < 0.001:
                    return False, f"{name} has unusual std: {std:.4f}"
        
        return True, "Smart init applied to all params"
    except Exception as e:
        return False, str(e)


def test_variance_preservation_xavier() -> Tuple[bool, str]:
    """Test that Xavier preserves variance through linear layers."""
    try:
        # Create a chain of linear layers
        layers = nn.ModuleList([nn.Linear(256, 256, bias=False) for _ in range(5)])
        
        # Initialize with Xavier
        for layer in layers:
            xavier_uniform_(layer.weight)
        
        # Pass input through
        x = torch.randn(32, 256)
        input_var = x.var().item()
        
        for layer in layers:
            x = layer(x)
        
        output_var = x.var().item()
        
        # Variance should be roughly preserved
        ratio = output_var / input_var
        if ratio < 0.3 or ratio > 3.0:
            return False, f"Variance ratio {ratio:.4f} (should be ~1)"
        
        return True, f"Variance ratio: {ratio:.4f}"
    except Exception as e:
        return False, str(e)


def test_variance_preservation_kaiming() -> Tuple[bool, str]:
    """Test that Kaiming preserves variance through ReLU layers."""
    try:
        # Create a chain of linear+ReLU layers
        layers = nn.ModuleList([nn.Linear(256, 256, bias=False) for _ in range(5)])
        
        # Initialize with Kaiming
        for layer in layers:
            kaiming_uniform_(layer.weight, nonlinearity='relu')
        
        # Pass input through with ReLU
        x = torch.randn(32, 256)
        input_var = x.var().item()
        
        for layer in layers:
            x = torch.relu(layer(x))
        
        output_var = x.var().item()
        
        # Variance should be roughly preserved
        ratio = output_var / input_var
        if ratio < 0.2 or ratio > 5.0:
            return False, f"Variance ratio {ratio:.4f} (should be ~1)"
        
        return True, f"Variance ratio: {ratio:.4f}"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("xavier_uniform_shape", test_xavier_uniform_shape),
        ("xavier_uniform_bounds", test_xavier_uniform_bounds),
        ("xavier_normal_statistics", test_xavier_normal_statistics),
        ("xavier_vs_pytorch", test_xavier_vs_pytorch),
        ("kaiming_uniform_shape", test_kaiming_uniform_shape),
        ("kaiming_normal_statistics", test_kaiming_normal_statistics),
        ("kaiming_fan_out", test_kaiming_fan_out),
        ("kaiming_leaky_relu", test_kaiming_leaky_relu),
        ("orthogonal_shape", test_orthogonal_shape),
        ("orthogonal_is_orthogonal", test_orthogonal_is_orthogonal),
        ("orthogonal_non_square", test_orthogonal_non_square),
        ("zeros_init", test_zeros_init),
        ("ones_init", test_ones_init),
        ("constant_init", test_constant_init),
        ("normal_init", test_normal_init),
        ("truncated_normal", test_truncated_normal),
        ("init_transformer_weights", test_init_transformer_weights),
        ("smart_init", test_smart_init),
        ("variance_preservation_xavier", test_variance_preservation_xavier),
        ("variance_preservation_kaiming", test_variance_preservation_kaiming),
    ]
    
    print(f"\n{'='*50}\nDay 21: Weight Initialization Strategies - Tests\n{'='*50}")
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    run_all_tests()
