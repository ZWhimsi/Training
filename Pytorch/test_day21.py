"""Test Suite for Day 21: Weight Initialization Strategies"""

import torch
import pytest
import torch.nn as nn
import math
try:
    from day21 import (xavier_uniform_, xavier_normal_,
                       kaiming_uniform_, kaiming_normal_,
                       orthogonal_, zeros_, ones_, constant_,
                       normal_, truncated_normal_,
                       init_transformer_weights, init_gpt_weights,
                       SimpleTransformerBlock, apply_custom_init, smart_init)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_xavier_uniform_shape():
    """Test Xavier uniform preserves shape and produces uniform distribution."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    tensor = torch.empty(128, 64)
    result = xavier_uniform_(tensor)
    
    assert result is not None, "Function returned None"
    assert result.shape == (128, 64), f"Shape changed: {result.shape}"
    
    assert result.std().item() >= 0.01, "Values not uniformly distributed"
    
    assert abs(result.mean().item()) <= 0.1, f"Mean not centered: {result.mean().item():.4f}"

def test_xavier_uniform_bounds():
    """Test Xavier uniform has correct bounds."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    fan_in, fan_out = 64, 128
    tensor = torch.empty(fan_out, fan_in)
    result = xavier_uniform_(tensor)
    
    assert result is not None, "Function returned None"
    
    expected_bound = math.sqrt(6.0 / (fan_in + fan_out))
    
    assert result.max().item() <= expected_bound * 1.01, f"Max {result.max().item():.4f} > bound {expected_bound:.4f}"
    assert result.min().item() >= -expected_bound * 1.01, f"Min {result.min().item():.4f} < -bound {-expected_bound:.4f}"

def test_xavier_normal_statistics():
    """Test Xavier normal has correct mean and std."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    fan_in, fan_out = 256, 256
    tensor = torch.empty(fan_out, fan_in)
    result = xavier_normal_(tensor)
    
    assert result is not None, "Function returned None"
    
    expected_std = math.sqrt(2.0 / (fan_in + fan_out))
    actual_std = result.std().item()
    
    assert abs(result.mean().item()) <= 0.1, f"Mean not ~0: {result.mean().item():.4f}"
    
    assert abs(actual_std - expected_std) <= 0.05, f"Std {actual_std:.4f} != expected {expected_std:.4f}"

def test_xavier_vs_pytorch():
    """Compare Xavier with PyTorch's implementation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    tensor1 = torch.empty(64, 64)
    xavier_uniform_(tensor1)
    
    torch.manual_seed(42)
    tensor2 = torch.empty(64, 64)
    nn.init.xavier_uniform_(tensor2)
    
    assert abs(tensor1.std().item() - tensor2.std().item()) <= 0.05, "Std differs significantly from PyTorch"

def test_kaiming_uniform_shape():
    """Test Kaiming uniform preserves shape and has correct bounds."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    fan_in = 64
    tensor = torch.empty(128, fan_in)
    result = kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')
    
    assert result is not None, "Function returned None"
    assert result.shape == (128, 64), f"Shape changed: {result.shape}"
    
    expected_std = math.sqrt(2.0) / math.sqrt(fan_in)
    expected_bound = math.sqrt(3) * expected_std
    
    assert result.max().item() <= expected_bound * 1.05, f"Max {result.max().item():.4f} > bound {expected_bound:.4f}"
    assert result.min().item() >= -expected_bound * 1.05, f"Min {result.min().item():.4f} < -bound"

def test_kaiming_normal_statistics():
    """Test Kaiming normal has correct statistics for ReLU."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    fan_in = 256
    tensor = torch.empty(128, fan_in)
    result = kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')
    
    assert result is not None, "Function returned None"
    
    expected_std = math.sqrt(2.0 / fan_in)
    actual_std = result.std().item()
    
    assert abs(result.mean().item()) <= 0.1, f"Mean not ~0: {result.mean().item():.4f}"
    
    assert abs(actual_std - expected_std) <= 0.05, f"Std {actual_std:.4f} != expected {expected_std:.4f}"

def test_kaiming_fan_out():
    """Test Kaiming with fan_out mode."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    fan_out = 256
    tensor = torch.empty(fan_out, 128)
    result = kaiming_normal_(tensor, mode='fan_out', nonlinearity='relu')
    
    assert result is not None, "Function returned None"
    
    expected_std = math.sqrt(2.0 / fan_out)
    actual_std = result.std().item()
    
    assert abs(actual_std - expected_std) <= 0.05, f"Std {actual_std:.4f} != expected {expected_std:.4f}"

def test_kaiming_leaky_relu():
    """Test Kaiming for leaky ReLU."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    fan_in = 256
    a = 0.2
    tensor = torch.empty(128, fan_in)
    result = kaiming_normal_(tensor, a=a, mode='fan_in', nonlinearity='leaky_relu')
    
    assert result is not None, "Function returned None"
    
    gain = math.sqrt(2.0 / (1 + a ** 2))
    expected_std = gain / math.sqrt(fan_in)
    actual_std = result.std().item()
    
    assert abs(actual_std - expected_std) <= 0.05, f"Std {actual_std:.4f} != expected {expected_std:.4f}"

def test_orthogonal_shape():
    """Test orthogonal initialization preserves shape and has unit norm rows."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    tensor = torch.empty(64, 64)
    result = orthogonal_(tensor)
    
    assert result is not None, "Function returned None"
    assert result.shape == (64, 64), f"Shape changed: {result.shape}"
    
    row_norms = result.norm(dim=1)
    assert torch.allclose(row_norms, torch.ones(64), atol=0.01), f"Row norms not ~1: mean={row_norms.mean().item():.4f}"

def test_orthogonal_is_orthogonal():
    """Test that orthogonal init produces orthogonal matrix."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    n = 64
    tensor = torch.empty(n, n)
    result = orthogonal_(tensor)
    
    assert result is not None, "Function returned None"
    
    identity = torch.eye(n)
    product = torch.mm(result, result.t())
    error = (product - identity).abs().mean().item()
    
    assert error <= 0.01, f"Not orthogonal: error={error:.6f}"

def test_orthogonal_non_square():
    """Test orthogonal init with non-square matrix."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    tensor = torch.empty(128, 64)
    result = orthogonal_(tensor)
    
    assert result is not None, "Function returned None"
    assert result.shape == (128, 64), f"Shape changed: {result.shape}"
    
    product = torch.mm(result.t(), result)
    identity = torch.eye(64)
    error = (product - identity).abs().mean().item()
    
    assert error <= 0.01, f"Columns not orthonormal: error={error:.6f}"

def test_zeros_init():
    """Test zeros initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    tensor = torch.randn(10, 10)
    result = zeros_(tensor)
    
    assert torch.allclose(result, torch.zeros_like(result)), "Not all zeros"

def test_ones_init():
    """Test ones initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    tensor = torch.randn(10, 10)
    result = ones_(tensor)
    
    assert torch.allclose(result, torch.ones_like(result)), "Not all ones"

def test_constant_init():
    """Test constant initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    tensor = torch.randn(10, 10)
    value = 3.14
    result = constant_(tensor, value)
    
    expected = torch.full_like(result, value)
    assert torch.allclose(result, expected), f"Not all {value}"

def test_normal_init():
    """Test normal initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    tensor = torch.empty(1000, 1000)
    mean, std = 2.0, 0.5
    result = normal_(tensor, mean, std)
    
    actual_mean = result.mean().item()
    actual_std = result.std().item()
    
    assert abs(actual_mean - mean) <= 0.1, f"Mean {actual_mean:.4f} != {mean}"
    assert abs(actual_std - std) <= 0.05, f"Std {actual_std:.4f} != {std}"

def test_truncated_normal():
    """Test truncated normal initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    tensor = torch.empty(1000, 1000)
    mean, std = 0.0, 1.0
    a, b = -2.0, 2.0
    
    result = truncated_normal_(tensor, mean, std, a, b)
    
    assert result is not None, "Function returned None"
    
    min_bound = mean + a * std
    max_bound = mean + b * std
    
    assert result.min().item() >= min_bound - 0.01, f"Min {result.min().item():.4f} < {min_bound}"
    assert result.max().item() <= max_bound + 0.01, f"Max {result.max().item():.4f} > {max_bound}"

def test_init_transformer_weights():
    """Test Transformer-specific initialization with correct statistics."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model = 64
    
    linear = nn.Linear(d_model, d_model)
    init_transformer_weights(linear, d_model)
    
    if linear.bias is not None:
        assert torch.allclose(linear.bias, torch.zeros_like(linear.bias)), "Bias not zeros"
    
    expected_std = math.sqrt(2.0 / (d_model + d_model))
    actual_std = linear.weight.std().item()
    assert abs(actual_std - expected_std) <= 0.1, f"Weight std {actual_std:.4f} not ~{expected_std:.4f}"
    
    emb = nn.Embedding(1000, d_model)
    init_transformer_weights(emb, d_model)
    
    expected_emb_std = math.sqrt(1.0 / d_model)
    actual_emb_std = emb.weight.std().item()
    assert abs(actual_emb_std - expected_emb_std) <= 0.1, f"Embedding std {actual_emb_std:.4f} not ~{expected_emb_std:.4f}"

def test_smart_init():
    """Test smart initialization on a model."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    block = SimpleTransformerBlock(d_model=64, num_heads=4)
    apply_custom_init(block, smart_init)
    
    for name, param in block.named_parameters():
        if param.dim() >= 2:
            std = param.std().item()
            assert std <= 1.0 or std < 0.001, f"{name} has unusual std: {std:.4f}"

def test_variance_preservation_xavier():
    """Test that Xavier preserves variance through linear layers."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    layers = nn.ModuleList([nn.Linear(256, 256, bias=False) for _ in range(5)])
    
    for layer in layers:
        xavier_uniform_(layer.weight)
    
    x = torch.randn(32, 256)
    input_var = x.var().item()
    
    for layer in layers:
        x = layer(x)
    
    output_var = x.var().item()
    
    ratio = output_var / input_var
    assert ratio < 0.3 or ratio <= 3.0, f"Variance ratio {ratio:.4f} (should be ~1)"

def test_variance_preservation_kaiming():
    """Test that Kaiming preserves variance through ReLU layers."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    layers = nn.ModuleList([nn.Linear(256, 256, bias=False) for _ in range(5)])
    
    for layer in layers:
        kaiming_uniform_(layer.weight, nonlinearity='relu')
    
    x = torch.randn(32, 256)
    input_var = x.var().item()
    
    for layer in layers:
        x = torch.relu(layer(x))
    
    output_var = x.var().item()
    
    ratio = output_var / input_var
    assert ratio < 0.2 or ratio <= 5.0, f"Variance ratio {ratio:.4f} (should be ~1)"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
