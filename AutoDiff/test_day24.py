"""Test Suite for Day 24: Sequential Container"""

import numpy as np
import sys
from typing import Tuple

try:
    from day24 import (
        Tensor,
        Module,
        Linear,
        ReLU,
        Sigmoid,
        Tanh,
        Sequential,
        ModuleList,
        Residual,
        Lambda,
        Flatten,
        Dropout,
        test_sequential_forward,
        test_sequential_parameters,
        test_sequential_backward,
        test_module_list,
        test_residual,
        test_flatten,
        test_dropout,
        test_deep_network
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_sequential_creation() -> Tuple[bool, str]:
    """Test Sequential container creation."""
    try:
        model = Sequential(
            Linear(10, 5),
            ReLU(),
            Linear(5, 2)
        )
        
        if model._modules is None:
            return False, "_modules is None"
        if len(model) != 3:
            return False, f"len = {len(model)}, expected 3"
        
        return True, "Sequential created with 3 modules"
    except Exception as e:
        return False, str(e)


def test_sequential_forward() -> Tuple[bool, str]:
    """Test Sequential forward pass."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        x = Tensor(np.random.randn(3, 4))
        y = model(x)
        
        if y is None:
            return False, "forward returned None"
        if y.shape != (3, 2):
            return False, f"shape = {y.shape}, expected (3, 2)"
        
        return True, "Sequential forward: (3,4) -> (3,2)"
    except Exception as e:
        return False, str(e)


def test_sequential_parameters() -> Tuple[bool, str]:
    """Test Sequential parameter collection."""
    try:
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
        
        params = model.parameters()
        if len(params) != 4:
            return False, f"params = {len(params)}, expected 4"
        
        return True, "Sequential collects 4 parameters"
    except Exception as e:
        return False, str(e)


def test_sequential_backward() -> Tuple[bool, str]:
    """Test Sequential backward pass."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        x = Tensor(np.random.randn(2, 4))
        y = model(x)
        
        if y is None:
            return False, "forward returned None"
        
        loss = y.sum()
        loss.backward()
        
        params = model.parameters()
        all_have_grad = all(np.any(p.grad != 0) for p in params)
        if not all_have_grad:
            return False, "Some params have zero gradient"
        
        return True, "All parameters have gradients"
    except Exception as e:
        return False, str(e)


def test_sequential_indexing() -> Tuple[bool, str]:
    """Test Sequential indexing."""
    try:
        linear1 = Linear(10, 5)
        relu = ReLU()
        linear2 = Linear(5, 2)
        
        model = Sequential(linear1, relu, linear2)
        
        if model[0] is not linear1:
            return False, "model[0] != linear1"
        if model[1] is not relu:
            return False, "model[1] != relu"
        if model[2] is not linear2:
            return False, "model[2] != linear2"
        
        return True, "Sequential indexing works"
    except Exception as e:
        return False, str(e)


def test_sequential_iteration() -> Tuple[bool, str]:
    """Test Sequential iteration."""
    try:
        modules = [Linear(10, 10), ReLU(), Linear(10, 5)]
        model = Sequential(*modules)
        
        iterated = list(model)
        if len(iterated) != 3:
            return False, f"iterated = {len(iterated)}"
        
        return True, "Sequential is iterable"
    except Exception as e:
        return False, str(e)


def test_sequential_zero_grad() -> Tuple[bool, str]:
    """Test Sequential zero_grad."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        x = Tensor(np.random.randn(2, 4))
        y = model(x)
        if y is not None:
            y.sum().backward()
        
        model.zero_grad()
        
        all_zero = all(np.all(p.grad == 0) for p in model.parameters())
        if not all_zero:
            return False, "Gradients not zeroed"
        
        return True, "zero_grad works"
    except Exception as e:
        return False, str(e)


def test_modulelist_creation() -> Tuple[bool, str]:
    """Test ModuleList creation."""
    try:
        layers = ModuleList([Linear(10, 10) for _ in range(5)])
        
        if layers._modules is None:
            return False, "_modules is None"
        if len(layers) != 5:
            return False, f"len = {len(layers)}, expected 5"
        
        return True, "ModuleList created with 5 modules"
    except Exception as e:
        return False, str(e)


def test_modulelist_append() -> Tuple[bool, str]:
    """Test ModuleList append."""
    try:
        layers = ModuleList()
        layers.append(Linear(10, 5))
        layers.append(ReLU())
        
        if len(layers) != 2:
            return False, f"len = {len(layers)}, expected 2"
        
        return True, "ModuleList append works"
    except Exception as e:
        return False, str(e)


def test_modulelist_parameters() -> Tuple[bool, str]:
    """Test ModuleList parameter collection."""
    try:
        layers = ModuleList([Linear(10, 10) for _ in range(3)])
        params = layers.parameters()
        
        if len(params) != 6:
            return False, f"params = {len(params)}, expected 6"
        
        return True, "ModuleList collects 6 parameters"
    except Exception as e:
        return False, str(e)


def test_residual_forward() -> Tuple[bool, str]:
    """Test Residual forward pass."""
    try:
        np.random.seed(42)
        block = Residual(Linear(4, 4, bias=False))
        
        if block.fn is None:
            return False, "fn is None"
        
        x = Tensor([[1, 0, 0, 0]])
        y = block(x)
        
        if y is None:
            return False, "forward returned None"
        if y.shape != (1, 4):
            return False, f"shape = {y.shape}"
        
        if y.data[0, 0] == 0:
            return False, "Skip connection not applied"
        
        return True, "Residual: x + F(x)"
    except Exception as e:
        return False, str(e)


def test_residual_backward() -> Tuple[bool, str]:
    """Test Residual backward pass."""
    try:
        np.random.seed(42)
        block = Residual(Linear(4, 4))
        
        if block.fn is None:
            return False, "fn is None"
        
        x = Tensor(np.random.randn(2, 4))
        y = block(x)
        
        if y is None:
            return False, "forward returned None"
        
        y.sum().backward()
        
        if np.all(x.grad == 0):
            return False, "Input gradient is zero"
        
        return True, "Residual backward works"
    except Exception as e:
        return False, str(e)


def test_residual_parameters() -> Tuple[bool, str]:
    """Test Residual parameter collection."""
    try:
        block = Residual(
            Sequential(
                Linear(10, 10),
                ReLU(),
                Linear(10, 10)
            )
        )
        
        params = block.parameters()
        if len(params) != 4:
            return False, f"params = {len(params)}, expected 4"
        
        return True, "Residual collects inner params"
    except Exception as e:
        return False, str(e)


def test_lambda_forward() -> Tuple[bool, str]:
    """Test Lambda module forward."""
    try:
        double = Lambda(lambda x: x * 2)
        
        if double.fn is None:
            return False, "fn is None"
        
        x = Tensor([1, 2, 3])
        y = double(x)
        
        if y is None:
            return False, "forward returned None"
        if not np.allclose(y.data, [2, 4, 6]):
            return False, f"values = {y.data}"
        
        return True, "Lambda doubles values"
    except Exception as e:
        return False, str(e)


def test_flatten_forward() -> Tuple[bool, str]:
    """Test Flatten forward pass."""
    try:
        flatten = Flatten()
        x = Tensor(np.random.randn(4, 3, 8, 8))
        y = flatten(x)
        
        if y is None:
            return False, "forward returned None"
        if y.shape != (4, 192):
            return False, f"shape = {y.shape}, expected (4, 192)"
        
        return True, "Flatten: (4,3,8,8) -> (4,192)"
    except Exception as e:
        return False, str(e)


def test_flatten_backward() -> Tuple[bool, str]:
    """Test Flatten backward pass."""
    try:
        flatten = Flatten()
        x = Tensor(np.random.randn(2, 3, 4))
        y = flatten(x)
        
        if y is None:
            return False, "forward returned None"
        
        y.backward()
        
        if x.grad.shape != (2, 3, 4):
            return False, f"grad shape = {x.grad.shape}"
        
        return True, "Flatten backward preserves shape"
    except Exception as e:
        return False, str(e)


def test_dropout_training() -> Tuple[bool, str]:
    """Test Dropout in training mode."""
    try:
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        dropout.train()
        
        x = Tensor(np.ones((100, 100)))
        y = dropout(x)
        
        if y is None:
            return False, "forward returned None"
        
        zero_ratio = np.mean(y.data == 0)
        if not (0.3 < zero_ratio < 0.7):
            return False, f"zero_ratio = {zero_ratio}"
        
        return True, "Dropout drops ~50% in training"
    except Exception as e:
        return False, str(e)


def test_dropout_eval() -> Tuple[bool, str]:
    """Test Dropout in eval mode."""
    try:
        dropout = Dropout(p=0.5)
        dropout.eval()
        
        x = Tensor(np.ones((10, 10)))
        y = dropout(x)
        
        if y is None:
            return False, "forward returned None"
        
        if not np.allclose(y.data, x.data):
            return False, "Dropout modified values in eval"
        
        return True, "Dropout is identity in eval"
    except Exception as e:
        return False, str(e)


def test_dropout_scaling() -> Tuple[bool, str]:
    """Test Dropout scaling during training."""
    try:
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        dropout.train()
        
        x = Tensor(np.ones((1000, 1000)))
        y = dropout(x)
        
        if y is None:
            return False, "forward returned None"
        
        non_zero = y.data[y.data != 0]
        if not np.allclose(non_zero, 2.0, atol=0.01):
            return False, f"Scaling wrong: {non_zero.mean()}"
        
        return True, "Dropout scales by 1/(1-p)"
    except Exception as e:
        return False, str(e)


def test_deep_sequential() -> Tuple[bool, str]:
    """Test deep Sequential network."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(784, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 10)
        )
        
        x = Tensor(np.random.randn(16, 784))
        y = model(x)
        
        if y is None:
            return False, "forward returned None"
        if y.shape != (16, 10):
            return False, f"shape = {y.shape}"
        
        y.sum().backward()
        
        params = model.parameters()
        all_grad = all(np.any(p.grad != 0) for p in params)
        if not all_grad:
            return False, "Some params missing gradient"
        
        return True, "5-layer network works"
    except Exception as e:
        return False, str(e)


def test_nested_sequential() -> Tuple[bool, str]:
    """Test nested Sequential containers."""
    try:
        np.random.seed(42)
        model = Sequential(
            Sequential(
                Linear(10, 20),
                ReLU()
            ),
            Sequential(
                Linear(20, 10),
                ReLU()
            ),
            Linear(10, 5)
        )
        
        x = Tensor(np.random.randn(4, 10))
        y = model(x)
        
        if y is None:
            return False, "forward returned None"
        if y.shape != (4, 5):
            return False, f"shape = {y.shape}"
        
        return True, "Nested Sequential works"
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test Sequential against PyTorch."""
    try:
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        our_model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        torch_model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        torch_model[0].weight.data = torch.tensor(our_model[0].weight.data.copy())
        torch_model[0].bias.data = torch.tensor(our_model[0].bias.data.copy())
        torch_model[2].weight.data = torch.tensor(our_model[2].weight.data.copy())
        torch_model[2].bias.data = torch.tensor(our_model[2].bias.data.copy())
        
        x_np = np.random.randn(3, 4)
        
        our_x = Tensor(x_np.copy())
        our_y = our_model(our_x)
        if our_y is None:
            return False, "Our forward returned None"
        our_y.sum().backward()
        
        torch_x = torch.tensor(x_np, requires_grad=True)
        torch_y = torch_model(torch_x)
        torch_y.sum().backward()
        
        if not np.allclose(our_y.data, torch_y.detach().numpy(), rtol=1e-5):
            return False, "Forward mismatch"
        
        if not np.allclose(our_model[0].weight.grad, 
                          torch_model[0].weight.grad.numpy(), rtol=1e-5):
            return False, "Weight gradient mismatch"
        
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("sequential_creation", test_sequential_creation),
        ("sequential_forward", test_sequential_forward),
        ("sequential_parameters", test_sequential_parameters),
        ("sequential_backward", test_sequential_backward),
        ("sequential_indexing", test_sequential_indexing),
        ("sequential_iteration", test_sequential_iteration),
        ("sequential_zero_grad", test_sequential_zero_grad),
        ("modulelist_creation", test_modulelist_creation),
        ("modulelist_append", test_modulelist_append),
        ("modulelist_parameters", test_modulelist_parameters),
        ("residual_forward", test_residual_forward),
        ("residual_backward", test_residual_backward),
        ("residual_parameters", test_residual_parameters),
        ("lambda_forward", test_lambda_forward),
        ("flatten_forward", test_flatten_forward),
        ("flatten_backward", test_flatten_backward),
        ("dropout_training", test_dropout_training),
        ("dropout_eval", test_dropout_eval),
        ("dropout_scaling", test_dropout_scaling),
        ("deep_sequential", test_deep_sequential),
        ("nested_sequential", test_nested_sequential),
        ("against_pytorch", test_against_pytorch),
    ]
    
    print(f"\n{'='*60}")
    print("Day 24: Sequential Container - Tests")
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
