"""Test Suite for Day 23: Activation Modules"""

import numpy as np
import sys
from typing import Tuple

try:
    from day23 import (
        Tensor,
        Module,
        ReLU,
        Sigmoid,
        Tanh,
        LeakyReLU,
        GELU,
        Softplus,
        ELU,
        test_relu,
        test_sigmoid,
        test_tanh,
        test_leaky_relu,
        test_gelu,
        test_softplus,
        test_elu
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_relu_forward() -> Tuple[bool, str]:
    """Test ReLU forward pass."""
    try:
        relu = ReLU()
        x = Tensor([-3, -1, 0, 1, 3])
        y = relu(x)
        
        if y is None or y.data is None:
            return False, "Returned None"
        
        expected = np.array([0, 0, 0, 1, 3])
        if not np.allclose(y.data, expected):
            return False, f"values = {y.data}"
        
        return True, "ReLU([-3,-1,0,1,3]) = [0,0,0,1,3]"
    except Exception as e:
        return False, str(e)


def test_relu_backward() -> Tuple[bool, str]:
    """Test ReLU backward pass."""
    try:
        relu = ReLU()
        x = Tensor([-2.0, -1.0, 0.5, 1.0, 2.0])
        y = relu(x)
        
        if y is None:
            return False, "Forward returned None"
        
        y.backward()
        
        expected = np.array([0, 0, 1, 1, 1])
        if not np.allclose(x.grad, expected):
            return False, f"grad = {x.grad}"
        
        return True, "ReLU gradient correct"
    except Exception as e:
        return False, str(e)


def test_relu_zero() -> Tuple[bool, str]:
    """Test ReLU at zero (edge case)."""
    try:
        relu = ReLU()
        x = Tensor([0.0])
        y = relu(x)
        
        if y is None:
            return False, "Forward returned None"
        
        if not np.allclose(y.data, [0]):
            return False, f"ReLU(0) = {y.data}"
        
        return True, "ReLU(0) = 0"
    except Exception as e:
        return False, str(e)


def test_sigmoid_forward() -> Tuple[bool, str]:
    """Test Sigmoid forward pass."""
    try:
        sigmoid = Sigmoid()
        x = Tensor([0.0])
        y = sigmoid(x)
        
        if y is None or y.data is None:
            return False, "Returned None"
        
        if not np.allclose(y.data, [0.5]):
            return False, f"sigmoid(0) = {y.data}, expected 0.5"
        
        return True, "sigmoid(0) = 0.5"
    except Exception as e:
        return False, str(e)


def test_sigmoid_backward() -> Tuple[bool, str]:
    """Test Sigmoid backward pass."""
    try:
        sigmoid = Sigmoid()
        x = Tensor([0.0])
        y = sigmoid(x)
        
        if y is None:
            return False, "Forward returned None"
        
        y.backward()
        
        if not np.allclose(x.grad, [0.25]):
            return False, f"grad = {x.grad}, expected 0.25"
        
        return True, "d(sigmoid)/dx at 0 = 0.25"
    except Exception as e:
        return False, str(e)


def test_sigmoid_range() -> Tuple[bool, str]:
    """Test Sigmoid output range."""
    try:
        sigmoid = Sigmoid()
        x = Tensor(np.linspace(-10, 10, 100))
        y = sigmoid(x)
        
        if y is None:
            return False, "Forward returned None"
        
        if not (np.all(y.data > 0) and np.all(y.data < 1)):
            return False, "Output not in (0, 1)"
        
        return True, "Sigmoid output in (0, 1)"
    except Exception as e:
        return False, str(e)


def test_sigmoid_stability() -> Tuple[bool, str]:
    """Test Sigmoid numerical stability."""
    try:
        sigmoid = Sigmoid()
        x = Tensor([-1000, 0, 1000])
        y = sigmoid(x)
        
        if y is None:
            return False, "Forward returned None"
        
        if np.any(np.isnan(y.data)) or np.any(np.isinf(y.data)):
            return False, "NaN or Inf in output"
        
        return True, "No overflow for large |x|"
    except Exception as e:
        return False, str(e)


def test_tanh_forward() -> Tuple[bool, str]:
    """Test Tanh forward pass."""
    try:
        tanh = Tanh()
        x = Tensor([0.0])
        y = tanh(x)
        
        if y is None or y.data is None:
            return False, "Returned None"
        
        if not np.allclose(y.data, [0]):
            return False, f"tanh(0) = {y.data}"
        
        return True, "tanh(0) = 0"
    except Exception as e:
        return False, str(e)


def test_tanh_backward() -> Tuple[bool, str]:
    """Test Tanh backward pass."""
    try:
        tanh = Tanh()
        x = Tensor([0.0])
        y = tanh(x)
        
        if y is None:
            return False, "Forward returned None"
        
        y.backward()
        
        if not np.allclose(x.grad, [1.0]):
            return False, f"grad = {x.grad}, expected 1"
        
        return True, "d(tanh)/dx at 0 = 1"
    except Exception as e:
        return False, str(e)


def test_tanh_range() -> Tuple[bool, str]:
    """Test Tanh output range."""
    try:
        tanh = Tanh()
        x = Tensor(np.linspace(-10, 10, 100))
        y = tanh(x)
        
        if y is None:
            return False, "Forward returned None"
        
        if not (np.all(y.data > -1) and np.all(y.data < 1)):
            return False, "Output not in (-1, 1)"
        
        return True, "Tanh output in (-1, 1)"
    except Exception as e:
        return False, str(e)


def test_leaky_relu_forward() -> Tuple[bool, str]:
    """Test LeakyReLU forward pass."""
    try:
        lrelu = LeakyReLU(alpha=0.1)
        x = Tensor([-2, -1, 0, 1, 2])
        y = lrelu(x)
        
        if y is None or y.data is None:
            return False, "Returned None"
        
        expected = np.array([-0.2, -0.1, 0, 1, 2])
        if not np.allclose(y.data, expected):
            return False, f"values = {y.data}"
        
        return True, "LeakyReLU with alpha=0.1 correct"
    except Exception as e:
        return False, str(e)


def test_leaky_relu_backward() -> Tuple[bool, str]:
    """Test LeakyReLU backward pass."""
    try:
        lrelu = LeakyReLU(alpha=0.2)
        x = Tensor([-1.0, 1.0])
        y = lrelu(x)
        
        if y is None:
            return False, "Forward returned None"
        
        y.backward()
        
        if not np.allclose(x.grad, [0.2, 1.0]):
            return False, f"grad = {x.grad}"
        
        return True, "LeakyReLU gradient correct"
    except Exception as e:
        return False, str(e)


def test_leaky_relu_no_dying() -> Tuple[bool, str]:
    """Test that LeakyReLU doesn't have dying problem."""
    try:
        lrelu = LeakyReLU(alpha=0.01)
        x = Tensor([-100.0])
        y = lrelu(x)
        
        if y is None:
            return False, "Forward returned None"
        
        y.backward()
        
        if x.grad[0] == 0:
            return False, "Gradient is zero for negative input"
        
        return True, "Non-zero gradient for negative input"
    except Exception as e:
        return False, str(e)


def test_gelu_forward() -> Tuple[bool, str]:
    """Test GELU forward pass."""
    try:
        gelu = GELU(approximate=True)
        x = Tensor([0.0])
        y = gelu(x)
        
        if y is None or y.data is None:
            return False, "Returned None"
        
        if not np.allclose(y.data, [0], atol=1e-5):
            return False, f"GELU(0) = {y.data}"
        
        return True, "GELU(0) ≈ 0"
    except Exception as e:
        return False, str(e)


def test_gelu_backward() -> Tuple[bool, str]:
    """Test GELU backward pass."""
    try:
        gelu = GELU(approximate=True)
        x = Tensor([0.0, 1.0, 2.0])
        y = gelu(x)
        
        if y is None:
            return False, "Forward returned None"
        
        y.backward()
        
        if np.any(np.isnan(x.grad)):
            return False, "NaN in gradient"
        
        return True, "GELU backward works"
    except Exception as e:
        return False, str(e)


def test_gelu_shape() -> Tuple[bool, str]:
    """Test GELU preserves shape."""
    try:
        gelu = GELU()
        x = Tensor(np.random.randn(4, 8))
        y = gelu(x)
        
        if y is None:
            return False, "Forward returned None"
        if y.shape != (4, 8):
            return False, f"shape = {y.shape}"
        
        return True, "GELU preserves shape"
    except Exception as e:
        return False, str(e)


def test_softplus_forward() -> Tuple[bool, str]:
    """Test Softplus forward pass."""
    try:
        sp = Softplus()
        x = Tensor([0.0])
        y = sp(x)
        
        if y is None or y.data is None:
            return False, "Returned None"
        
        if not np.allclose(y.data, [np.log(2)]):
            return False, f"softplus(0) = {y.data}, expected {np.log(2)}"
        
        return True, "softplus(0) = ln(2)"
    except Exception as e:
        return False, str(e)


def test_softplus_positive() -> Tuple[bool, str]:
    """Test Softplus is always positive."""
    try:
        sp = Softplus()
        x = Tensor([-10, -5, 0, 5, 10])
        y = sp(x)
        
        if y is None:
            return False, "Forward returned None"
        
        if not np.all(y.data > 0):
            return False, "Output not always positive"
        
        return True, "Softplus always positive"
    except Exception as e:
        return False, str(e)


def test_softplus_backward() -> Tuple[bool, str]:
    """Test Softplus backward pass."""
    try:
        sp = Softplus()
        x = Tensor([0.0])
        y = sp(x)
        
        if y is None:
            return False, "Forward returned None"
        
        y.backward()
        
        if not np.allclose(x.grad, [0.5]):
            return False, f"grad = {x.grad}, expected 0.5"
        
        return True, "d(softplus)/dx at 0 = 0.5"
    except Exception as e:
        return False, str(e)


def test_elu_forward() -> Tuple[bool, str]:
    """Test ELU forward pass."""
    try:
        elu = ELU(alpha=1.0)
        x = Tensor([-1, 0, 1])
        y = elu(x)
        
        if y is None or y.data is None:
            return False, "Returned None"
        
        if not np.allclose(y.data[1:], [0, 1]):
            return False, f"ELU(0)={y.data[1]}, ELU(1)={y.data[2]}"
        
        if not (y.data[0] > -1 and y.data[0] < 0):
            return False, f"ELU(-1) = {y.data[0]}, expected in (-1, 0)"
        
        return True, "ELU forward correct"
    except Exception as e:
        return False, str(e)


def test_elu_backward() -> Tuple[bool, str]:
    """Test ELU backward pass."""
    try:
        elu = ELU(alpha=1.0)
        x = Tensor([1.0])
        y = elu(x)
        
        if y is None:
            return False, "Forward returned None"
        
        y.backward()
        
        if not np.allclose(x.grad, [1.0]):
            return False, f"grad = {x.grad}, expected 1"
        
        return True, "ELU gradient = 1 for positive"
    except Exception as e:
        return False, str(e)


def test_against_pytorch_relu() -> Tuple[bool, str]:
    """Test ReLU against PyTorch."""
    try:
        import torch
        import torch.nn.functional as F
        
        np.random.seed(42)
        x_np = np.random.randn(10) * 2
        
        our_x = Tensor(x_np.copy())
        our_y = ReLU()(our_x)
        if our_y is None:
            return False, "Our ReLU returned None"
        our_y.backward()
        
        torch_x = torch.tensor(x_np, requires_grad=True)
        torch_y = F.relu(torch_x)
        torch_y.sum().backward()
        
        if not np.allclose(our_y.data, torch_y.detach().numpy()):
            return False, "Forward mismatch"
        if not np.allclose(our_x.grad, torch_x.grad.numpy()):
            return False, "Gradient mismatch"
        
        return True, "ReLU matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def test_against_pytorch_sigmoid() -> Tuple[bool, str]:
    """Test Sigmoid against PyTorch."""
    try:
        import torch
        
        np.random.seed(42)
        x_np = np.random.randn(10)
        
        our_x = Tensor(x_np.copy())
        our_y = Sigmoid()(our_x)
        if our_y is None:
            return False, "Our Sigmoid returned None"
        our_y.backward()
        
        torch_x = torch.tensor(x_np, requires_grad=True)
        torch_y = torch.sigmoid(torch_x)
        torch_y.sum().backward()
        
        if not np.allclose(our_y.data, torch_y.detach().numpy()):
            return False, "Forward mismatch"
        if not np.allclose(our_x.grad, torch_x.grad.numpy()):
            return False, "Gradient mismatch"
        
        return True, "Sigmoid matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def test_against_pytorch_tanh() -> Tuple[bool, str]:
    """Test Tanh against PyTorch."""
    try:
        import torch
        
        np.random.seed(42)
        x_np = np.random.randn(10)
        
        our_x = Tensor(x_np.copy())
        our_y = Tanh()(our_x)
        if our_y is None:
            return False, "Our Tanh returned None"
        our_y.backward()
        
        torch_x = torch.tensor(x_np, requires_grad=True)
        torch_y = torch.tanh(torch_x)
        torch_y.sum().backward()
        
        if not np.allclose(our_y.data, torch_y.detach().numpy()):
            return False, "Forward mismatch"
        if not np.allclose(our_x.grad, torch_x.grad.numpy()):
            return False, "Gradient mismatch"
        
        return True, "Tanh matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("relu_forward", test_relu_forward),
        ("relu_backward", test_relu_backward),
        ("relu_zero", test_relu_zero),
        ("sigmoid_forward", test_sigmoid_forward),
        ("sigmoid_backward", test_sigmoid_backward),
        ("sigmoid_range", test_sigmoid_range),
        ("sigmoid_stability", test_sigmoid_stability),
        ("tanh_forward", test_tanh_forward),
        ("tanh_backward", test_tanh_backward),
        ("tanh_range", test_tanh_range),
        ("leaky_relu_forward", test_leaky_relu_forward),
        ("leaky_relu_backward", test_leaky_relu_backward),
        ("leaky_relu_no_dying", test_leaky_relu_no_dying),
        ("gelu_forward", test_gelu_forward),
        ("gelu_backward", test_gelu_backward),
        ("gelu_shape", test_gelu_shape),
        ("softplus_forward", test_softplus_forward),
        ("softplus_positive", test_softplus_positive),
        ("softplus_backward", test_softplus_backward),
        ("elu_forward", test_elu_forward),
        ("elu_backward", test_elu_backward),
        ("against_pytorch_relu", test_against_pytorch_relu),
        ("against_pytorch_sigmoid", test_against_pytorch_sigmoid),
        ("against_pytorch_tanh", test_against_pytorch_tanh),
    ]
    
    print(f"\n{'='*60}")
    print("Day 23: Activation Modules - Tests")
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
