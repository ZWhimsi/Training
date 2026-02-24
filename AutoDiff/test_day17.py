"""Test Suite for Day 17: Parameter and Module"""

import numpy as np
import pytest

from day17 import Tensor, Parameter, Module


def test_parameter_creation():
    """Test Parameter creation."""
    p = Parameter([[1.0, 2.0], [3.0, 4.0]])
    
    assert p is not None, "Parameter returned None"
    assert p.data is not None, "Parameter data is None"
    assert np.allclose(p.data, [[1, 2], [3, 4]]), f"Data: {p.data}"


def test_parameter_requires_grad():
    """Test that Parameter requires grad by default."""
    p = Parameter([1.0, 2.0])
    
    assert hasattr(p, 'requires_grad'), "No requires_grad attribute"
    assert p.requires_grad is True, "requires_grad should be True"


def test_parameter_is_tensor():
    """Test that Parameter is a Tensor."""
    p = Parameter([1.0, 2.0])
    
    assert isinstance(p, Tensor), "Parameter should be a Tensor subclass"


def test_module_empty():
    """Test empty Module."""
    m = Module()
    params = list(m.parameters())
    
    assert params is not None, "parameters() returned None"
    assert len(params) == 0, f"Empty module should have 0 params, got {len(params)}"


def test_module_with_parameter():
    """Test Module with a Parameter."""
    class SimpleModule(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter([1.0, 2.0, 3.0])
    
    m = SimpleModule()
    params = list(m.parameters())
    
    assert len(params) == 1, f"Should have 1 param, got {len(params)}"
    assert np.allclose(params[0].data, [1, 2, 3]), f"Wrong param: {params[0].data}"


def test_module_multiple_parameters():
    """Test Module with multiple Parameters."""
    class MultiParamModule(Module):
        def __init__(self):
            super().__init__()
            self.w1 = Parameter([1.0, 2.0])
            self.w2 = Parameter([3.0, 4.0])
            self.b = Parameter([0.5])
    
    m = MultiParamModule()
    params = list(m.parameters())
    
    assert len(params) == 3, f"Should have 3 params, got {len(params)}"


def test_module_nested():
    """Test nested Modules."""
    class Inner(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter([1.0, 2.0])
    
    class Outer(Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()
            self.bias = Parameter([0.0])
    
    m = Outer()
    params = list(m.parameters())
    
    assert len(params) == 2, f"Should have 2 params (1 inner + 1 outer), got {len(params)}"


def test_module_zero_grad():
    """Test zero_grad method."""
    class SimpleModule(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter([1.0, 2.0])
    
    m = SimpleModule()
    m.weight.grad = np.array([5.0, 10.0])
    
    m.zero_grad()
    
    assert np.allclose(m.weight.grad, [0, 0]), f"Grad not zeroed: {m.weight.grad}"


def test_module_train_eval():
    """Test train/eval mode."""
    m = Module()
    
    m.train()
    assert m.training is True, "Should be in training mode"
    
    m.eval()
    assert m.training is False, "Should be in eval mode"


def test_module_forward():
    """Test that Module forward raises NotImplementedError or works if overridden."""
    class ForwardModule(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter([2.0])
        
        def forward(self, x):
            return x * self.weight
    
    m = ForwardModule()
    x = Tensor([3.0])
    y = m(x)
    
    assert y is not None, "forward returned None"
    assert np.allclose(y.data, [6.0]), f"y = {y.data}, expected [6]"


def test_module_call():
    """Test that Module is callable via __call__."""
    class CallableModule(Module):
        def __init__(self):
            super().__init__()
            self.scale = Parameter([10.0])
        
        def forward(self, x):
            return x + self.scale
    
    m = CallableModule()
    assert callable(m), "Module should be callable"
    
    result = m(Tensor([5.0]))
    assert np.allclose(result.data, [15.0]), f"result = {result.data}"


def test_parameter_backward():
    """Test Parameter backward pass."""
    p = Parameter([2.0, 3.0])
    x = Tensor([1.0, 1.0])
    y = p * x  # [2, 3]
    z = y.sum()  # 5
    z.backward()
    
    assert p.grad is not None, "Parameter grad is None"
    # dz/dp = x = [1, 1]
    assert np.allclose(p.grad, [1, 1]), f"p.grad = {p.grad}"


def test_named_parameters():
    """Test named_parameters method."""
    class NamedModule(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter([1.0, 2.0])
            self.bias = Parameter([0.5])
    
    m = NamedModule()
    
    try:
        named = list(m.named_parameters())
        names = [n for n, _ in named]
        assert 'weight' in names and 'bias' in names, f"Names: {names}"
    except AttributeError:
        pytest.skip("named_parameters not implemented")


def test_module_repr():
    """Test Module string representation."""
    class ReprModule(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter([1.0])
    
    m = ReprModule()
    s = repr(m)
    
    assert s is not None, "__repr__ returned None"
    assert len(s) > 0, "Empty repr"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
