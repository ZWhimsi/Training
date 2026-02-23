"""Test Suite for Day 25: Parameter and Module Base Class"""

import numpy as np
import sys
import os
import tempfile
from typing import Tuple

try:
    from day25 import (
        Tensor,
        Parameter,
        Module,
        Linear,
        ReLU,
        Sigmoid,
        Tanh,
        Sequential,
        save_model,
        load_model,
        test_parameter,
        test_module_registration,
        test_submodule_registration,
        test_state_dict,
        test_train_eval_mode,
        test_sequential,
        test_num_parameters,
        test_zero_grad
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_parameter_creation() -> Tuple[bool, str]:
    """Test Parameter creation."""
    try:
        p = Parameter(np.random.randn(3, 4))
        
        if p is None:
            return False, "Parameter is None"
        if not hasattr(p, 'data'):
            return False, "No data attribute"
        if p.shape != (3, 4):
            return False, f"shape = {p.shape}"
        
        return True, "Parameter created successfully"
    except Exception as e:
        return False, str(e)


def test_parameter_is_tensor() -> Tuple[bool, str]:
    """Test that Parameter inherits from Tensor."""
    try:
        p = Parameter(np.array([1, 2, 3]))
        
        if not isinstance(p, Tensor):
            return False, "Parameter is not a Tensor"
        
        return True, "Parameter inherits from Tensor"
    except Exception as e:
        return False, str(e)


def test_parameter_requires_grad() -> Tuple[bool, str]:
    """Test Parameter requires_grad default."""
    try:
        p = Parameter(np.array([1, 2, 3]))
        
        if not getattr(p, 'requires_grad', False):
            return False, "requires_grad is False"
        
        return True, "Parameter requires_grad=True"
    except Exception as e:
        return False, str(e)


def test_module_init() -> Tuple[bool, str]:
    """Test Module initialization."""
    try:
        class TestModule(Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        m = TestModule()
        
        if m._parameters is None:
            return False, "_parameters is None"
        if m._modules is None:
            return False, "_modules is None"
        
        return True, "Module initialized"
    except Exception as e:
        return False, str(e)


def test_parameter_registration() -> Tuple[bool, str]:
    """Test automatic parameter registration."""
    try:
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(np.random.randn(3, 4))
            
            def forward(self, x):
                return x
        
        m = TestModule()
        
        if 'weight' not in m._parameters:
            return False, "weight not registered"
        
        return True, "Parameter auto-registered"
    except Exception as e:
        return False, str(e)


def test_module_registration() -> Tuple[bool, str]:
    """Test automatic submodule registration."""
    try:
        class Parent(Module):
            def __init__(self):
                super().__init__()
                self.child = Linear(10, 5)
            
            def forward(self, x):
                return self.child(x)
        
        m = Parent()
        
        if 'child' not in m._modules:
            return False, "child not registered"
        
        return True, "Submodule auto-registered"
    except Exception as e:
        return False, str(e)


def test_parameters_iterator() -> Tuple[bool, str]:
    """Test parameters() iterator."""
    try:
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.w1 = Parameter(np.random.randn(3, 4))
                self.w2 = Parameter(np.random.randn(4, 5))
            
            def forward(self, x):
                return x
        
        m = TestModule()
        params = list(m.parameters())
        
        if len(params) != 2:
            return False, f"len = {len(params)}, expected 2"
        
        return True, "parameters() returns 2 params"
    except Exception as e:
        return False, str(e)


def test_recursive_parameters() -> Tuple[bool, str]:
    """Test recursive parameter collection."""
    try:
        class Net(Module):
            def __init__(self):
                super().__init__()
                self.layer1 = Linear(10, 20)
                self.layer2 = Linear(20, 5)
            
            def forward(self, x):
                return self.layer2(self.layer1(x))
        
        net = Net()
        params = list(net.parameters())
        
        if len(params) != 4:
            return False, f"len = {len(params)}, expected 4"
        
        return True, "Recursive params collected"
    except Exception as e:
        return False, str(e)


def test_named_parameters() -> Tuple[bool, str]:
    """Test named_parameters() iterator."""
    try:
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(np.random.randn(3, 4))
            
            def forward(self, x):
                return x
        
        m = TestModule()
        named = dict(m.named_parameters())
        
        if 'weight' not in named:
            return False, "weight not in named_parameters"
        
        return True, "named_parameters works"
    except Exception as e:
        return False, str(e)


def test_named_parameters_recursive() -> Tuple[bool, str]:
    """Test named_parameters with submodules."""
    try:
        class Net(Module):
            def __init__(self):
                super().__init__()
                self.layer = Linear(10, 5)
            
            def forward(self, x):
                return self.layer(x)
        
        net = Net()
        named = dict(net.named_parameters())
        
        has_weight = any('weight' in name for name in named.keys())
        has_bias = any('bias' in name for name in named.keys())
        
        if not has_weight or not has_bias:
            return False, f"Keys: {list(named.keys())}"
        
        return True, "Nested named_parameters works"
    except Exception as e:
        return False, str(e)


def test_modules_iterator() -> Tuple[bool, str]:
    """Test modules() iterator."""
    try:
        class Net(Module):
            def __init__(self):
                super().__init__()
                self.layer1 = Linear(10, 5)
                self.layer2 = Linear(5, 2)
            
            def forward(self, x):
                return self.layer2(self.layer1(x))
        
        net = Net()
        mods = list(net.modules())
        
        if len(mods) != 3:
            return False, f"len = {len(mods)}, expected 3"
        
        return True, "modules() iterates all"
    except Exception as e:
        return False, str(e)


def test_named_modules() -> Tuple[bool, str]:
    """Test named_modules() iterator."""
    try:
        class Net(Module):
            def __init__(self):
                super().__init__()
                self.layer = Linear(10, 5)
            
            def forward(self, x):
                return self.layer(x)
        
        net = Net()
        named = dict(net.named_modules())
        
        if '' not in named:
            return False, "Root module not found"
        if 'layer' not in named:
            return False, "layer not in named_modules"
        
        return True, "named_modules works"
    except Exception as e:
        return False, str(e)


def test_state_dict_save() -> Tuple[bool, str]:
    """Test state_dict generation."""
    try:
        class Net(Module):
            def __init__(self):
                super().__init__()
                self.layer = Linear(5, 3)
            
            def forward(self, x):
                return self.layer(x)
        
        net = Net()
        state = net.state_dict()
        
        if not state:
            return False, "state_dict is empty"
        
        has_values = any(isinstance(v, np.ndarray) for v in state.values())
        if not has_values:
            return False, "No numpy arrays in state"
        
        return True, "state_dict created"
    except Exception as e:
        return False, str(e)


def test_state_dict_load() -> Tuple[bool, str]:
    """Test state_dict loading."""
    try:
        class Net(Module):
            def __init__(self):
                super().__init__()
                self.layer = Linear(5, 3)
            
            def forward(self, x):
                return self.layer(x)
        
        net1 = Net()
        state = net1.state_dict()
        
        net2 = Net()
        if hasattr(net2.layer, 'weight'):
            net2.layer.weight.data = np.random.randn(*net2.layer.weight.shape)
        
        net2.load_state_dict(state)
        
        if hasattr(net1.layer, 'weight') and hasattr(net2.layer, 'weight'):
            if not np.allclose(net1.layer.weight.data, net2.layer.weight.data):
                return False, "Weights don't match after load"
        
        return True, "state_dict loaded"
    except Exception as e:
        return False, str(e)


def test_train_mode() -> Tuple[bool, str]:
    """Test train mode."""
    try:
        m = Module()
        m._parameters = {}
        m._modules = {}
        
        m.train()
        if not m.training:
            return False, "training = False after train()"
        
        return True, "train() sets training=True"
    except Exception as e:
        return False, str(e)


def test_eval_mode() -> Tuple[bool, str]:
    """Test eval mode."""
    try:
        m = Module()
        m._parameters = {}
        m._modules = {}
        
        m.eval()
        if m.training:
            return False, "training = True after eval()"
        
        return True, "eval() sets training=False"
    except Exception as e:
        return False, str(e)


def test_recursive_train_eval() -> Tuple[bool, str]:
    """Test train/eval propagates to submodules."""
    try:
        class Net(Module):
            def __init__(self):
                super().__init__()
                self.layer = Linear(5, 3)
            
            def forward(self, x):
                return self.layer(x)
        
        net = Net()
        net.eval()
        
        if net.training:
            return False, "Parent not in eval"
        if net.layer.training:
            return False, "Child not in eval"
        
        net.train()
        
        if not net.training:
            return False, "Parent not in train"
        if not net.layer.training:
            return False, "Child not in train"
        
        return True, "train/eval propagates"
    except Exception as e:
        return False, str(e)


def test_sequential_params() -> Tuple[bool, str]:
    """Test Sequential parameter collection."""
    try:
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
        
        params = list(model.parameters())
        if len(params) != 4:
            return False, f"len = {len(params)}, expected 4"
        
        return True, "Sequential collects params"
    except Exception as e:
        return False, str(e)


def test_sequential_forward() -> Tuple[bool, str]:
    """Test Sequential forward pass."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
        
        x = Tensor(np.random.randn(4, 10))
        y = model(x)
        
        if y is None:
            return False, "forward returned None"
        if y.shape != (4, 5):
            return False, f"shape = {y.shape}"
        
        return True, "Sequential forward works"
    except Exception as e:
        return False, str(e)


def test_sequential_backward() -> Tuple[bool, str]:
    """Test Sequential backward pass."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
        
        x = Tensor(np.random.randn(4, 10))
        y = model(x)
        
        if y is None:
            return False, "forward returned None"
        
        y.sum().backward()
        
        params = list(model.parameters())
        all_grad = all(np.any(p.grad != 0) for p in params)
        
        if not all_grad:
            return False, "Some params missing gradient"
        
        return True, "Sequential backward works"
    except Exception as e:
        return False, str(e)


def test_zero_grad() -> Tuple[bool, str]:
    """Test zero_grad method."""
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(5, 10),
            ReLU(),
            Linear(10, 3)
        )
        
        x = Tensor(np.random.randn(4, 5))
        y = model(x)
        
        if y is not None:
            y.sum().backward()
        
        has_grad = any(np.any(p.grad != 0) for p in model.parameters())
        if not has_grad:
            return False, "No gradients computed"
        
        model.zero_grad()
        
        all_zero = all(np.all(p.grad == 0) for p in model.parameters())
        if not all_zero:
            return False, "Gradients not zeroed"
        
        return True, "zero_grad works"
    except Exception as e:
        return False, str(e)


def test_num_parameters() -> Tuple[bool, str]:
    """Test num_parameters method."""
    try:
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
        
        num = model.num_parameters()
        expected = 10*20 + 20 + 20*5 + 5
        
        if num != expected:
            return False, f"num = {num}, expected {expected}"
        
        return True, f"num_parameters = {expected}"
    except Exception as e:
        return False, str(e)


def test_linear_forward() -> Tuple[bool, str]:
    """Test Linear layer forward."""
    try:
        np.random.seed(42)
        layer = Linear(4, 3)
        
        x = Tensor(np.random.randn(2, 4))
        y = layer(x)
        
        if y is None:
            return False, "forward returned None"
        if y.shape != (2, 3):
            return False, f"shape = {y.shape}"
        
        return True, "Linear forward works"
    except Exception as e:
        return False, str(e)


def test_linear_backward() -> Tuple[bool, str]:
    """Test Linear layer backward."""
    try:
        np.random.seed(42)
        layer = Linear(4, 3)
        
        x = Tensor(np.random.randn(2, 4))
        y = layer(x)
        
        if y is None:
            return False, "forward returned None"
        
        y.sum().backward()
        
        if not hasattr(layer, 'weight') or layer.weight is None:
            return False, "No weight"
        
        if np.all(layer.weight.grad == 0):
            return False, "Weight gradient is zero"
        
        return True, "Linear backward works"
    except Exception as e:
        return False, str(e)


def test_save_load_model() -> Tuple[bool, str]:
    """Test save_model and load_model."""
    try:
        np.random.seed(42)
        
        model1 = Sequential(
            Linear(5, 10),
            ReLU(),
            Linear(10, 3)
        )
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name
        
        try:
            save_model(model1, filepath)
            
            if not os.path.exists(filepath):
                return False, "File not created"
            
            model2 = Sequential(
                Linear(5, 10),
                ReLU(),
                Linear(10, 3)
            )
            
            load_model(model2, filepath)
            
            state1 = model1.state_dict()
            state2 = model2.state_dict()
            
            if not state1 or not state2:
                return True, "save/load ran (state_dict empty)"
            
            for key in state1:
                if key in state2:
                    if not np.allclose(state1[key], state2[key]):
                        return False, f"Mismatch in {key}"
            
            return True, "save/load model works"
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    except Exception as e:
        return False, str(e)


def test_against_pytorch() -> Tuple[bool, str]:
    """Test against PyTorch nn.Module."""
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
        
        our_params = list(our_model.parameters())
        torch_params = list(torch_model.parameters())
        
        if len(our_params) != len(torch_params):
            return False, f"Param count mismatch: {len(our_params)} vs {len(torch_params)}"
        
        torch_model[0].weight.data = torch.tensor(our_params[0].data.copy())
        torch_model[0].bias.data = torch.tensor(our_params[1].data.copy())
        torch_model[2].weight.data = torch.tensor(our_params[2].data.copy())
        torch_model[2].bias.data = torch.tensor(our_params[3].data.copy())
        
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
        
        return True, "Matches PyTorch"
    except ImportError:
        return True, "PyTorch not installed (skipped)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("parameter_creation", test_parameter_creation),
        ("parameter_is_tensor", test_parameter_is_tensor),
        ("parameter_requires_grad", test_parameter_requires_grad),
        ("module_init", test_module_init),
        ("parameter_registration", test_parameter_registration),
        ("module_registration", test_module_registration),
        ("parameters_iterator", test_parameters_iterator),
        ("recursive_parameters", test_recursive_parameters),
        ("named_parameters", test_named_parameters),
        ("named_parameters_recursive", test_named_parameters_recursive),
        ("modules_iterator", test_modules_iterator),
        ("named_modules", test_named_modules),
        ("state_dict_save", test_state_dict_save),
        ("state_dict_load", test_state_dict_load),
        ("train_mode", test_train_mode),
        ("eval_mode", test_eval_mode),
        ("recursive_train_eval", test_recursive_train_eval),
        ("sequential_params", test_sequential_params),
        ("sequential_forward", test_sequential_forward),
        ("sequential_backward", test_sequential_backward),
        ("zero_grad", test_zero_grad),
        ("num_parameters", test_num_parameters),
        ("linear_forward", test_linear_forward),
        ("linear_backward", test_linear_backward),
        ("save_load_model", test_save_load_model),
        ("against_pytorch", test_against_pytorch),
    ]
    
    print(f"\n{'='*60}")
    print("Day 25: Parameter and Module Base Class - Tests")
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
