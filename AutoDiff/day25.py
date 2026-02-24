"""
Day 25: Parameter and Module Base Class
=======================================
Estimated time: 3-4 hours
Prerequisites: Days 21-24 (Loss, Linear, Activations, Sequential)

Learning objectives:
- Implement proper Parameter class for learnable weights
- Build robust Module base class with parameter registration
- Understand named_parameters and named_modules
- Implement state_dict save/load functionality
- Create a complete mini deep learning framework

Key concepts:
- Parameter: A Tensor that's marked as learnable
  - Automatically registered when assigned to Module
  - Collected by parameters() for optimizer

- Module: Base class for all neural network components
  - Tracks child modules and parameters
  - Provides train/eval mode switching
  - Supports state dict serialization

- Parameter registration:
  - Direct assignment: self.weight = Parameter(...)
  - Submodule assignment: self.layer = Linear(...)
  - Both automatically registered

Mathematical background:
- Parameters are the learnable variables optimized during training
- State dict allows saving/loading trained models
- Proper registration ensures gradients flow correctly
"""

import numpy as np
from typing import Dict, List, Iterator, Tuple, Optional, Any, Set
from collections import OrderedDict


class Tensor:
    """Tensor class with autodiff support."""
    
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.requires_grad = requires_grad
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    def __repr__(self):
        return f"Tensor(shape={self.shape})"
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = np.ones_like(self.data)
        
        for v in reversed(topo):
            v._backward()
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    def detach(self):
        """Return a new Tensor detached from computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)
    
    def clone(self):
        """Return a copy that shares the computation graph."""
        out = Tensor(self.data.copy(), (self,), 'clone')
        
        def _backward():
            self.grad += out.grad
        out._backward = _backward
        return out
    
    @staticmethod
    def unbroadcast(grad, original_shape):
        while grad.ndim > len(original_shape):
            grad = grad.sum(axis=0)
        for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
            if orig_dim == 1 and grad_dim > 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += Tensor.unbroadcast(out.grad, self.shape)
            other.grad += Tensor.unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += Tensor.unbroadcast(other.data * out.grad, self.shape)
            other.grad += Tensor.unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, n):
        out = Tensor(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n-1)) * out.grad
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'sum')
        
        def _backward():
            if axis is None:
                self.grad += np.full(self.shape, out.grad)
            else:
                grad = out.grad
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'mean')
        count = self.data.size if axis is None else self.data.shape[axis]
        
        def _backward():
            if axis is None:
                self.grad += np.full(self.shape, out.grad / count)
            else:
                grad = out.grad / count
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward
        return out
    
    def transpose(self, axes=None):
        if axes is None:
            axes = tuple(reversed(range(self.data.ndim)))
        out = Tensor(np.transpose(self.data, axes), (self,), 'T')
        inv_axes = [0] * len(axes)
        for i, ax in enumerate(axes):
            inv_axes[ax] = i
        
        def _backward():
            self.grad += np.transpose(out.grad, inv_axes)
        out._backward = _backward
        return out
    
    @property
    def T(self):
        return self.transpose()
    
    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        return self.matmul(other)


# ============================================================================
# Exercise 1: Parameter Class
# ============================================================================

class Parameter(Tensor):
    """
    A Tensor that is automatically registered as a learnable parameter.
    
    Parameters are special Tensors that:
    - Are meant to be learned during training
    - Are automatically collected by Module.parameters()
    - Always require gradients
    
    Example:
        class MyLayer(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(np.random.randn(10, 5))
    """
    
    def __init__(self, data, requires_grad: bool = True):
        """
        Initialize a Parameter.
        
        Args:
            data: Initial values (numpy array or Tensor)
            requires_grad: Whether to compute gradients (default: True)
        """
        # API hints:
        # - isinstance(data, Tensor) -> check if already Tensor
        # - data.data -> extract numpy array from Tensor
        # - super().__init__(data, requires_grad=requires_grad) -> call parent
        
        pass
    
    def __repr__(self):
        return f"Parameter(shape={self.shape})"


# ============================================================================
# Exercise 2: Module Base Class
# ============================================================================

class Module:
    """
    Base class for all neural network modules.
    
    Provides:
    - Automatic parameter registration
    - Child module tracking
    - Train/eval mode switching
    - State dict save/load
    
    Subclass this to create custom layers and models.
    """
    
    def __init__(self):
        """Initialize Module."""
        # API hints:
        # - OrderedDict() -> ordered dictionary for params/modules
        # - self._parameters -> stores Parameter instances
        # - self._modules -> stores child Module instances
        # - self._training -> boolean for train/eval mode
        
        self._parameters = None  # Replace
        self._modules = None     # Replace
        self._training = True
    
    def __setattr__(self, name: str, value: Any):
        """
        Override setattr to automatically register Parameters and Modules.
        
        When you do self.weight = Parameter(...), this method detects
        that it's a Parameter and registers it.
        """
        # API hints:
        # - isinstance(value, Parameter) -> check if Parameter
        # - isinstance(value, Module) -> check if Module
        # - self._parameters[name] = value -> register in dict
        # - object.__setattr__(self, name, value) -> actually set attr
        
        object.__setattr__(self, name, value)  # Replace with registration logic
    
    def __call__(self, *args, **kwargs) -> Tensor:
        """Call forward method."""
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass - override in subclasses."""
        raise NotImplementedError
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Return an iterator over module parameters.
        
        Args:
            recurse: If True, includes parameters of submodules
        
        Yields:
            Parameter objects
        """
        # API hints:
        # - self._parameters.values() -> iterate parameter dict values
        # - yield param -> yield each parameter
        # - yield from module.parameters() -> recursively yield from children
        # - recurse flag controls whether to include submodule params
        
        return iter([])  # Replace
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        """
        Return an iterator over module parameters with names.
        
        Args:
            prefix: Prefix to prepend to parameter names
            recurse: If True, includes parameters of submodules
        
        Yields:
            Tuples of (name, Parameter)
        """
        # API hints:
        # - self._parameters.items() -> (name, param) pairs
        # - f"{prefix}.{name}" if prefix else name -> build full name
        # - yield (full_name, param) -> yield tuple
        # - Recurse into self._modules with updated prefix
        
        return iter([])  # Replace
    
    def modules(self) -> Iterator['Module']:
        """
        Return an iterator over all modules (including self).
        
        Yields:
            Module objects
        """
        # API hints:
        # - yield self -> include this module
        # - self._modules.values() -> child modules
        # - yield from module.modules() -> recursively yield
        
        return iter([])  # Replace
    
    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        """
        Return an iterator over all modules with names.
        
        Yields:
            Tuples of (name, Module)
        """
        # API hints:
        # - yield (prefix, self) -> this module with its name
        # - self._modules.items() -> (name, module) pairs
        # - Build prefix: f"{prefix}.{name}" if prefix else name
        # - yield from module.named_modules(prefix=...) -> recurse
        
        return iter([])  # Replace
    
    def zero_grad(self):
        """Set gradients of all parameters to zero."""
        for param in self.parameters():
            param.zero_grad()
    
    def train(self, mode: bool = True):
        """
        Set module to training mode.
        
        This affects modules like Dropout and BatchNorm.
        """
        self._training = mode
        if self._modules:
            for module in self._modules.values():
                module.train(mode)
        return self
    
    def eval(self):
        """Set module to evaluation mode."""
        return self.train(False)
    
    @property
    def training(self) -> bool:
        return self._training
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary containing all parameters.
        
        This is used for saving models.
        
        Returns:
            Dictionary mapping parameter names to numpy arrays
        """
        # API hints:
        # - OrderedDict() -> create ordered dict
        # - self.named_parameters() -> iterate (name, param) pairs
        # - param.data.copy() -> copy the numpy array
        # - state[name] = data -> store in dict
        
        return {}  # Replace
    
    def load_state_dict(self, state_dict: Dict[str, np.ndarray]):
        """
        Load parameters from a state dictionary.
        
        Args:
            state_dict: Dictionary mapping parameter names to numpy arrays
        """
        # API hints:
        # - self.named_parameters() -> iterate (name, param) pairs
        # - name in state_dict -> check if param exists
        # - param.data = state_dict[name].copy() -> load values
        
        pass
    
    def num_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.data.size for p in self.parameters())
    
    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        if self._modules:
            for name, module in self._modules.items():
                mod_str = repr(module).replace('\n', '\n  ')
                lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return "\n".join(lines)


# ============================================================================
# Exercise 3: Linear Layer with Proper Registration
# ============================================================================

class Linear(Module):
    """
    Linear layer with proper Parameter registration.
    
    y = xW^T + b
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        # API hints:
        # - super().__init__() -> call Module.__init__
        # - Xavier scale = sqrt(2.0 / (in + out))
        # - Parameter(data) -> create learnable parameter
        # - np.random.randn(out, in) * scale -> init weights
        # - np.zeros(out) -> init bias to zeros
        
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """Linear transformation."""
        # API hints:
        # - x.ndim -> check dimensions
        # - x.data.reshape(1, -1) -> make 2D
        # - x @ self.weight.T -> matrix multiply
        # - out + self.bias -> add bias
        # - Formula: y = xW^T + b
        
        return None
    
    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features})"


# ============================================================================
# Exercise 4: Activation Modules
# ============================================================================

class ReLU(Module):
    """ReLU activation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(np.maximum(0, x.data), (x,), 'relu')
        
        def _backward():
            x.grad += (x.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        result = np.where(x.data >= 0,
                         1 / (1 + np.exp(-x.data)),
                         np.exp(x.data) / (1 + np.exp(x.data)))
        out = Tensor(result, (x,), 'sigmoid')
        
        def _backward():
            x.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "Sigmoid()"


class Tanh(Module):
    """Tanh activation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(np.tanh(x.data), (x,), 'tanh')
        
        def _backward():
            x.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "Tanh()"


# ============================================================================
# Exercise 5: Sequential Container
# ============================================================================

class Sequential(Module):
    """Sequential container with proper module registration."""
    
    def __init__(self, *modules: Module):
        # API hints:
        # - super().__init__() -> call Module.__init__
        # - enumerate(modules) -> get (index, module) pairs
        # - setattr(self, str(i), module) -> auto-register modules
        
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """Pass through all modules in sequence."""
        # API hints:
        # - self._modules.values() -> iterate child modules
        # - module(x) -> call module forward
        # - Chain outputs: input to next = output of previous
        
        return None
    
    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx] if self._modules else None
    
    def __len__(self) -> int:
        return len(self._modules) if self._modules else 0
    
    def __iter__(self):
        return iter(self._modules.values()) if self._modules else iter([])


# ============================================================================
# Exercise 6: Model Saving and Loading
# ============================================================================

def save_model(model: Module, filepath: str):
    """
    Save model parameters to a file.
    
    Args:
        model: Module to save
        filepath: Path to save file
    """
    # API hints:
    # - model.state_dict() -> get parameters as dict
    # - np.savez(filepath, **state_dict) -> save to .npz file
    
    pass


def load_model(model: Module, filepath: str):
    """
    Load model parameters from a file.
    
    Args:
        model: Module to load into
        filepath: Path to load file
    """
    # API hints:
    # - np.load(filepath) -> load .npz file
    # - loaded.files -> list of array names
    # - {key: loaded[key] for key in loaded.files} -> convert to dict
    # - model.load_state_dict(state_dict) -> load into model
    
    pass


# ============================================================================
# Test Functions
# ============================================================================

def test_parameter():
    """Test Parameter class."""
    results = {}
    
    try:
        p = Parameter(np.random.randn(3, 4))
        results['creates'] = p is not None and hasattr(p, 'data')
        results['shape'] = p.shape == (3, 4) if hasattr(p, 'shape') else False
        results['requires_grad'] = getattr(p, 'requires_grad', False)
    except Exception as e:
        results['creates'] = False
        results['shape'] = False
        results['requires_grad'] = False
    
    return results


def test_module_registration():
    """Test automatic parameter registration."""
    results = {}
    
    try:
        class TestModule(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter(np.random.randn(5, 3))
                self.bias = Parameter(np.zeros(5))
            
            def forward(self, x):
                return x @ self.weight.T + self.bias
        
        m = TestModule()
        params = list(m.parameters())
        results['registered'] = len(params) == 2
        
        named = dict(m.named_parameters())
        results['named'] = 'weight' in named and 'bias' in named
    except Exception as e:
        results['registered'] = False
        results['named'] = False
    
    return results


def test_submodule_registration():
    """Test submodule registration."""
    results = {}
    
    try:
        class Net(Module):
            def __init__(self):
                super().__init__()
                self.layer1 = Linear(10, 20)
                self.layer2 = Linear(20, 5)
            
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        
        net = Net()
        
        if net._modules is not None:
            results['modules_registered'] = len(net._modules) == 2
        else:
            results['modules_registered'] = False
        
        params = list(net.parameters())
        results['params_collected'] = len(params) == 4
        
        modules = list(net.modules())
        results['modules_iter'] = len(modules) == 3
    except Exception as e:
        results['modules_registered'] = False
        results['params_collected'] = False
        results['modules_iter'] = False
    
    return results


def test_state_dict():
    """Test state dict save/load."""
    results = {}
    
    try:
        class Net(Module):
            def __init__(self):
                super().__init__()
                self.layer = Linear(5, 3)
            
            def forward(self, x):
                return self.layer(x)
        
        net1 = Net()
        state = net1.state_dict()
        
        results['has_state'] = len(state) > 0
        
        net2 = Net()
        old_weight = net2.layer.weight.data.copy() if hasattr(net2.layer, 'weight') else None
        net2.load_state_dict(state)
        
        if old_weight is not None and hasattr(net2.layer, 'weight'):
            results['loaded'] = np.allclose(net2.layer.weight.data, net1.layer.weight.data)
        else:
            results['loaded'] = False
    except Exception as e:
        results['has_state'] = False
        results['loaded'] = False
    
    return results


def test_train_eval_mode():
    """Test train/eval mode switching."""
    results = {}
    
    try:
        class Net(Module):
            def __init__(self):
                super().__init__()
                self.linear = Linear(5, 3)
            
            def forward(self, x):
                return self.linear(x)
        
        net = Net()
        
        net.train()
        results['train_mode'] = net.training
        
        net.eval()
        results['eval_mode'] = not net.training
    except Exception as e:
        results['train_mode'] = False
        results['eval_mode'] = False
    
    return results


def test_sequential():
    """Test Sequential with proper registration."""
    results = {}
    
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
        
        params = list(model.parameters())
        results['params'] = len(params) == 4
        
        x = Tensor(np.random.randn(4, 10))
        y = model(x)
        
        results['forward'] = y is not None and y.shape == (4, 5)
        
        if y is not None:
            y.sum().backward()
            results['backward'] = all(np.any(p.grad != 0) for p in params)
        else:
            results['backward'] = False
    except Exception as e:
        results['params'] = False
        results['forward'] = False
        results['backward'] = False
    
    return results


def test_num_parameters():
    """Test num_parameters method."""
    results = {}
    
    try:
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
        
        num = model.num_parameters()
        expected = 10*20 + 20 + 20*5 + 5
        results['count'] = num == expected
    except Exception as e:
        results['count'] = False
    
    return results


def test_zero_grad():
    """Test zero_grad method."""
    results = {}
    
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
        
        model.zero_grad()
        all_zero = all(np.all(p.grad == 0) for p in model.parameters())
        
        results['zero_grad'] = has_grad and all_zero
    except Exception as e:
        results['zero_grad'] = False
    
    return results


if __name__ == "__main__":
    print("Day 25: Parameter and Module Base Class")
    print("=" * 60)
    
    print("\nParameter Class:")
    param_results = test_parameter()
    for name, passed in param_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nModule Registration:")
    reg_results = test_module_registration()
    for name, passed in reg_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSubmodule Registration:")
    sub_results = test_submodule_registration()
    for name, passed in sub_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nState Dict:")
    state_results = test_state_dict()
    for name, passed in state_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nTrain/Eval Mode:")
    mode_results = test_train_eval_mode()
    for name, passed in mode_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSequential:")
    seq_results = test_sequential()
    for name, passed in seq_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nNum Parameters:")
    num_results = test_num_parameters()
    for name, passed in num_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nZero Grad:")
    zg_results = test_zero_grad()
    for name, passed in zg_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day25.py for comprehensive tests!")
