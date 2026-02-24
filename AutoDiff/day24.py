"""
Day 24: Sequential Container
============================
Estimated time: 2-3 hours
Prerequisites: Days 21-23 (Loss functions, Linear layer, Activations)

Learning objectives:
- Implement nn.Sequential equivalent container
- Understand module composition patterns
- Learn to chain modules with automatic forward propagation
- Build reusable network architectures

Key concepts:
- Sequential container: Chains modules in order
  - forward(x) calls each module in sequence
  - Parameters collected from all child modules

- Module composition patterns:
  - Sequential: Linear chain (A -> B -> C)
  - Parallel: Multiple branches
  - Residual: Skip connections (x + F(x))

- Container responsibilities:
  - Forward propagation through children
  - Parameter aggregation
  - Gradient zeroing

Mathematical background:
- For modules f1, f2, f3:
  - Sequential(f1, f2, f3)(x) = f3(f2(f1(x)))
  - Chain rule applies naturally through composition
"""

import numpy as np
from typing import List, Union, Iterator, Tuple, Optional


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
                    grad = np.expand_dims(grad, axis=axis) if isinstance(axis, int) else grad
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
# Base Module Class
# ============================================================================

class Module:
    """
    Base class for all neural network modules.
    
    All custom modules should inherit from this class.
    """
    
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def parameters(self) -> List[Tensor]:
        """Return list of all parameters."""
        return []
    
    def zero_grad(self):
        """Reset gradients of all parameters to zero."""
        for p in self.parameters():
            p.zero_grad()
    
    def train(self):
        """Set module to training mode."""
        pass
    
    def eval(self):
        """Set module to evaluation mode."""
        pass


# ============================================================================
# Linear Layer
# ============================================================================

class Linear(Module):
    """Linear layer: y = xW^T + b"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(np.random.randn(out_features, in_features) * scale)
        self.bias = Tensor(np.zeros(out_features)) if bias else None
    
    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim == 1:
            x = Tensor(x.data.reshape(1, -1))
        out = x @ self.weight.T
        if self.use_bias:
            out = out + self.bias
        return out
    
    def parameters(self) -> List[Tensor]:
        if self.use_bias and self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]
    
    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features}, bias={self.use_bias})"


# ============================================================================
# Activation Modules
# ============================================================================

class ReLU(Module):
    """ReLU activation: max(0, x)"""
    
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(np.maximum(0, x.data), (x,), 'relu')
        
        def _backward():
            x.grad += (x.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    
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
    """Tanh activation"""
    
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(np.tanh(x.data), (x,), 'tanh')
        
        def _backward():
            x.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "Tanh()"


# ============================================================================
# Exercise 1: Sequential Container
# ============================================================================

class Sequential(Module):
    """
    A sequential container that chains modules.
    
    Modules are added in order and executed sequentially.
    
    Example:
        model = Sequential(
            Linear(784, 128),
            ReLU(),
            Linear(128, 10)
        )
        output = model(input)
    """
    
    def __init__(self, *modules: Module):
        """
        Initialize with variable number of modules.
        
        Args:
            *modules: Modules to add in sequence
        """
        # API hints:
        # - list(modules) -> convert args tuple to list
        # - self._modules -> store as instance attribute
        
        self._modules = None  # Replace
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Pass input through all modules in sequence.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor after passing through all modules
        """
        # API hints:
        # - for module in self._modules: -> iterate modules
        # - module(x) -> call module's forward
        # - Chain: output of one becomes input of next
        
        return None
    
    def parameters(self) -> List[Tensor]:
        """
        Collect parameters from all modules.
        
        Returns:
            List of all parameter tensors
        """
        # API hints:
        # - module.parameters() -> get params from each module
        # - list.extend(items) -> add multiple items to list
        # - Iterate self._modules to collect all params
        
        return []  # Replace
    
    def __getitem__(self, idx: int) -> Module:
        """Get module by index."""
        return self._modules[idx] if self._modules else None
    
    def __len__(self) -> int:
        """Return number of modules."""
        return len(self._modules) if self._modules else 0
    
    def __iter__(self) -> Iterator[Module]:
        """Iterate over modules."""
        return iter(self._modules) if self._modules else iter([])
    
    def __repr__(self):
        lines = ["Sequential("]
        if self._modules:
            for i, module in enumerate(self._modules):
                lines.append(f"  ({i}): {module}")
        lines.append(")")
        return "\n".join(lines)


# ============================================================================
# Exercise 2: ModuleList Container
# ============================================================================

class ModuleList(Module):
    """
    A list of modules that properly tracks parameters.
    
    Unlike a regular Python list, ModuleList properly registers
    all modules for parameter collection.
    
    Example:
        layers = ModuleList([Linear(10, 10) for _ in range(5)])
    """
    
    def __init__(self, modules: List[Module] = None):
        """
        Initialize with optional list of modules.
        
        Args:
            modules: List of modules (default: empty list)
        """
        # API hints:
        # - list(modules) if modules else [] -> handle None
        # - self._modules -> store module list
        
        self._modules = None  # Replace
    
    def append(self, module: Module):
        """Add a module to the end."""
        # API hints:
        # - self._modules.append(module) -> add to list
        pass
    
    def extend(self, modules: List[Module]):
        """Extend with a list of modules."""
        # API hints:
        # - self._modules.extend(modules) -> add multiple modules
        pass
    
    def parameters(self) -> List[Tensor]:
        """Collect parameters from all modules."""
        # API hints:
        # - module.parameters() -> get params from module
        # - list.extend() -> add multiple items
        # - Iterate self._modules
        
        return []  # Replace
    
    def __getitem__(self, idx: int) -> Module:
        return self._modules[idx] if self._modules else None
    
    def __setitem__(self, idx: int, module: Module):
        if self._modules:
            self._modules[idx] = module
    
    def __len__(self) -> int:
        return len(self._modules) if self._modules else 0
    
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules) if self._modules else iter([])
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("ModuleList does not implement forward")


# ============================================================================
# Exercise 3: Residual Block
# ============================================================================

class Residual(Module):
    """
    Residual connection: output = x + F(x)
    
    Skip connections help gradient flow in deep networks.
    
    Example:
        block = Residual(
            Sequential(
                Linear(64, 64),
                ReLU(),
                Linear(64, 64)
            )
        )
    """
    
    def __init__(self, fn: Module):
        """
        Initialize with a module to wrap.
        
        Args:
            fn: The module F in x + F(x)
        """
        # API hints:
        # - self.fn = fn -> store the wrapped module
        
        self.fn = None  # Replace
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply residual connection.
        
        Returns: x + fn(x)
        """
        # API hints:
        # - self.fn(x) -> pass through wrapped module
        # - x + result -> add skip connection
        # - Formula: output = x + F(x)
        
        return None
    
    def parameters(self) -> List[Tensor]:
        """Return parameters from wrapped module."""
        return self.fn.parameters() if self.fn else []


# ============================================================================
# Exercise 4: Lambda Module
# ============================================================================

class Lambda(Module):
    """
    Wraps a function as a module.
    
    Useful for simple transformations that don't have parameters.
    
    Example:
        flatten = Lambda(lambda x: Tensor(x.data.reshape(x.data.shape[0], -1)))
    """
    
    def __init__(self, fn):
        """
        Initialize with a function.
        
        Args:
            fn: Function that takes Tensor and returns Tensor
        """
        # API hints:
        # - self.fn = fn -> store the function
        
        self.fn = None  # Replace
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply the function."""
        # API hints:
        # - self.fn(x) -> call stored function
        
        return None


# ============================================================================
# Exercise 5: Flatten Module
# ============================================================================

class Flatten(Module):
    """
    Flatten tensor to 2D (batch, features).
    
    Commonly used between conv layers and linear layers.
    """
    
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        """
        Args:
            start_dim: First dim to flatten (default: 1, preserves batch)
            end_dim: Last dim to flatten (default: -1, to end)
        """
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Flatten dimensions from start_dim to end_dim.
        
        For typical use (batch, C, H, W) -> (batch, C*H*W)
        """
        # API hints:
        # - x.shape -> get current shape
        # - x.data.reshape(new_shape) -> reshape data
        # - self.start_dim, self.end_dim -> dimension range
        # - new_shape = shape[:start] + (-1,) + shape[end:]
        # - Backward: reshape gradient back to original shape
        
        return None
    
    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"


# ============================================================================
# Exercise 6: Dropout Module
# ============================================================================

class Dropout(Module):
    """
    Dropout regularization layer.
    
    During training: Randomly zeros elements with probability p
    During inference: Identity function (scaled by (1-p) during training)
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of zeroing an element
        """
        self.p = p
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply dropout.
        
        Training: Zero random elements and scale by 1/(1-p)
        Inference: Pass through unchanged
        """
        # API hints:
        # - self.training -> check if in training mode
        # - self.p -> dropout probability
        # - np.random.binomial(1, 1-p, shape) -> generate mask
        # - Scale by 1/(1-p) to maintain expected value
        # - In eval mode, return input unchanged
        
        return None
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def __repr__(self):
        return f"Dropout(p={self.p})"


# ============================================================================
# Test Functions
# ============================================================================

def test_sequential_forward():
    """Test Sequential forward pass."""
    results = {}
    
    np.random.seed(42)
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2)
    )
    
    if model._modules is not None:
        results['num_modules'] = len(model) == 3
    else:
        results['num_modules'] = False
    
    x = Tensor(np.random.randn(2, 4))
    y = model(x)
    
    if y is not None and y.data is not None:
        results['output_shape'] = y.shape == (2, 2)
    else:
        results['output_shape'] = False
    
    return results


def test_sequential_parameters():
    """Test Sequential parameter collection."""
    results = {}
    
    np.random.seed(42)
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2)
    )
    
    params = model.parameters()
    results['has_params'] = len(params) == 4
    
    return results


def test_sequential_backward():
    """Test Sequential backward pass."""
    results = {}
    
    np.random.seed(42)
    model = Sequential(
        Linear(4, 8),
        ReLU(),
        Linear(8, 2)
    )
    
    x = Tensor(np.random.randn(2, 4))
    y = model(x)
    
    if y is None:
        return {'backward': False}
    
    loss = y.sum()
    loss.backward()
    
    params = model.parameters()
    results['backward'] = all(np.any(p.grad != 0) for p in params) if params else False
    
    return results


def test_module_list():
    """Test ModuleList."""
    results = {}
    
    np.random.seed(42)
    layers = ModuleList([Linear(10, 10) for _ in range(3)])
    
    if layers._modules is not None:
        results['length'] = len(layers) == 3
    else:
        results['length'] = False
    
    params = layers.parameters()
    results['params'] = len(params) == 6
    
    return results


def test_residual():
    """Test Residual connection."""
    results = {}
    
    np.random.seed(42)
    block = Residual(Linear(4, 4, bias=False))
    
    if block.fn is None:
        return {'forward': False, 'skip_connection': False}
    
    x = Tensor([[1, 0, 0, 0]])
    y = block(x)
    
    if y is not None:
        results['forward'] = y.shape == (1, 4)
        results['skip_connection'] = y.data[0, 0] != 0
    else:
        results['forward'] = False
        results['skip_connection'] = False
    
    return results


def test_flatten():
    """Test Flatten module."""
    results = {}
    
    flatten = Flatten()
    x = Tensor(np.random.randn(2, 3, 4, 5))
    y = flatten(x)
    
    if y is not None and y.data is not None:
        results['shape'] = y.shape == (2, 60)
        
        y.backward()
        results['backward'] = x.grad.shape == (2, 3, 4, 5)
    else:
        results['shape'] = False
        results['backward'] = False
    
    return results


def test_dropout():
    """Test Dropout module."""
    results = {}
    
    np.random.seed(42)
    dropout = Dropout(p=0.5)
    
    x = Tensor(np.ones((100, 100)))
    
    dropout.train()
    y_train = dropout(x)
    
    if y_train is not None:
        zero_ratio = np.mean(y_train.data == 0)
        results['drops_training'] = 0.3 < zero_ratio < 0.7
    else:
        results['drops_training'] = False
    
    dropout.eval()
    x2 = Tensor(np.ones((10, 10)))
    y_eval = dropout(x2)
    
    if y_eval is not None:
        results['identity_eval'] = np.allclose(y_eval.data, x2.data)
    else:
        results['identity_eval'] = False
    
    return results


def test_deep_network():
    """Test building a deep network with Sequential."""
    results = {}
    
    np.random.seed(42)
    model = Sequential(
        Linear(784, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, 64),
        ReLU(),
        Linear(64, 10)
    )
    
    if model._modules is None:
        return {'forward': False, 'backward': False}
    
    x = Tensor(np.random.randn(32, 784))
    y = model(x)
    
    if y is not None:
        results['forward'] = y.shape == (32, 10)
        
        loss = y.sum()
        loss.backward()
        
        params = model.parameters()
        results['backward'] = all(np.any(p.grad != 0) for p in params) if params else False
    else:
        results['forward'] = False
        results['backward'] = False
    
    return results


if __name__ == "__main__":
    print("Day 24: Sequential Container")
    print("=" * 60)
    
    print("\nSequential Forward:")
    seq_forward = test_sequential_forward()
    for name, passed in seq_forward.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSequential Parameters:")
    seq_params = test_sequential_parameters()
    for name, passed in seq_params.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSequential Backward:")
    seq_backward = test_sequential_backward()
    for name, passed in seq_backward.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nModuleList:")
    ml_results = test_module_list()
    for name, passed in ml_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nResidual:")
    res_results = test_residual()
    for name, passed in res_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nFlatten:")
    flat_results = test_flatten()
    for name, passed in flat_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nDropout:")
    drop_results = test_dropout()
    for name, passed in drop_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nDeep Network:")
    deep_results = test_deep_network()
    for name, passed in deep_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day24.py for comprehensive tests!")
