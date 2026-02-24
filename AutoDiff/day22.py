"""
Day 22: Linear Layer Module
===========================
Estimated time: 2-3 hours
Prerequisites: Days 15-21 (Tensor operations, matmul, loss functions)

Learning objectives:
- Implement nn.Linear equivalent with learnable weights and bias
- Understand parameter initialization strategies
- Implement forward pass for linear transformation
- Understand how gradients flow through linear layers

Key concepts:
- Linear layer: y = xW^T + b
  - x: input of shape (batch, in_features)
  - W: weight matrix of shape (out_features, in_features)
  - b: bias vector of shape (out_features,)
  - y: output of shape (batch, out_features)

- Weight initialization:
  - Xavier/Glorot: scale = sqrt(2 / (fan_in + fan_out))
  - Kaiming/He: scale = sqrt(2 / fan_in)
  - Important for proper gradient flow at initialization

- Gradient computation:
  - dy/dW = x^T @ upstream_grad (accumulated over batch)
  - dy/db = sum(upstream_grad, axis=0)
  - dy/dx = upstream_grad @ W

Mathematical background:
- Forward: y = xW^T + b
- Backward:
  - dL/dW = dL/dy^T @ x (or equivalently x^T @ dL/dy summed properly)
  - dL/db = sum_batch(dL/dy)
  - dL/dx = dL/dy @ W
"""

import numpy as np
from typing import Tuple, Optional, List


class Tensor:
    """Tensor class with matmul support."""
    
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
        return f"Tensor(shape={self.shape}, data=\n{self.data})"
    
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
        """Reset gradient to zero."""
        self.grad = np.zeros_like(self.data)
    
    @staticmethod
    def unbroadcast(grad, original_shape):
        """Reduce gradient to original shape."""
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
    
    def sum(self, axis=None, keepdims=False):
        """Sum reduction."""
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'sum')
        
        def _backward():
            if axis is None:
                self.grad += np.full(self.shape, out.grad)
            else:
                grad = out.grad
                if not keepdims:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        """Mean reduction."""
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'mean')
        
        if axis is None:
            count = self.data.size
        elif isinstance(axis, int):
            count = self.data.shape[axis]
        else:
            count = np.prod([self.data.shape[ax] for ax in axis])
        
        def _backward():
            if axis is None:
                self.grad += np.full(self.shape, out.grad / count)
            else:
                grad = out.grad / count
                if not keepdims:
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward
        return out
    
    def __pow__(self, n):
        out = Tensor(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n-1)) * out.grad
        out._backward = _backward
        return out
    
    def transpose(self, axes=None):
        """Transpose tensor."""
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
        """Matrix multiplication: self @ other."""
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
    
    def relu(self):
        """ReLU activation."""
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out


# ============================================================================
# Exercise 1: Linear Layer Implementation
# ============================================================================

class Linear:
    """
    Linear (fully connected) layer: y = xW^T + b
    
    Equivalent to PyTorch's nn.Linear.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds learnable bias (default: True)
    
    Shapes:
        - Input: (batch, in_features) or (in_features,)
        - Output: (batch, out_features) or (out_features,)
        - Weight: (out_features, in_features)
        - Bias: (out_features,)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        
        # API hints:
        # - np.sqrt(value) -> square root
        # - np.random.randn(rows, cols) -> random normal values
        # - Xavier init scale = sqrt(2.0 / (fan_in + fan_out))
        # - Weight shape: (out_features, in_features)
        # - Tensor(data) -> create tensor from numpy array
        # - np.zeros(size) -> create zero array for bias
        
        self.weight = None  # Replace
        self.bias = None    # Replace (None if bias=False)
        self.use_bias = bias
    
    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Compute linear transformation.
        
        y = xW^T + b
        
        Args:
            x: Input tensor of shape (batch, in_features) or (in_features,)
        
        Returns:
            Output tensor of shape (batch, out_features) or (out_features,)
        """
        # Handle 1D input
        squeeze_output = False
        if x.data.ndim == 1:
            x = Tensor(x.data.reshape(1, -1), requires_grad=x.requires_grad)
            squeeze_output = True
        
        # API hints:
        # - x @ self.weight.T -> matrix multiplication (batch, in) @ (in, out)
        # - tensor.T -> transpose
        # - out + self.bias -> add bias (broadcasts)
        # - Formula: y = xW^T + b
        
        out = None  # Replace
        
        # Handle 1D output
        if squeeze_output and out is not None:
            out = Tensor(out.data.squeeze(0), out._prev, out._op)
            out._backward = out._backward
        
        return out
    
    def parameters(self) -> List[Tensor]:
        """Return list of parameters."""
        if self.use_bias and self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight] if self.weight is not None else []
    
    def zero_grad(self):
        """Reset all parameter gradients."""
        for p in self.parameters():
            p.zero_grad()


# ============================================================================
# Exercise 2: Kaiming Initialization
# ============================================================================

class LinearKaiming:
    """
    Linear layer with Kaiming/He initialization.
    
    Kaiming init is better for ReLU networks:
    - scale = sqrt(2.0 / fan_in)
    - This accounts for ReLU zeroing out half the values
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # API hints:
        # - Kaiming init scale = sqrt(2.0 / fan_in)
        # - np.sqrt(value) -> square root
        # - np.random.randn(rows, cols) * scale -> scaled random init
        # - Tensor(data) -> create tensor
        # - np.zeros(size) -> zero array for bias
        
        self.weight = None  # Replace
        self.bias = None    # Replace
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """Same forward as Linear."""
        if x.data.ndim == 1:
            x = Tensor(x.data.reshape(1, -1))
        
        # API hints:
        # - x @ self.weight.T -> matrix multiplication
        # - out + self.bias -> add bias if present
        
        return None
    
    def parameters(self) -> List[Tensor]:
        if self.use_bias and self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight] if self.weight is not None else []


# ============================================================================
# Exercise 3: Linear Layer without Bias
# ============================================================================

class LinearNoBias:
    """
    Linear layer without bias term.
    
    Sometimes useful for specific architectures or when
    batch normalization follows the linear layer.
    """
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        
        # API hints:
        # - Xavier scale = sqrt(2.0 / (in_features + out_features))
        # - np.random.randn(out_features, in_features) * scale
        # - Tensor(data) -> create weight tensor
        
        self.weight = None  # Replace
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward without bias."""
        if x.data.ndim == 1:
            x = Tensor(x.data.reshape(1, -1))
        
        # API hints:
        # - x @ self.weight.T -> matrix multiplication
        # - Formula: y = xW^T (no bias term)
        
        return None
    
    def parameters(self) -> List[Tensor]:
        return [self.weight] if self.weight is not None else []


# ============================================================================
# Exercise 4: Multi-Layer Perceptron (MLP)
# ============================================================================

class MLP:
    """
    Simple Multi-Layer Perceptron.
    
    Combines multiple linear layers with ReLU activations.
    
    Example:
        mlp = MLP(784, [128, 64], 10)  # Input: 784, Hidden: [128, 64], Output: 10
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        self.layers = []
        
        # API hints:
        # - sizes = [input_size] + hidden_sizes + [output_size]
        # - Linear(in_size, out_size) -> create linear layer
        # - self.layers.append(layer) -> add to layer list
        # - Loop: for i in range(len(sizes) - 1)
        
        pass
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through all layers.
        
        Applies ReLU after each layer except the last.
        """
        # API hints:
        # - self.layers[:-1] -> all layers except last
        # - self.layers[-1] -> last layer (no activation)
        # - layer(x) -> forward through layer
        # - tensor.relu() -> apply ReLU activation
        # - Pattern: hidden layers get ReLU, output layer doesn't
        
        return None
    
    def parameters(self) -> List[Tensor]:
        """Return all parameters from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        """Reset gradients for all parameters."""
        for p in self.parameters():
            p.zero_grad()


# ============================================================================
# Exercise 5: Gradient Checking
# ============================================================================

def numerical_gradient(f, x: Tensor, eps: float = 1e-5) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.
    
    This is useful for verifying analytical gradients.
    
    Args:
        f: Function that takes Tensor and returns scalar Tensor
        x: Input tensor
        eps: Small perturbation for finite differences
    
    Returns:
        Numerical gradient array with same shape as x
    """
    grad = np.zeros_like(x.data)
    
    # API hints:
    # - np.ndindex(x.shape) -> iterate over all indices
    # - x.data[idx] -> access/modify element at index
    # - f(Tensor(x.data)).data -> evaluate function, get scalar result
    # - Formula: grad[idx] = (f(x+eps) - f(x-eps)) / (2*eps)
    # - Central difference is more accurate than forward difference
    
    return grad


# ============================================================================
# Test Functions
# ============================================================================

def test_linear_forward():
    """Test linear layer forward pass."""
    results = {}
    
    np.random.seed(42)
    layer = Linear(4, 3)
    
    if layer.weight is not None:
        results['weight_shape'] = layer.weight.shape == (3, 4)
    else:
        results['weight_shape'] = False
    
    if layer.bias is not None:
        results['bias_shape'] = layer.bias.shape == (3,)
    else:
        results['bias_shape'] = False
    
    x = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])  # (2, 4)
    y = layer(x)
    
    if y is not None and y.data is not None:
        results['output_shape'] = y.shape == (2, 3)
    else:
        results['output_shape'] = False
    
    return results


def test_linear_backward():
    """Test linear layer backward pass."""
    results = {}
    
    np.random.seed(42)
    layer = Linear(4, 3)
    
    if layer.weight is None or layer.bias is None:
        return {'weight_grad': False, 'bias_grad': False, 'input_grad': False}
    
    x = Tensor([[1.0, 2.0, 3.0, 4.0]])
    y = layer(x)
    
    if y is None:
        return {'weight_grad': False, 'bias_grad': False, 'input_grad': False}
    
    loss = y.sum()
    loss.backward()
    
    results['weight_grad'] = np.any(layer.weight.grad != 0)
    results['bias_grad'] = np.any(layer.bias.grad != 0)
    results['input_grad'] = np.any(x.grad != 0)
    
    return results


def test_xavier_init():
    """Test Xavier initialization variance."""
    results = {}
    
    np.random.seed(42)
    layers = [Linear(100, 100) for _ in range(10)]
    
    variances = []
    for layer in layers:
        if layer.weight is not None:
            variances.append(np.var(layer.weight.data))
    
    if variances:
        expected_var = 2.0 / (100 + 100)
        mean_var = np.mean(variances)
        results['variance_close'] = abs(mean_var - expected_var) < 0.01
    else:
        results['variance_close'] = False
    
    return results


def test_kaiming_init():
    """Test Kaiming initialization variance."""
    results = {}
    
    np.random.seed(42)
    layers = [LinearKaiming(100, 100) for _ in range(10)]
    
    variances = []
    for layer in layers:
        if layer.weight is not None:
            variances.append(np.var(layer.weight.data))
    
    if variances:
        expected_var = 2.0 / 100
        mean_var = np.mean(variances)
        results['variance_close'] = abs(mean_var - expected_var) < 0.02
    else:
        results['variance_close'] = False
    
    return results


def test_mlp_forward():
    """Test MLP forward pass."""
    results = {}
    
    np.random.seed(42)
    mlp = MLP(10, [20, 15], 5)
    
    if mlp.layers:
        results['num_layers'] = len(mlp.layers) == 3
    else:
        results['num_layers'] = False
        results['output_shape'] = False
        return results
    
    x = Tensor(np.random.randn(4, 10))
    y = mlp(x)
    
    if y is not None and y.data is not None:
        results['output_shape'] = y.shape == (4, 5)
    else:
        results['output_shape'] = False
    
    return results


def test_mlp_backward():
    """Test MLP backward pass."""
    results = {}
    
    np.random.seed(42)
    mlp = MLP(10, [20], 5)
    
    x = Tensor(np.random.randn(4, 10))
    y = mlp(x)
    
    if y is None:
        return {'params_have_grad': False}
    
    loss = y.sum()
    loss.backward()
    
    results['params_have_grad'] = all(
        np.any(p.grad != 0) for p in mlp.parameters()
    )
    
    return results


def test_gradient_check():
    """Test numerical gradient matches analytical."""
    results = {}
    
    np.random.seed(42)
    layer = Linear(3, 2)
    
    if layer.weight is None:
        return {'gradient_match': False}
    
    x = Tensor([[1.0, 2.0, 3.0]])
    
    def f(inp):
        y = layer.forward(inp)
        return y.sum() if y is not None else Tensor(0)
    
    y = layer(x)
    if y is None:
        return {'gradient_match': False}
    
    loss = y.sum()
    loss.backward()
    
    num_grad = numerical_gradient(f, x)
    results['gradient_match'] = np.allclose(x.grad, num_grad, rtol=1e-4)
    
    return results


if __name__ == "__main__":
    print("Day 22: Linear Layer Module")
    print("=" * 60)
    
    print("\nLinear Forward:")
    forward_results = test_linear_forward()
    for name, passed in forward_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLinear Backward:")
    backward_results = test_linear_backward()
    for name, passed in backward_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nXavier Initialization:")
    xavier_results = test_xavier_init()
    for name, passed in xavier_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nKaiming Initialization:")
    kaiming_results = test_kaiming_init()
    for name, passed in kaiming_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMLP Forward:")
    mlp_forward_results = test_mlp_forward()
    for name, passed in mlp_forward_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMLP Backward:")
    mlp_backward_results = test_mlp_backward()
    for name, passed in mlp_backward_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nGradient Check:")
    grad_check_results = test_gradient_check()
    for name, passed in grad_check_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day22.py for comprehensive tests!")
