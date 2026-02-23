"""
Day 23: Activation Modules
==========================
Estimated time: 2-3 hours
Prerequisites: Days 15-22 (Tensor operations, Linear layer)

Learning objectives:
- Implement activation functions as modular components
- Understand activation gradients and their properties
- Learn when to use different activations
- Build reusable activation modules

Key concepts:
- ReLU: f(x) = max(0, x)
  - Gradient: 1 if x > 0, 0 otherwise
  - Pros: Simple, sparse activation, no vanishing gradient for positive values
  - Cons: "Dying ReLU" problem for negative inputs

- Sigmoid: f(x) = 1 / (1 + exp(-x))
  - Gradient: sigmoid(x) * (1 - sigmoid(x))
  - Pros: Output in (0, 1), good for probabilities
  - Cons: Vanishing gradient for large |x|, not zero-centered

- Tanh: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
  - Gradient: 1 - tanh(x)^2
  - Pros: Zero-centered, output in (-1, 1)
  - Cons: Vanishing gradient for large |x|

- LeakyReLU: f(x) = max(alpha*x, x) where alpha is small (e.g., 0.01)
  - Gradient: 1 if x > 0, alpha otherwise
  - Solves dying ReLU problem

- GELU: Gaussian Error Linear Unit (used in Transformers)
  - f(x) = x * Phi(x) where Phi is CDF of standard normal
  - Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

Mathematical background:
- Activations introduce non-linearity (essential for deep learning)
- Without them, stacked linear layers = single linear layer
- Gradient flow determines trainability of deep networks
"""

import numpy as np
from typing import Optional, List


class Tensor:
    """Tensor class with activation support."""
    
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
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                self.grad += np.broadcast_to(grad, self.shape)
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
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
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        """Hyperbolic tangent."""
        out = Tensor(np.tanh(self.data), (self,), 'tanh')
        
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out


# ============================================================================
# Base Module Class (Simplified)
# ============================================================================

class Module:
    """Base class for all modules."""
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def parameters(self) -> List[Tensor]:
        return []


# ============================================================================
# Exercise 1: ReLU Module
# ============================================================================

class ReLU(Module):
    """
    Rectified Linear Unit activation.
    
    f(x) = max(0, x)
    
    Properties:
    - Non-saturating for positive values
    - Sparse activation
    - Computationally efficient
    - Can cause "dying ReLU" problem
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ReLU activation.
        
        Args:
            x: Input tensor
        
        Returns:
            Tensor with ReLU applied element-wise
        
        Gradient: 1 if x > 0, 0 otherwise
        """
        # TODO: Implement ReLU forward pass
        # HINT:
        # result = np.maximum(0, x.data)
        # out = Tensor(result, (x,), 'relu')
        
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            # Gradient is 1 where x > 0, 0 elsewhere
            # HINT: x.grad += (x.data > 0) * out.grad
            pass  # Replace
        
        if out is not None:
            out._backward = _backward
        return out


# ============================================================================
# Exercise 2: Sigmoid Module
# ============================================================================

class Sigmoid(Module):
    """
    Sigmoid activation.
    
    f(x) = 1 / (1 + exp(-x))
    
    Properties:
    - Output in (0, 1) - good for probabilities
    - Smooth and differentiable everywhere
    - Suffers from vanishing gradient for large |x|
    - Not zero-centered
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply sigmoid activation.
        
        Args:
            x: Input tensor
        
        Returns:
            Tensor with sigmoid applied element-wise
        
        Gradient: sigmoid(x) * (1 - sigmoid(x))
        """
        # TODO: Implement numerically stable sigmoid
        # HINT: For stability, handle positive and negative x differently
        # For x >= 0: sigmoid = 1 / (1 + exp(-x))
        # For x < 0: sigmoid = exp(x) / (1 + exp(x))
        # Or just use: sigmoid = np.where(x.data >= 0, 
        #                                  1 / (1 + np.exp(-x.data)),
        #                                  np.exp(x.data) / (1 + np.exp(x.data)))
        
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
            # HINT: x.grad += out.data * (1 - out.data) * out.grad
            pass  # Replace
        
        if out is not None:
            out._backward = _backward
        return out


# ============================================================================
# Exercise 3: Tanh Module
# ============================================================================

class Tanh(Module):
    """
    Hyperbolic tangent activation.
    
    f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Properties:
    - Output in (-1, 1) - zero-centered
    - Stronger gradients than sigmoid
    - Still suffers from vanishing gradient for large |x|
    """
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply tanh activation.
        
        Args:
            x: Input tensor
        
        Returns:
            Tensor with tanh applied element-wise
        
        Gradient: 1 - tanh(x)^2
        """
        # TODO: Implement tanh forward pass
        # HINT: Use np.tanh for numerical stability
        # result = np.tanh(x.data)
        # out = Tensor(result, (x,), 'tanh')
        
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            # d(tanh)/dx = 1 - tanh^2
            # HINT: x.grad += (1 - out.data ** 2) * out.grad
            pass  # Replace
        
        if out is not None:
            out._backward = _backward
        return out


# ============================================================================
# Exercise 4: LeakyReLU Module
# ============================================================================

class LeakyReLU(Module):
    """
    Leaky ReLU activation.
    
    f(x) = x if x > 0, else alpha * x
    
    Properties:
    - Solves "dying ReLU" problem
    - Small gradient for negative values
    - alpha is typically 0.01
    """
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Leaky ReLU activation.
        
        Args:
            x: Input tensor
        
        Returns:
            Tensor with Leaky ReLU applied element-wise
        
        Gradient: 1 if x > 0, alpha otherwise
        """
        # TODO: Implement Leaky ReLU forward pass
        # HINT: result = np.where(x.data > 0, x.data, self.alpha * x.data)
        
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            # Gradient is 1 for x > 0, alpha for x <= 0
            # HINT: 
            # grad_mask = np.where(x.data > 0, 1, self.alpha)
            # x.grad += grad_mask * out.grad
            pass  # Replace
        
        if out is not None:
            out._backward = _backward
        return out


# ============================================================================
# Exercise 5: GELU Module
# ============================================================================

class GELU(Module):
    """
    Gaussian Error Linear Unit.
    
    f(x) = x * Phi(x)
    
    Where Phi(x) is the CDF of standard normal distribution.
    
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    Properties:
    - Smooth approximation to ReLU
    - Used in BERT, GPT, and other Transformers
    - Better gradient flow than ReLU
    """
    
    def __init__(self, approximate: bool = True):
        self.approximate = approximate
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply GELU activation.
        
        Args:
            x: Input tensor
        
        Returns:
            Tensor with GELU applied element-wise
        """
        if self.approximate:
            # TODO: Implement GELU approximation
            # HINT:
            # sqrt_2_pi = np.sqrt(2 / np.pi)
            # inner = sqrt_2_pi * (x.data + 0.044715 * x.data ** 3)
            # result = 0.5 * x.data * (1 + np.tanh(inner))
            
            out = None  # Replace
            
            # TODO: Implement backward pass (approximate)
            def _backward():
                # This is complex - use chain rule through tanh
                # HINT:
                # sqrt_2_pi = np.sqrt(2 / np.pi)
                # inner = sqrt_2_pi * (x.data + 0.044715 * x.data ** 3)
                # tanh_inner = np.tanh(inner)
                # sech2 = 1 - tanh_inner ** 2
                # inner_grad = sqrt_2_pi * (1 + 3 * 0.044715 * x.data ** 2)
                # grad = 0.5 * (1 + tanh_inner) + 0.5 * x.data * sech2 * inner_grad
                # x.grad += grad * out.grad
                pass  # Replace
            
            if out is not None:
                out._backward = _backward
            return out
        else:
            from scipy import special
            result = x.data * special.ndtr(x.data)
            out = Tensor(result, (x,), 'gelu_exact')
            
            def _backward():
                pdf = np.exp(-0.5 * x.data ** 2) / np.sqrt(2 * np.pi)
                cdf = special.ndtr(x.data)
                x.grad += (cdf + x.data * pdf) * out.grad
            
            out._backward = _backward
            return out


# ============================================================================
# Exercise 6: Softplus Module
# ============================================================================

class Softplus(Module):
    """
    Softplus activation.
    
    f(x) = log(1 + exp(x))
    
    Properties:
    - Smooth approximation to ReLU
    - Always positive output
    - Gradient is sigmoid
    """
    
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        self.beta = beta
        self.threshold = threshold
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Softplus activation.
        
        For numerical stability, use linear approximation for large x.
        """
        # TODO: Implement numerically stable softplus
        # HINT:
        # For x > threshold: softplus(x) ≈ x (to avoid overflow)
        # Otherwise: softplus(x) = log(1 + exp(beta*x)) / beta
        # result = np.where(x.data * self.beta > self.threshold,
        #                   x.data,
        #                   np.log(1 + np.exp(self.beta * x.data)) / self.beta)
        
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            # d(softplus)/dx = sigmoid(beta*x)
            # HINT:
            # sigmoid = 1 / (1 + np.exp(-self.beta * x.data))
            # x.grad += sigmoid * out.grad
            pass  # Replace
        
        if out is not None:
            out._backward = _backward
        return out


# ============================================================================
# Exercise 7: ELU Module
# ============================================================================

class ELU(Module):
    """
    Exponential Linear Unit.
    
    f(x) = x if x > 0, else alpha * (exp(x) - 1)
    
    Properties:
    - Smooth and differentiable everywhere
    - Zero-centered outputs
    - Saturates for negative values (more robust to noise)
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply ELU activation.
        """
        # TODO: Implement ELU forward pass
        # HINT:
        # result = np.where(x.data > 0, x.data, self.alpha * (np.exp(x.data) - 1))
        
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            # d(ELU)/dx = 1 if x > 0, else alpha * exp(x) = output + alpha
            # HINT:
            # grad = np.where(x.data > 0, 1, out.data + self.alpha)
            # x.grad += grad * out.grad
            pass  # Replace
        
        if out is not None:
            out._backward = _backward
        return out


# ============================================================================
# Test Functions
# ============================================================================

def test_relu():
    """Test ReLU activation."""
    results = {}
    
    relu = ReLU()
    x = Tensor([-2, -1, 0, 1, 2])
    y = relu(x)
    
    if y is not None and y.data is not None:
        expected = np.array([0, 0, 0, 1, 2])
        results['forward'] = np.allclose(y.data, expected)
        
        y.backward()
        expected_grad = np.array([0, 0, 0, 1, 1])
        results['backward'] = np.allclose(x.grad, expected_grad)
    else:
        results['forward'] = False
        results['backward'] = False
    
    return results


def test_sigmoid():
    """Test Sigmoid activation."""
    results = {}
    
    sigmoid = Sigmoid()
    x = Tensor([0.0])
    y = sigmoid(x)
    
    if y is not None and y.data is not None:
        results['center'] = np.allclose(y.data, [0.5])
        
        y.backward()
        results['grad_at_zero'] = np.allclose(x.grad, [0.25])
    else:
        results['center'] = False
        results['grad_at_zero'] = False
    
    x2 = Tensor([-5.0, 0.0, 5.0])
    y2 = sigmoid(x2)
    
    if y2 is not None:
        results['range'] = np.all(y2.data > 0) and np.all(y2.data < 1)
    else:
        results['range'] = False
    
    return results


def test_tanh():
    """Test Tanh activation."""
    results = {}
    
    tanh = Tanh()
    x = Tensor([0.0])
    y = tanh(x)
    
    if y is not None and y.data is not None:
        results['center'] = np.allclose(y.data, [0])
        
        y.backward()
        results['grad_at_zero'] = np.allclose(x.grad, [1])
    else:
        results['center'] = False
        results['grad_at_zero'] = False
    
    x2 = Tensor([-5.0, 0.0, 5.0])
    y2 = tanh(x2)
    
    if y2 is not None:
        results['range'] = np.all(y2.data > -1) and np.all(y2.data < 1)
    else:
        results['range'] = False
    
    return results


def test_leaky_relu():
    """Test Leaky ReLU activation."""
    results = {}
    
    lrelu = LeakyReLU(alpha=0.1)
    x = Tensor([-2, -1, 0, 1, 2])
    y = lrelu(x)
    
    if y is not None and y.data is not None:
        expected = np.array([-0.2, -0.1, 0, 1, 2])
        results['forward'] = np.allclose(y.data, expected)
        
        y.backward()
        expected_grad = np.array([0.1, 0.1, 0.1, 1, 1])
        results['backward'] = np.allclose(x.grad, expected_grad)
    else:
        results['forward'] = False
        results['backward'] = False
    
    return results


def test_gelu():
    """Test GELU activation."""
    results = {}
    
    gelu = GELU(approximate=True)
    x = Tensor([0.0])
    y = gelu(x)
    
    if y is not None and y.data is not None:
        results['center'] = np.allclose(y.data, [0], atol=1e-5)
    else:
        results['center'] = False
    
    x2 = Tensor([1.0, 2.0, 3.0])
    y2 = gelu(x2)
    
    if y2 is not None:
        results['positive_similar_to_relu'] = np.all(y2.data > 0)
        results['less_than_input'] = np.all(y2.data < x2.data)
    else:
        results['positive_similar_to_relu'] = False
        results['less_than_input'] = False
    
    return results


def test_softplus():
    """Test Softplus activation."""
    results = {}
    
    sp = Softplus()
    x = Tensor([0.0])
    y = sp(x)
    
    if y is not None and y.data is not None:
        results['at_zero'] = np.allclose(y.data, [np.log(2)])
    else:
        results['at_zero'] = False
    
    x2 = Tensor([-10, 0, 10])
    y2 = sp(x2)
    
    if y2 is not None:
        results['always_positive'] = np.all(y2.data > 0)
    else:
        results['always_positive'] = False
    
    return results


def test_elu():
    """Test ELU activation."""
    results = {}
    
    elu = ELU(alpha=1.0)
    x = Tensor([-2, -1, 0, 1, 2])
    y = elu(x)
    
    if y is not None and y.data is not None:
        results['positive_pass_through'] = np.allclose(y.data[3:], [1, 2])
        results['negative_bounded'] = np.all(y.data[:2] > -1) and np.all(y.data[:2] < 0)
    else:
        results['positive_pass_through'] = False
        results['negative_bounded'] = False
    
    return results


if __name__ == "__main__":
    print("Day 23: Activation Modules")
    print("=" * 60)
    
    print("\nReLU:")
    relu_results = test_relu()
    for name, passed in relu_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSigmoid:")
    sigmoid_results = test_sigmoid()
    for name, passed in sigmoid_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nTanh:")
    tanh_results = test_tanh()
    for name, passed in tanh_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLeaky ReLU:")
    lrelu_results = test_leaky_relu()
    for name, passed in lrelu_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nGELU:")
    gelu_results = test_gelu()
    for name, passed in gelu_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSoftplus:")
    softplus_results = test_softplus()
    for name, passed in softplus_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nELU:")
    elu_results = test_elu()
    for name, passed in elu_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day23.py for comprehensive tests!")
