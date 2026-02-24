"""
Day 28: Adam Optimizer
======================
Estimated time: 3-4 hours
Prerequisites: Day 26-27 (SGD, Momentum)

Learning objectives:
- Understand adaptive learning rate methods
- Implement Adam optimizer with bias correction
- Compare Adam variants (AdaGrad, RMSprop)
- Learn when to use Adam vs SGD

Key concepts:
- Adaptive Learning Rates:
  - Different parameters may need different learning rates
  - Parameters with large gradients get smaller updates
  - Parameters with small gradients get larger updates

- Adam = Adaptive Moment Estimation:
  - Combines momentum (first moment) with RMSprop (second moment)
  - Bias correction for early iterations
  - Generally works well out of the box

Mathematical background:

Adam update rule:
    m_t = β₁ * m_{t-1} + (1 - β₁) * g_t        # First moment (momentum)
    v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²       # Second moment (squared grad)
    
    # Bias correction (important for early steps)
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    
    θ_{t+1} = θ_t - lr * m̂_t / (√v̂_t + ε)

Default hyperparameters (from paper):
    β₁ = 0.9    (momentum coefficient)
    β₂ = 0.999  (squared gradient coefficient)
    ε = 1e-8    (numerical stability)
    lr = 0.001  (learning rate)

Why bias correction?
    - m_0 = 0 and v_0 = 0 initially
    - Early estimates are biased toward zero
    - Correction: divide by (1 - β^t) to unbias

AdaGrad (precursor):
    v_t = v_{t-1} + g_t²
    θ_{t+1} = θ_t - lr * g_t / (√v_t + ε)
    Problem: Learning rate monotonically decreases

RMSprop (fixes AdaGrad):
    v_t = ρ * v_{t-1} + (1 - ρ) * g_t²
    θ_{t+1} = θ_t - lr * g_t / (√v_t + ε)
    Uses exponential moving average instead of sum
"""

import numpy as np
from typing import List, Iterator, Dict, Any, Optional
from collections import OrderedDict
import math


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
    
    @property
    def T(self):
        out = Tensor(self.data.T, (self,), 'T')
        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out


class Parameter(Tensor):
    """A Tensor marked as a learnable parameter."""
    
    def __init__(self, data, requires_grad: bool = True):
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data, requires_grad=requires_grad)
    
    def __repr__(self):
        return f"Parameter(shape={self.shape})"


class Module:
    """Base class for neural network modules."""
    
    def __init__(self):
        self._parameters: Dict[str, Parameter] = OrderedDict()
        self._modules: Dict[str, 'Module'] = OrderedDict()
        self._training = True
    
    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            if hasattr(self, '_parameters') and self._parameters is not None:
                self._parameters[name] = value
        elif isinstance(value, Module):
            if hasattr(self, '_modules') and self._modules is not None:
                self._modules[name] = value
        object.__setattr__(self, name, value)
    
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if self._parameters:
            for param in self._parameters.values():
                yield param
        if recurse and self._modules:
            for module in self._modules.values():
                yield from module.parameters(recurse=True)
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()


class Linear(Module):
    """Linear layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Parameter(np.random.randn(out_features, in_features) * scale)
        if bias:
            self.bias = Parameter(np.zeros(out_features))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = Tensor(x.data.reshape(1, -1))
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


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


class Sequential(Module):
    """Sequential container."""
    
    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            setattr(self, str(i), module)
    
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x


class Optimizer:
    """Base class for optimizers."""
    
    def __init__(self, params, defaults: Dict):
        self.defaults = defaults
        self.param_groups = []
        self.state: Dict[int, Dict] = {}
        
        params = list(params)
        if len(params) == 0:
            raise ValueError("optimizer got empty parameter list")
        
        if isinstance(params[0], dict):
            for group in params:
                self.add_param_group(group)
        else:
            self.add_param_group({'params': params})
    
    def add_param_group(self, param_group: Dict):
        params = list(param_group['params'])
        param_group['params'] = params
        for key, value in self.defaults.items():
            param_group.setdefault(key, value)
        self.param_groups.append(param_group)
    
    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.zero_grad()
    
    def step(self):
        raise NotImplementedError


# ============================================================================
# Exercise 1: AdaGrad Optimizer
# ============================================================================

class Adagrad(Optimizer):
    """
    Adagrad optimizer - adapts learning rate per-parameter.
    
    AdaGrad accumulates squared gradients and divides the learning rate
    by the square root of this sum. This gives smaller updates for
    frequently updated parameters.
    
    Update rule:
        v_t = v_{t-1} + g_t²
        θ_{t+1} = θ_t - lr * g_t / (√v_t + ε)
    
    Pros:
        - Automatic per-parameter learning rate
        - Good for sparse gradients
    
    Cons:
        - Learning rate monotonically decreases
        - Can stop learning too early
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.01)
        eps: Term added for numerical stability (default: 1e-10)
    """
    
    def __init__(self, params, lr: float = 0.01, eps: float = 1e-10):
        """
        Initialize Adagrad optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            eps: Small constant for numerical stability
        """
        # API hints:
        # - Validate: lr >= 0.0
        # - defaults = {'lr': lr, 'eps': eps}
        # - super().__init__(params, defaults)
        
        pass
    
    def step(self):
        """
        Perform Adagrad update.
        
        For each parameter:
        1. Accumulate squared gradient: v += g²
        2. Update: θ -= lr * g / (√v + ε)
        """
        # API hints:
        # - self.param_groups -> iterate groups
        # - id(p) -> unique state key
        # - self.state[param_id] = {'sum': np.zeros_like(p.data)}
        # - state['sum'] += grad ** 2
        # - np.sqrt(arr) -> square root
        # - Formula: θ -= lr * g / (√sum + ε)
        
        pass


# ============================================================================
# Exercise 2: RMSprop Optimizer
# ============================================================================

class RMSprop(Optimizer):
    """
    RMSprop optimizer - fixes AdaGrad's decaying learning rate.
    
    Instead of accumulating all squared gradients, RMSprop uses an
    exponential moving average. This prevents the learning rate from
    decreasing too quickly.
    
    Update rule:
        v_t = α * v_{t-1} + (1 - α) * g_t²
        θ_{t+1} = θ_t - lr * g_t / (√v_t + ε)
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.01)
        alpha: Smoothing constant (default: 0.99)
        eps: Term for numerical stability (default: 1e-8)
    """
    
    def __init__(self, params, lr: float = 0.01, alpha: float = 0.99, 
                 eps: float = 1e-8):
        """
        Initialize RMSprop optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            alpha: Decay rate for moving average
            eps: Small constant for numerical stability
        """
        # API hints:
        # - Validate: lr >= 0.0, 0 <= alpha <= 1
        # - defaults = {'lr': lr, 'alpha': alpha, 'eps': eps}
        # - super().__init__(params, defaults)
        
        pass
    
    def step(self):
        """
        Perform RMSprop update.
        
        For each parameter:
        1. Update moving average: v = α * v + (1 - α) * g²
        2. Update: θ -= lr * g / (√v + ε)
        """
        # API hints:
        # - self.param_groups -> iterate groups
        # - id(p) -> unique state key
        # - self.state[param_id] = {'square_avg': np.zeros_like(p.data)}
        # - Formula: v = α * v + (1 - α) * g²
        # - np.sqrt(arr) -> square root
        # - Formula: θ -= lr * g / (√v + ε)
        
        pass


# ============================================================================
# Exercise 3: Adam Optimizer
# ============================================================================

class Adam(Optimizer):
    """
    Adam optimizer - Adaptive Moment Estimation.
    
    Adam combines ideas from momentum (first moment) and RMSprop (second 
    moment), with bias correction for early training steps.
    
    Update rule:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t        # First moment
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²       # Second moment
        m̂_t = m_t / (1 - β₁^t)                     # Bias correction
        v̂_t = v_t / (1 - β₂^t)                     # Bias correction
        θ_{t+1} = θ_t - lr * m̂_t / (√v̂_t + ε)
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for moment estimates (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
    
    Example:
        optimizer = Adam(model.parameters(), lr=0.001)
        # or with custom betas:
        optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    """
    
    def __init__(self, params, lr: float = 0.001, 
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            betas: (β₁, β₂) for moment estimates
            eps: Small constant for numerical stability
        """
        # API hints:
        # - Validate: lr >= 0.0, 0 <= beta1 < 1, 0 <= beta2 < 1
        # - defaults = {'lr': lr, 'betas': betas, 'eps': eps}
        # - self._step_count = 0 -> track iteration for bias correction
        # - super().__init__(params, defaults)
        
        self._step_count = 0
        pass
    
    def step(self):
        """
        Perform Adam update with bias correction.
        
        For each parameter:
        1. Update first moment: m = β₁ * m + (1 - β₁) * g
        2. Update second moment: v = β₂ * v + (1 - β₂) * g²
        3. Bias correct: m̂ = m / (1 - β₁^t), v̂ = v / (1 - β₂^t)
        4. Update: θ -= lr * m̂ / (√v̂ + ε)
        """
        # API hints:
        # - self._step_count += 1 -> increment iteration
        # - beta1, beta2 = group['betas']
        # - state = {'exp_avg': zeros, 'exp_avg_sq': zeros}
        # - m = β₁ * m + (1 - β₁) * g (first moment)
        # - v = β₂ * v + (1 - β₂) * g² (second moment)
        # - bias_correction = 1 - beta ** step_count
        # - m_hat = m / bias_correction1, v_hat = v / bias_correction2
        # - Formula: θ -= lr * m_hat / (√v_hat + ε)
        
        pass


# ============================================================================
# Exercise 4: AdamW (Adam with Weight Decay)
# ============================================================================

class AdamW(Optimizer):
    """
    AdamW optimizer - Adam with decoupled weight decay.
    
    Standard L2 regularization doesn't work well with adaptive optimizers.
    AdamW applies weight decay directly to parameters, not to gradients.
    
    Update rule:
        m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
        v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)
        θ_{t+1} = θ_t - lr * (m̂_t / (√v̂_t + ε) + λ * θ_t)
    
    Note: Weight decay (λ) is applied AFTER the Adam update, not to gradients.
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 0.001)
        betas: Coefficients for moment estimates (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
    """
    
    def __init__(self, params, lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        """
        Initialize AdamW optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            betas: (β₁, β₂) for moment estimates
            eps: Small constant for numerical stability
            weight_decay: Weight decay coefficient
        """
        # API hints:
        # - Validate: lr >= 0.0, weight_decay >= 0.0
        # - defaults = {'lr': ..., 'betas': ..., 'eps': ..., 'weight_decay': ...}
        # - self._step_count = 0
        # - super().__init__(params, defaults)
        
        self._step_count = 0
        pass
    
    def step(self):
        """
        Perform AdamW update with decoupled weight decay.
        
        Key difference from Adam: weight decay applied directly to params.
        """
        # API hints:
        # - Same as Adam for moment updates and bias correction
        # - group['weight_decay'] -> weight decay coefficient
        # - adam_update = m_hat / (√v_hat + ε)
        # - Decoupled decay: p.data -= lr * (adam_update + λ * p.data)
        # - Key: weight decay on params directly, NOT on gradients
        
        pass


# ============================================================================
# Exercise 5: Comparing Optimizers
# ============================================================================

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error loss."""
    diff = pred - target
    return (diff * diff).mean()


def compare_optimizers(X: np.ndarray, Y: np.ndarray, 
                       epochs: int = 100) -> Dict[str, List[float]]:
    """
    Compare different optimizers on the same problem.
    
    Args:
        X: Input data
        Y: Target data
        epochs: Number of training iterations
    
    Returns:
        Dict mapping optimizer name to list of losses
    """
    # API hints:
    # - optimizers = {'name': lambda params: Optimizer(params, ...)}
    # - np.random.seed(42) -> same init for fair comparison
    # - Sequential(Linear(...), ReLU(), Linear(...)) -> model
    # - model.parameters() -> params for optimizer
    # - optimizer.zero_grad(), loss.backward(), optimizer.step()
    # - Track losses per optimizer
    
    return {}  # Replace


# Simple SGD for comparison
class SGD(Optimizer):
    """Basic SGD optimizer."""
    
    def __init__(self, params, lr: float = 0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {'lr': lr}
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is not None:
                    p.data -= lr * p.grad


# ============================================================================
# Test Functions
# ============================================================================

def test_adagrad():
    """Test Adagrad optimizer."""
    results = {}
    
    try:
        np.random.seed(42)
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        opt = Adagrad([p], lr=0.1)
        
        results['creates'] = opt is not None
        
        for _ in range(10):
            p.grad = np.array([0.1, 0.2, 0.3])
            opt.step()
        
        results['updates'] = not np.allclose(p.data, [1.0, 2.0, 3.0])
        
        has_sum = any('sum' in s for s in opt.state.values())
        results['accumulates'] = has_sum
    except Exception as e:
        results['creates'] = False
        results['updates'] = False
        results['accumulates'] = False
    
    return results


def test_rmsprop():
    """Test RMSprop optimizer."""
    results = {}
    
    try:
        np.random.seed(42)
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        opt = RMSprop([p], lr=0.1, alpha=0.9)
        
        results['creates'] = opt is not None
        
        for _ in range(10):
            p.grad = np.array([0.1, 0.2, 0.3])
            opt.step()
        
        results['updates'] = not np.allclose(p.data, [1.0, 2.0, 3.0])
    except Exception as e:
        results['creates'] = False
        results['updates'] = False
    
    return results


def test_adam():
    """Test Adam optimizer."""
    results = {}
    
    try:
        np.random.seed(42)
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        opt = Adam([p], lr=0.1, betas=(0.9, 0.999))
        
        results['creates'] = opt is not None
        
        for _ in range(10):
            p.grad = np.array([0.1, 0.2, 0.3])
            opt.step()
        
        results['updates'] = not np.allclose(p.data, [1.0, 2.0, 3.0])
        
        has_moments = any('exp_avg' in s and 'exp_avg_sq' in s 
                         for s in opt.state.values())
        results['has_moments'] = has_moments
    except Exception as e:
        results['creates'] = False
        results['updates'] = False
        results['has_moments'] = False
    
    return results


def test_adam_bias_correction():
    """Test Adam bias correction."""
    results = {}
    
    try:
        np.random.seed(42)
        
        p1 = Parameter(np.array([5.0]))
        opt1 = Adam([p1], lr=0.1, betas=(0.9, 0.999))
        
        p1.grad = np.array([1.0])
        opt1.step()
        
        first_update = 5.0 - p1.data[0]
        results['first_step_large'] = first_update > 0.05
    except Exception as e:
        results['first_step_large'] = False
    
    return results


def test_adamw():
    """Test AdamW optimizer."""
    results = {}
    
    try:
        np.random.seed(42)
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        opt = AdamW([p], lr=0.1, weight_decay=0.1)
        
        results['creates'] = opt is not None
        
        for _ in range(10):
            p.grad = np.zeros_like(p.data)
            opt.step()
        
        results['decays'] = np.all(np.abs(p.data) < np.array([1.0, 2.0, 3.0]))
    except Exception as e:
        results['creates'] = False
        results['decays'] = False
    
    return results


def test_compare_optimizers():
    """Test optimizer comparison."""
    results = {}
    
    try:
        np.random.seed(42)
        X = np.random.randn(20, 4)
        Y = np.random.randn(20, 2)
        
        comparison = compare_optimizers(X, Y, epochs=50)
        
        results['runs'] = len(comparison) > 0
        
        if comparison:
            all_converge = all(
                losses[-1] < losses[0] 
                for losses in comparison.values()
            )
            results['all_converge'] = all_converge
        else:
            results['all_converge'] = False
    except Exception as e:
        results['runs'] = False
        results['all_converge'] = False
    
    return results


if __name__ == "__main__":
    print("Day 28: Adam Optimizer")
    print("=" * 60)
    
    print("\nAdagrad:")
    ada_results = test_adagrad()
    for name, passed in ada_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRMSprop:")
    rms_results = test_rmsprop()
    for name, passed in rms_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nAdam:")
    adam_results = test_adam()
    for name, passed in adam_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nAdam Bias Correction:")
    bias_results = test_adam_bias_correction()
    for name, passed in bias_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nAdamW:")
    adamw_results = test_adamw()
    for name, passed in adamw_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nCompare Optimizers:")
    comp_results = test_compare_optimizers()
    for name, passed in comp_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day28.py for comprehensive tests!")
