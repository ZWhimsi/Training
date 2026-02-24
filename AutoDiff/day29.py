"""
Day 29: Gradient Clipping and Regularization
=============================================
Estimated time: 3-4 hours
Prerequisites: Day 26-28 (Optimizers)

Learning objectives:
- Implement gradient clipping by value and by norm
- Understand exploding/vanishing gradient problems
- Implement L1 and L2 regularization
- Build weight decay into optimizers

Key concepts:
- Gradient Clipping: Prevents exploding gradients
  - Clip by value: Clamp each gradient element
  - Clip by norm: Scale gradients if total norm exceeds threshold

- Regularization: Prevents overfitting
  - L1: Promotes sparsity (many zero weights)
  - L2: Penalizes large weights (weight decay)
  - Dropout: Random zeroing during training

Mathematical background:

Gradient Clipping by Value:
    g_clipped = clamp(g, -clip_value, clip_value)

Gradient Clipping by Norm:
    total_norm = sqrt(Σ ||g_i||²)
    if total_norm > max_norm:
        g_i = g_i * (max_norm / total_norm)

L1 Regularization (Lasso):
    L_total = L_original + λ * Σ|θ|
    ∂L_total/∂θ = ∂L_original/∂θ + λ * sign(θ)

L2 Regularization (Ridge / Weight Decay):
    L_total = L_original + (λ/2) * Σθ²
    ∂L_total/∂θ = ∂L_original/∂θ + λ * θ
    
    Equivalent to:
    θ_{t+1} = θ_t - lr * grad - lr * λ * θ_t
            = (1 - lr * λ) * θ_t - lr * grad

Exploding Gradients:
    - Gradients grow exponentially through layers
    - Common in RNNs and deep networks
    - Symptoms: NaN/Inf losses, unstable training

Vanishing Gradients:
    - Gradients shrink exponentially through layers
    - Network stops learning
    - Solutions: ReLU, skip connections, careful init
"""

import numpy as np
from typing import List, Iterator, Dict, Any, Optional, Union
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
    
    def train(self, mode: bool = True):
        self._training = mode
        if self._modules:
            for module in self._modules.values():
                module.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    @property
    def training(self) -> bool:
        return self._training


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


# ============================================================================
# Exercise 1: Gradient Clipping by Value
# ============================================================================

def clip_grad_value_(parameters: Iterator[Parameter], 
                     clip_value: float) -> None:
    """
    Clip gradients by value (in-place).
    
    Each gradient element is clamped to [-clip_value, clip_value].
    
    Args:
        parameters: Iterator of parameters with gradients
        clip_value: Maximum absolute value for any gradient element
    
    Example:
        loss.backward()
        clip_grad_value_(model.parameters(), clip_value=1.0)
        optimizer.step()
    """
    # API hints:
    # - for p in parameters -> iterate params
    # - p.grad -> access gradient
    # - np.clip(arr, -value, value, out=arr) -> clamp in-place
    
    pass


# ============================================================================
# Exercise 2: Gradient Clipping by Norm
# ============================================================================

def clip_grad_norm_(parameters: Union[Iterator[Parameter], List[Parameter]], 
                    max_norm: float, 
                    norm_type: float = 2.0) -> float:
    """
    Clip gradients by total norm (in-place).
    
    If total gradient norm exceeds max_norm, scale all gradients so
    the total norm equals max_norm.
    
    Args:
        parameters: Parameters with gradients
        max_norm: Maximum allowed norm
        norm_type: Type of norm (2 = L2, inf = max)
    
    Returns:
        Total gradient norm before clipping
    
    Example:
        loss.backward()
        total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    """
    # API hints:
    # - list(parameters) -> materialize iterator
    # - For L2: total_norm = sqrt(Σ||g||²)
    # - For inf: total_norm = max(abs(g))
    # - clip_coef = max_norm / (total_norm + eps)
    # - if clip_coef < 1: scale all grads by clip_coef
    # - Formula: g *= (max_norm / total_norm)
    
    return 0.0  # Replace


# ============================================================================
# Exercise 3: L1 Regularization
# ============================================================================

def l1_regularization(parameters: Iterator[Parameter], 
                      lambda_l1: float) -> Tensor:
    """
    Compute L1 regularization term.
    
    L1 penalty = λ * Σ|θ|
    
    L1 regularization promotes sparsity - many weights become exactly zero.
    This is useful for feature selection.
    
    Args:
        parameters: Model parameters
        lambda_l1: Regularization strength
    
    Returns:
        Regularization loss term
    """
    # API hints:
    # - np.abs(p.data).sum() -> sum of absolute values
    # - Tensor(np.array(0.0)) -> initialize accumulator
    # - reg_loss + Tensor(...) -> accumulate
    # - Formula: L1 = λ * Σ|θ|
    
    return Tensor(np.array(0.0))  # Replace


def apply_l1_gradient_(parameters: Iterator[Parameter], 
                       lambda_l1: float) -> None:
    """
    Apply L1 regularization gradient directly to parameter gradients.
    
    L1 gradient = λ * sign(θ)
    
    Args:
        parameters: Model parameters
        lambda_l1: Regularization strength
    """
    # API hints:
    # - np.sign(p.data) -> sign of parameter values
    # - p.grad += λ * sign(θ) -> add to gradient
    
    pass


# ============================================================================
# Exercise 4: L2 Regularization (Weight Decay)
# ============================================================================

def l2_regularization(parameters: Iterator[Parameter], 
                      lambda_l2: float) -> Tensor:
    """
    Compute L2 regularization term.
    
    L2 penalty = (λ/2) * Σθ²
    
    L2 regularization penalizes large weights, encouraging smaller values.
    Also known as weight decay when applied directly to updates.
    
    Args:
        parameters: Model parameters
        lambda_l2: Regularization strength
    
    Returns:
        Regularization loss term
    """
    # API hints:
    # - (p.data ** 2).sum() -> sum of squared values
    # - Tensor(np.array(0.0)) -> initialize accumulator
    # - Formula: L2 = (λ/2) * Σθ²
    
    return Tensor(np.array(0.0))  # Replace


def apply_l2_gradient_(parameters: Iterator[Parameter], 
                       lambda_l2: float) -> None:
    """
    Apply L2 regularization gradient directly to parameter gradients.
    
    L2 gradient = λ * θ
    
    This is equivalent to weight decay when applied before the update.
    
    Args:
        parameters: Model parameters
        lambda_l2: Regularization strength
    """
    # API hints:
    # - p.grad += λ * p.data -> add weight decay gradient
    
    pass


# ============================================================================
# Exercise 5: SGD with Weight Decay
# ============================================================================

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


class SGD(Optimizer):
    """
    SGD with momentum and weight decay.
    
    Weight decay can be applied in two ways:
    1. L2 regularization: Add λ*θ to gradient
    2. Decoupled weight decay: Subtract λ*θ directly from param
    
    When decoupled=False (default):
        g = g + weight_decay * θ
        v = momentum * v + g
        θ = θ - lr * v
    
    When decoupled=True (AdamW-style):
        v = momentum * v + g
        θ = θ - lr * v - lr * weight_decay * θ
    
    Args:
        params: Parameters to optimize
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: L2 penalty coefficient
        decoupled: If True, use decoupled weight decay
    """
    
    def __init__(self, params, lr: float, momentum: float = 0.0,
                 weight_decay: float = 0.0, decoupled: bool = False):
        """Initialize SGD with weight decay."""
        # API hints:
        # - Validate: lr >= 0.0
        # - defaults = {'lr': lr, 'momentum': ..., 'weight_decay': ..., 'decoupled': ...}
        # - super().__init__(params, defaults)
        
        pass
    
    def step(self):
        """
        Perform SGD step with optional momentum and weight decay.
        """
        # API hints:
        # - grad = p.grad.copy() -> don't modify original
        # - L2 (coupled): grad += weight_decay * p.data
        # - Momentum: v = momentum * v + grad; update = v
        # - No momentum: update = grad
        # - p.data -= lr * update
        # - Decoupled: p.data -= lr * weight_decay * p.data (after update)
        
        pass


# ============================================================================
# Exercise 6: Dropout Layer
# ============================================================================

class Dropout(Module):
    """
    Dropout layer for regularization.
    
    During training: Randomly zeros elements with probability p,
    and scales remaining elements by 1/(1-p).
    
    During evaluation: No dropout, pass input unchanged.
    
    Args:
        p: Probability of dropping an element (default: 0.5)
    
    Example:
        dropout = Dropout(p=0.5)
        dropout.train()   # Enable dropout
        out = dropout(x)  # 50% of elements zeroed, rest scaled by 2
        
        dropout.eval()    # Disable dropout
        out = dropout(x)  # No change to input
    """
    
    def __init__(self, p: float = 0.5):
        """
        Initialize Dropout layer.
        
        Args:
            p: Dropout probability (0 to 1)
        """
        # API hints:
        # - super().__init__()
        # - Validate: 0 <= p <= 1
        # - self.p = p
        
        super().__init__()
        self.p = 0.5  # Replace
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply dropout.
        
        Training: Zero p% of elements, scale by 1/(1-p)
        Eval: Pass through unchanged
        """
        # API hints:
        # - self.training -> check if in training mode
        # - if not training or p==0: return x unchanged
        # - mask = (np.random.random(shape) > p).astype(np.float64)
        # - scale = 1.0 / (1.0 - p) -> maintain expected value
        # - Backward: x.grad += out.grad * mask * scale
        
        return x  # Replace
    
    def __repr__(self):
        return f"Dropout(p={self.p})"


# ============================================================================
# Exercise 7: Training with Regularization
# ============================================================================

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error loss."""
    diff = pred - target
    return (diff * diff).mean()


def train_with_regularization(model: Module, optimizer: Optimizer,
                              X: np.ndarray, Y: np.ndarray,
                              epochs: int, 
                              lambda_l1: float = 0.0,
                              lambda_l2: float = 0.0,
                              clip_norm: Optional[float] = None) -> Dict:
    """
    Training loop with regularization and gradient clipping.
    
    Args:
        model: Neural network
        optimizer: Optimizer
        X: Training inputs
        Y: Training targets
        epochs: Number of epochs
        lambda_l1: L1 regularization strength
        lambda_l2: L2 regularization strength
        clip_norm: Max gradient norm (None = no clipping)
    
    Returns:
        Dict with losses, grad_norms, and sparsity metrics
    """
    # API hints:
    # - history = {'losses': [], 'reg_losses': [], 'grad_norms': [], 'weight_norms': []}
    # - optimizer.zero_grad() -> clear gradients
    # - mse_loss(pred, y) -> data loss
    # - l1_regularization(...), l2_regularization(...) -> reg terms
    # - loss.backward() -> compute gradients
    # - clip_grad_norm_(...) -> optional clipping
    # - optimizer.step() -> update params
    # - Track metrics per epoch
    
    return {}  # Replace


# ============================================================================
# Test Functions
# ============================================================================

def test_clip_grad_value():
    """Test gradient clipping by value."""
    results = {}
    
    try:
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        p.grad = np.array([5.0, -10.0, 2.0])
        
        clip_grad_value_([p], clip_value=3.0)
        
        results['clips_max'] = p.grad[1] >= -3.0
        results['clips_min'] = p.grad[0] <= 3.0
        results['preserves'] = np.isclose(p.grad[2], 2.0)
    except Exception as e:
        results['clips_max'] = False
        results['clips_min'] = False
        results['preserves'] = False
    
    return results


def test_clip_grad_norm():
    """Test gradient clipping by norm."""
    results = {}
    
    try:
        p = Parameter(np.array([3.0, 4.0]))  # norm = 5
        p.grad = np.array([3.0, 4.0])
        
        total_norm = clip_grad_norm_([p], max_norm=2.5)
        
        results['returns_norm'] = np.isclose(total_norm, 5.0)
        
        new_norm = np.sqrt(np.sum(p.grad ** 2))
        results['clips_to_max'] = np.isclose(new_norm, 2.5, rtol=1e-4)
    except Exception as e:
        results['returns_norm'] = False
        results['clips_to_max'] = False
    
    return results


def test_l1_regularization():
    """Test L1 regularization."""
    results = {}
    
    try:
        p = Parameter(np.array([1.0, -2.0, 3.0]))
        
        reg = l1_regularization([p], lambda_l1=0.1)
        
        expected = 0.1 * (1.0 + 2.0 + 3.0)
        results['correct'] = np.isclose(reg.data, expected)
    except Exception as e:
        results['correct'] = False
    
    return results


def test_l2_regularization():
    """Test L2 regularization."""
    results = {}
    
    try:
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        
        reg = l2_regularization([p], lambda_l2=0.1)
        
        expected = 0.05 * (1.0 + 4.0 + 9.0)
        results['correct'] = np.isclose(reg.data, expected)
    except Exception as e:
        results['correct'] = False
    
    return results


def test_sgd_weight_decay():
    """Test SGD with weight decay."""
    results = {}
    
    try:
        p = Parameter(np.array([10.0, 10.0, 10.0]))
        opt = SGD([p], lr=0.1, weight_decay=0.1)
        
        initial = p.data.copy()
        
        for _ in range(10):
            p.grad = np.zeros_like(p.data)
            opt.step()
        
        results['decays'] = np.all(np.abs(p.data) < np.abs(initial))
    except Exception as e:
        results['decays'] = False
    
    return results


def test_dropout():
    """Test Dropout layer."""
    results = {}
    
    try:
        np.random.seed(42)
        dropout = Dropout(p=0.5)
        
        x = Tensor(np.ones((100, 100)))
        
        dropout.train()
        out_train = dropout(x)
        
        zeros_train = np.sum(out_train.data == 0)
        results['drops_during_train'] = zeros_train > 1000
        
        mean_nonzero = np.mean(out_train.data[out_train.data != 0])
        results['scales_correctly'] = np.isclose(mean_nonzero, 2.0, rtol=0.2)
        
        dropout.eval()
        out_eval = dropout(x)
        results['passthrough_eval'] = np.allclose(out_eval.data, x.data)
    except Exception as e:
        results['drops_during_train'] = False
        results['scales_correctly'] = False
        results['passthrough_eval'] = False
    
    return results


def test_training_with_regularization():
    """Test training loop with regularization."""
    results = {}
    
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 16),
            ReLU(),
            Linear(16, 2)
        )
        
        opt = SGD(list(model.parameters()), lr=0.01)
        
        X = np.random.randn(20, 4)
        Y = np.random.randn(20, 2)
        
        history = train_with_regularization(
            model, opt, X, Y,
            epochs=50,
            lambda_l2=0.01,
            clip_norm=1.0
        )
        
        results['returns_history'] = len(history.get('losses', [])) > 0
        
        if results['returns_history']:
            results['loss_decreases'] = history['losses'][-1] < history['losses'][0]
        else:
            results['loss_decreases'] = False
    except Exception as e:
        results['returns_history'] = False
        results['loss_decreases'] = False
    
    return results


if __name__ == "__main__":
    print("Day 29: Gradient Clipping and Regularization")
    print("=" * 60)
    
    print("\nClip Grad Value:")
    cgv_results = test_clip_grad_value()
    for name, passed in cgv_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nClip Grad Norm:")
    cgn_results = test_clip_grad_norm()
    for name, passed in cgn_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nL1 Regularization:")
    l1_results = test_l1_regularization()
    for name, passed in l1_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nL2 Regularization:")
    l2_results = test_l2_regularization()
    for name, passed in l2_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSGD Weight Decay:")
    wd_results = test_sgd_weight_decay()
    for name, passed in wd_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nDropout:")
    do_results = test_dropout()
    for name, passed in do_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nTraining with Regularization:")
    tr_results = test_training_with_regularization()
    for name, passed in tr_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day29.py for comprehensive tests!")
