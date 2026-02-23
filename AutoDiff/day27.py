"""
Day 27: Momentum and Learning Rate Scheduling
=============================================
Estimated time: 3-4 hours
Prerequisites: Day 26 (SGD Optimizer)

Learning objectives:
- Implement SGD with momentum
- Understand Nesterov accelerated gradient
- Build learning rate schedulers
- Combine momentum with scheduling

Key concepts:
- Momentum: Accelerates SGD by accumulating velocity
  - Helps navigate ravines (high curvature directions)
  - Dampens oscillations
  - Can escape shallow local minima

- Nesterov Momentum: "Look-ahead" momentum
  - Evaluates gradient at projected position
  - Often converges faster than classical momentum

- Learning Rate Scheduling:
  - Reduce lr as training progresses
  - Common strategies: step decay, exponential, cosine

Mathematical background:

Classical Momentum:
    v_t = μ * v_{t-1} + ∇L(θ_t)
    θ_{t+1} = θ_t - lr * v_t
    
    Or with dampening (PyTorch style):
    v_t = μ * v_{t-1} + (1 - dampening) * ∇L(θ_t)
    θ_{t+1} = θ_t - lr * v_t

Nesterov Momentum:
    v_t = μ * v_{t-1} + ∇L(θ_t - lr * μ * v_{t-1})
    θ_{t+1} = θ_t - lr * v_t
    
    Simplified (PyTorch implementation):
    v_t = μ * v_{t-1} + g_t
    θ_{t+1} = θ_t - lr * (g_t + μ * v_t)

Learning Rate Schedules:
    Step: lr = lr_0 * γ^(epoch // step_size)
    Exponential: lr = lr_0 * γ^epoch
    Cosine: lr = lr_min + 0.5 * (lr_0 - lr_min) * (1 + cos(π * t / T))
"""

import numpy as np
from typing import List, Iterator, Dict, Any, Optional, Callable
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
# Exercise 1: SGD with Momentum
# ============================================================================

class SGDMomentum(Optimizer):
    """
    SGD with momentum.
    
    Update rule (classical momentum):
        v_t = momentum * v_{t-1} + gradient
        param = param - lr * v_t
    
    With dampening (PyTorch style):
        v_t = momentum * v_{t-1} + (1 - dampening) * gradient
        param = param - lr * v_t
    
    Args:
        params: Parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        dampening: Dampening for momentum (default: 0)
    
    Example:
        opt = SGDMomentum(model.parameters(), lr=0.01, momentum=0.9)
    """
    
    def __init__(self, params, lr: float, momentum: float = 0.0, 
                 dampening: float = 0.0):
        """
        Initialize SGD with momentum.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
            momentum: Momentum coefficient (0 = no momentum)
            dampening: Dampening for momentum (0 = no dampening)
        """
        # TODO: Initialize with momentum parameters
        # HINT:
        # if lr < 0.0:
        #     raise ValueError(f"Invalid learning rate: {lr}")
        # if momentum < 0.0:
        #     raise ValueError(f"Invalid momentum: {momentum}")
        # 
        # defaults = {
        #     'lr': lr,
        #     'momentum': momentum,
        #     'dampening': dampening
        # }
        # super().__init__(params, defaults)
        
        pass  # Replace with initialization
    
    def step(self):
        """
        Perform SGD step with momentum.
        
        For each parameter:
        1. Get or initialize velocity buffer
        2. Update velocity: v = momentum * v + (1 - dampening) * grad
        3. Update parameter: param -= lr * v
        """
        # TODO: Implement momentum SGD update
        # HINT:
        # for group in self.param_groups:
        #     momentum = group['momentum']
        #     dampening = group['dampening']
        #     lr = group['lr']
        #     
        #     for p in group['params']:
        #         if p.grad is None:
        #             continue
        #         
        #         # Get state for this parameter (use id(p) as key)
        #         param_id = id(p)
        #         if param_id not in self.state:
        #             self.state[param_id] = {}
        #         
        #         state = self.state[param_id]
        #         
        #         grad = p.grad
        #         
        #         if momentum != 0:
        #             # Get or initialize velocity buffer
        #             if 'velocity' not in state:
        #                 state['velocity'] = np.zeros_like(p.data)
        #             
        #             v = state['velocity']
        #             v[:] = momentum * v + (1 - dampening) * grad
        #             
        #             p.data -= lr * v
        #         else:
        #             p.data -= lr * grad
        
        pass  # Replace


# ============================================================================
# Exercise 2: Nesterov Accelerated Gradient
# ============================================================================

class SGDNesterov(Optimizer):
    """
    SGD with Nesterov accelerated gradient.
    
    Nesterov momentum "looks ahead" by evaluating the gradient at the
    projected position, leading to faster convergence.
    
    Update rule (PyTorch simplified version):
        v_t = momentum * v_{t-1} + gradient
        param = param - lr * (gradient + momentum * v_t)
    
    This is mathematically equivalent to Nesterov but more efficient
    as it doesn't require recomputing the gradient.
    
    Args:
        params: Parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (required for Nesterov)
    """
    
    def __init__(self, params, lr: float, momentum: float = 0.9):
        """Initialize SGD with Nesterov momentum."""
        # TODO: Initialize Nesterov optimizer
        # HINT:
        # if lr < 0.0:
        #     raise ValueError(f"Invalid learning rate: {lr}")
        # if momentum <= 0.0:
        #     raise ValueError("Nesterov requires momentum > 0")
        # 
        # defaults = {'lr': lr, 'momentum': momentum, 'nesterov': True}
        # super().__init__(params, defaults)
        
        pass  # Replace with initialization
    
    def step(self):
        """
        Perform Nesterov momentum update.
        
        For each parameter:
        1. Update velocity: v = momentum * v + grad
        2. Update param: param -= lr * (grad + momentum * v)
        """
        # TODO: Implement Nesterov update
        # HINT:
        # for group in self.param_groups:
        #     momentum = group['momentum']
        #     lr = group['lr']
        #     
        #     for p in group['params']:
        #         if p.grad is None:
        #             continue
        #         
        #         param_id = id(p)
        #         if param_id not in self.state:
        #             self.state[param_id] = {}
        #         state = self.state[param_id]
        #         
        #         grad = p.grad
        #         
        #         if 'velocity' not in state:
        #             state['velocity'] = np.zeros_like(p.data)
        #         
        #         v = state['velocity']
        #         v[:] = momentum * v + grad
        #         
        #         # Nesterov: use gradient + look-ahead term
        #         p.data -= lr * (grad + momentum * v)
        
        pass  # Replace


# ============================================================================
# Exercise 3: Learning Rate Scheduler Base Class
# ============================================================================

class LRScheduler:
    """
    Base class for learning rate schedulers.
    
    Schedulers adjust the learning rate during training, typically
    decreasing it to allow finer convergence as training progresses.
    
    Usage:
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        
        for epoch in range(100):
            train(...)
            scheduler.step()
    """
    
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            last_epoch: Index of last epoch (-1 means starting fresh)
        """
        # TODO: Initialize scheduler
        # HINT:
        # self.optimizer = optimizer
        # self.last_epoch = last_epoch
        # 
        # # Store base learning rates (initial lr values)
        # self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        # 
        # # Initialize at epoch 0 if starting fresh
        # if last_epoch == -1:
        #     self.last_epoch = 0
        #     self._step_count = 0
        
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = []  # Replace
        self._step_count = 0
    
    def get_lr(self) -> List[float]:
        """
        Compute learning rate for current epoch.
        
        Override in subclasses to implement specific schedules.
        
        Returns:
            List of learning rates (one per param group)
        """
        raise NotImplementedError
    
    def step(self, epoch: Optional[int] = None):
        """
        Update learning rates.
        
        Args:
            epoch: Epoch number (if None, increment automatically)
        """
        # TODO: Implement step
        # HINT:
        # if epoch is None:
        #     self.last_epoch += 1
        # else:
        #     self.last_epoch = epoch
        # 
        # self._step_count += 1
        # 
        # # Get new learning rates
        # new_lrs = self.get_lr()
        # 
        # # Update optimizer
        # for group, lr in zip(self.optimizer.param_groups, new_lrs):
        #     group['lr'] = lr
        
        pass  # Replace
    
    def get_last_lr(self) -> List[float]:
        """Return last computed learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


# ============================================================================
# Exercise 4: Step Learning Rate Scheduler
# ============================================================================

class StepLR(LRScheduler):
    """
    Decays learning rate by gamma every step_size epochs.
    
    lr = base_lr * gamma^(epoch // step_size)
    
    Args:
        optimizer: Optimizer to schedule
        step_size: Period of learning rate decay
        gamma: Multiplicative factor (default: 0.1)
    
    Example:
        # lr = 0.1 for epochs 0-29
        # lr = 0.01 for epochs 30-59
        # lr = 0.001 for epochs 60-89
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    """
    
    def __init__(self, optimizer: Optimizer, step_size: int, 
                 gamma: float = 0.1, last_epoch: int = -1):
        """
        Initialize StepLR scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            step_size: Decay lr every this many epochs
            gamma: Decay factor
            last_epoch: Index of last epoch
        """
        # TODO: Initialize StepLR
        # HINT:
        # self.step_size = step_size
        # self.gamma = gamma
        # super().__init__(optimizer, last_epoch)
        
        pass  # Replace with initialization
    
    def get_lr(self) -> List[float]:
        """
        Compute stepped learning rates.
        
        Returns:
            List of learning rates for each param group
        """
        # TODO: Implement step decay
        # HINT:
        # factor = self.gamma ** (self.last_epoch // self.step_size)
        # return [base_lr * factor for base_lr in self.base_lrs]
        
        return []  # Replace


# ============================================================================
# Exercise 5: Exponential Learning Rate Scheduler
# ============================================================================

class ExponentialLR(LRScheduler):
    """
    Decays learning rate exponentially every epoch.
    
    lr = base_lr * gamma^epoch
    
    Args:
        optimizer: Optimizer to schedule
        gamma: Multiplicative factor (e.g., 0.95)
    
    Example:
        # With gamma=0.95:
        # epoch 0: lr = 0.1
        # epoch 10: lr = 0.1 * 0.95^10 ≈ 0.0598
        # epoch 50: lr = 0.1 * 0.95^50 ≈ 0.0077
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    """
    
    def __init__(self, optimizer: Optimizer, gamma: float, 
                 last_epoch: int = -1):
        """
        Initialize ExponentialLR scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            gamma: Decay rate per epoch
            last_epoch: Index of last epoch
        """
        # TODO: Initialize ExponentialLR
        # HINT:
        # self.gamma = gamma
        # super().__init__(optimizer, last_epoch)
        
        pass  # Replace with initialization
    
    def get_lr(self) -> List[float]:
        """
        Compute exponentially decayed learning rates.
        
        Returns:
            List of learning rates for each param group
        """
        # TODO: Implement exponential decay
        # HINT:
        # factor = self.gamma ** self.last_epoch
        # return [base_lr * factor for base_lr in self.base_lrs]
        
        return []  # Replace


# ============================================================================
# Exercise 6: Cosine Annealing Scheduler
# ============================================================================

class CosineAnnealingLR(LRScheduler):
    """
    Cosine annealing learning rate scheduler.
    
    Smoothly decreases lr from base_lr to eta_min following a cosine curve.
    
    lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(π * epoch / T_max))
    
    Args:
        optimizer: Optimizer to schedule
        T_max: Maximum number of epochs
        eta_min: Minimum learning rate (default: 0)
    
    Properties:
        - Smooth decay (no sudden drops)
        - Starts slow, accelerates, then slows again
        - Reaches eta_min at epoch T_max
    
    Example:
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    """
    
    def __init__(self, optimizer: Optimizer, T_max: int, 
                 eta_min: float = 0.0, last_epoch: int = -1):
        """
        Initialize CosineAnnealingLR scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            T_max: Maximum epochs (period of cosine)
            eta_min: Minimum learning rate
            last_epoch: Index of last epoch
        """
        # TODO: Initialize CosineAnnealingLR
        # HINT:
        # self.T_max = T_max
        # self.eta_min = eta_min
        # super().__init__(optimizer, last_epoch)
        
        pass  # Replace with initialization
    
    def get_lr(self) -> List[float]:
        """
        Compute cosine-annealed learning rates.
        
        Returns:
            List of learning rates for each param group
        """
        # TODO: Implement cosine annealing
        # HINT:
        # cosine_factor = (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        # return [
        #     self.eta_min + (base_lr - self.eta_min) * cosine_factor
        #     for base_lr in self.base_lrs
        # ]
        
        return []  # Replace


# ============================================================================
# Exercise 7: Warmup Scheduler
# ============================================================================

class WarmupLR(LRScheduler):
    """
    Linear warmup followed by constant learning rate.
    
    During warmup:
        lr = base_lr * (epoch / warmup_epochs)
    
    After warmup:
        lr = base_lr
    
    Warmup helps stabilize training in the early stages, especially
    with large batch sizes or when using momentum.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_epochs: Number of warmup epochs
    
    Example:
        scheduler = WarmupLR(optimizer, warmup_epochs=5)
        # epoch 0: lr = base_lr * 0/5 = 0
        # epoch 1: lr = base_lr * 1/5 = 0.2 * base_lr
        # epoch 5+: lr = base_lr
    """
    
    def __init__(self, optimizer: Optimizer, warmup_epochs: int,
                 last_epoch: int = -1):
        """
        Initialize WarmupLR scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of epochs to warm up
            last_epoch: Index of last epoch
        """
        # TODO: Initialize WarmupLR
        # HINT:
        # self.warmup_epochs = warmup_epochs
        # super().__init__(optimizer, last_epoch)
        
        pass  # Replace with initialization
    
    def get_lr(self) -> List[float]:
        """
        Compute warmup learning rates.
        
        Returns:
            List of learning rates for each param group
        """
        # TODO: Implement linear warmup
        # HINT:
        # if self.last_epoch < self.warmup_epochs:
        #     # Linear warmup: scale from 0 to base_lr
        #     warmup_factor = self.last_epoch / self.warmup_epochs
        #     return [base_lr * warmup_factor for base_lr in self.base_lrs]
        # else:
        #     # After warmup: use base learning rate
        #     return list(self.base_lrs)
        
        return []  # Replace


# ============================================================================
# Helper Functions
# ============================================================================

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error loss."""
    diff = pred - target
    return (diff * diff).mean()


def train_with_scheduler(model: Module, optimizer: Optimizer,
                         scheduler: LRScheduler, X: np.ndarray, 
                         Y: np.ndarray, epochs: int) -> Dict[str, List]:
    """
    Train with learning rate scheduling.
    
    Args:
        model: Neural network
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        X: Training inputs
        Y: Training targets
        epochs: Number of epochs
    
    Returns:
        Dict with 'losses' and 'learning_rates' lists
    """
    history = {'losses': [], 'learning_rates': []}
    
    for epoch in range(epochs):
        x = Tensor(X)
        y = Tensor(Y)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        
        history['losses'].append(float(loss.data))
        history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        scheduler.step()
    
    return history


# ============================================================================
# Test Functions
# ============================================================================

def test_sgd_momentum():
    """Test SGD with momentum."""
    results = {}
    
    try:
        np.random.seed(42)
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        opt = SGDMomentum([p], lr=0.1, momentum=0.9)
        
        results['creates'] = opt is not None
        
        for _ in range(5):
            p.grad = np.array([0.1, 0.1, 0.1])
            opt.step()
        
        results['updates'] = not np.allclose(p.data, [1.0, 2.0, 3.0])
        
        has_velocity = any('velocity' in s for s in opt.state.values())
        results['has_velocity'] = has_velocity
    except Exception as e:
        results['creates'] = False
        results['updates'] = False
        results['has_velocity'] = False
    
    return results


def test_nesterov():
    """Test Nesterov momentum."""
    results = {}
    
    try:
        np.random.seed(42)
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        opt = SGDNesterov([p], lr=0.1, momentum=0.9)
        
        results['creates'] = opt is not None
        
        for _ in range(5):
            p.grad = np.array([0.1, 0.1, 0.1])
            opt.step()
        
        results['updates'] = not np.allclose(p.data, [1.0, 2.0, 3.0])
    except Exception as e:
        results['creates'] = False
        results['updates'] = False
    
    return results


def test_step_lr():
    """Test StepLR scheduler."""
    results = {}
    
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = StepLR(opt, step_size=5, gamma=0.1)
        
        results['creates'] = scheduler is not None
        
        lrs = []
        for i in range(15):
            lrs.append(opt.param_groups[0]['lr'])
            scheduler.step()
        
        results['decays'] = lrs[0] > lrs[5] > lrs[10]
    except Exception as e:
        results['creates'] = False
        results['decays'] = False
    
    return results


def test_exponential_lr():
    """Test ExponentialLR scheduler."""
    results = {}
    
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = ExponentialLR(opt, gamma=0.9)
        
        results['creates'] = scheduler is not None
        
        lrs = []
        for i in range(10):
            lrs.append(opt.param_groups[0]['lr'])
            scheduler.step()
        
        expected = [0.1 * (0.9 ** i) for i in range(10)]
        results['correct'] = all(np.isclose(a, b, rtol=1e-5) for a, b in zip(lrs, expected))
    except Exception as e:
        results['creates'] = False
        results['correct'] = False
    
    return results


def test_cosine_lr():
    """Test CosineAnnealingLR scheduler."""
    results = {}
    
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0.001)
        
        results['creates'] = scheduler is not None
        
        lrs = []
        for i in range(100):
            lrs.append(opt.param_groups[0]['lr'])
            scheduler.step()
        
        results['starts_high'] = lrs[0] > 0.09
        results['ends_low'] = lrs[-1] < 0.01
        results['smooth'] = all(abs(lrs[i] - lrs[i+1]) < 0.01 for i in range(len(lrs)-1))
    except Exception as e:
        results['creates'] = False
        results['starts_high'] = False
        results['ends_low'] = False
        results['smooth'] = False
    
    return results


def test_warmup_lr():
    """Test WarmupLR scheduler."""
    results = {}
    
    try:
        params = [Parameter(np.random.randn(3))]
        opt = SGDMomentum(params, lr=0.1, momentum=0.9)
        scheduler = WarmupLR(opt, warmup_epochs=5)
        
        results['creates'] = scheduler is not None
        
        lrs = []
        for i in range(10):
            lrs.append(opt.param_groups[0]['lr'])
            scheduler.step()
        
        results['warmup_increases'] = lrs[1] > lrs[0]
        results['reaches_target'] = np.isclose(lrs[5], 0.1, rtol=0.1)
    except Exception as e:
        results['creates'] = False
        results['warmup_increases'] = False
        results['reaches_target'] = False
    
    return results


if __name__ == "__main__":
    print("Day 27: Momentum and Learning Rate Scheduling")
    print("=" * 60)
    
    print("\nSGD Momentum:")
    mom_results = test_sgd_momentum()
    for name, passed in mom_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nNesterov:")
    nes_results = test_nesterov()
    for name, passed in nes_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nStepLR:")
    step_results = test_step_lr()
    for name, passed in step_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nExponentialLR:")
    exp_results = test_exponential_lr()
    for name, passed in exp_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nCosineAnnealingLR:")
    cos_results = test_cosine_lr()
    for name, passed in cos_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nWarmupLR:")
    warm_results = test_warmup_lr()
    for name, passed in warm_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day27.py for comprehensive tests!")
