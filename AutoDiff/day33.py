"""
Day 33: Batch Normalization
===========================
Estimated time: 4-5 hours
Prerequisites: Days 21-32 (Module system, convolution, gradients)

Learning objectives:
- Understand why batch normalization helps training
- Implement BatchNorm forward pass for training and inference
- Derive and implement the backward pass
- Handle running mean/variance for inference

Key concepts:
- Batch Normalization: Normalizes layer inputs to have zero mean and unit variance
  - Reduces internal covariate shift
  - Allows higher learning rates
  - Acts as regularization

- Training mode:
  - Compute mean and variance over batch dimension
  - Update running statistics using exponential moving average
  
- Inference mode:
  - Use stored running mean and variance
  - No batch statistics needed (works with single samples)

Mathematical background:
- Forward pass (training):
  μ_B = (1/m) Σ x_i           (batch mean)
  σ²_B = (1/m) Σ (x_i - μ_B)² (batch variance)
  x̂_i = (x_i - μ_B) / √(σ²_B + ε)  (normalize)
  y_i = γ * x̂_i + β          (scale and shift)

- Backward pass (gradients):
  dγ = Σ (dy_i * x̂_i)
  dβ = Σ dy_i
  dx̂ = dy * γ
  
  dσ² = Σ dx̂_i * (x_i - μ_B) * (-0.5) * (σ²_B + ε)^(-3/2)
  dμ = Σ dx̂_i * (-1/√(σ²_B + ε)) + dσ² * (-2/m) * Σ(x_i - μ_B)
  dx_i = dx̂_i / √(σ²_B + ε) + dσ² * 2(x_i - μ_B)/m + dμ/m

- Running statistics update:
  running_mean = momentum * running_mean + (1 - momentum) * batch_mean
  running_var = momentum * running_var + (1 - momentum) * batch_var
"""

import numpy as np
from typing import Tuple, Optional, List


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
                self.grad += np.broadcast_to(grad, self.shape).copy()
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'mean')
        count = self.data.size if axis is None else np.prod([self.data.shape[i] for i in (axis if isinstance(axis, tuple) else (axis,))])
        
        def _backward():
            if axis is None:
                self.grad += np.full(self.shape, out.grad / count)
            else:
                grad = out.grad / count
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.broadcast_to(grad, self.shape).copy()
        out._backward = _backward
        return out
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)


def _unbroadcast(grad, original_shape):
    """Reverse broadcasting by summing over broadcast dimensions."""
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
        if orig_dim == 1 and grad_dim > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Module:
    """Base class for neural network modules."""
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, x):
        raise NotImplementedError
    
    def parameters(self):
        return []
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    
    def train(self):
        self._training = True
    
    def eval(self):
        self._training = False


# ============================================================================
# Exercise 1: BatchNorm1d Forward Pass
# ============================================================================

def batchnorm1d_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                        running_mean: np.ndarray, running_var: np.ndarray,
                        training: bool, momentum: float = 0.1,
                        eps: float = 1e-5) -> Tuple[np.ndarray, dict]:
    """
    Compute 1D batch normalization forward pass.
    
    Args:
        x: Input of shape (N, C) or (N, C, L)
        gamma: Scale parameter of shape (C,)
        beta: Shift parameter of shape (C,)
        running_mean: Running mean of shape (C,)
        running_var: Running variance of shape (C,)
        training: Whether in training mode
        momentum: Momentum for running stats update
        eps: Small constant for numerical stability
    
    Returns:
        out: Normalized output same shape as x
        cache: Dictionary with values needed for backward pass
    
    Cache should contain:
        - x_norm: Normalized input
        - mean: Batch mean (or running mean)
        - var: Batch variance (or running var)
        - std: Standard deviation
        - gamma: Scale parameter
        - x: Original input
    """
    # TODO: Implement batch normalization forward
    # HINT:
    # cache = {}
    # 
    # if training:
    #     # Compute batch statistics
    #     if x.ndim == 2:
    #         mean = np.mean(x, axis=0)
    #         var = np.var(x, axis=0)
    #     else:  # (N, C, L)
    #         mean = np.mean(x, axis=(0, 2))
    #         var = np.var(x, axis=(0, 2))
    #     
    #     # Update running statistics
    #     running_mean[:] = momentum * running_mean + (1 - momentum) * mean
    #     running_var[:] = momentum * running_var + (1 - momentum) * var
    # else:
    #     mean = running_mean
    #     var = running_var
    # 
    # std = np.sqrt(var + eps)
    # 
    # # Normalize
    # if x.ndim == 2:
    #     x_norm = (x - mean) / std
    #     out = gamma * x_norm + beta
    # else:
    #     x_norm = (x - mean[None, :, None]) / std[None, :, None]
    #     out = gamma[None, :, None] * x_norm + beta[None, :, None]
    # 
    # cache['x_norm'] = x_norm
    # cache['mean'] = mean
    # cache['var'] = var
    # cache['std'] = std
    # cache['gamma'] = gamma
    # cache['x'] = x
    # 
    # return out, cache
    
    pass  # Replace with implementation


# ============================================================================
# Exercise 2: BatchNorm1d Backward Pass
# ============================================================================

def batchnorm1d_backward(dy: np.ndarray, cache: dict, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 1D batch normalization backward pass.
    
    Args:
        dy: Upstream gradient, same shape as forward output
        cache: Cache from forward pass
        eps: Epsilon used in forward
    
    Returns:
        dx: Gradient w.r.t. input
        dgamma: Gradient w.r.t. scale parameter
        dbeta: Gradient w.r.t. shift parameter
    """
    # TODO: Implement batch normalization backward
    # HINT:
    # x_norm = cache['x_norm']
    # std = cache['std']
    # gamma = cache['gamma']
    # x = cache['x']
    # mean = cache['mean']
    # 
    # if x.ndim == 2:
    #     N = x.shape[0]
    #     
    #     # Gradients for gamma and beta
    #     dgamma = np.sum(dy * x_norm, axis=0)
    #     dbeta = np.sum(dy, axis=0)
    #     
    #     # Gradient for normalized input
    #     dx_norm = dy * gamma
    #     
    #     # Gradient for input (using the batch norm gradient formula)
    #     dx = (1.0 / N) * (1.0 / std) * (
    #         N * dx_norm 
    #         - np.sum(dx_norm, axis=0) 
    #         - x_norm * np.sum(dx_norm * x_norm, axis=0)
    #     )
    # else:  # (N, C, L)
    #     N, C, L = x.shape
    #     m = N * L  # Total samples per channel
    #     
    #     dgamma = np.sum(dy * x_norm, axis=(0, 2))
    #     dbeta = np.sum(dy, axis=(0, 2))
    #     
    #     dx_norm = dy * gamma[None, :, None]
    #     
    #     dx = (1.0 / m) * (1.0 / std[None, :, None]) * (
    #         m * dx_norm
    #         - np.sum(dx_norm, axis=(0, 2), keepdims=True).transpose(1, 0, 2)
    #         - x_norm * np.sum(dx_norm * x_norm, axis=(0, 2), keepdims=True).transpose(1, 0, 2)
    #     )
    # 
    # return dx, dgamma, dbeta
    
    pass  # Replace with implementation


# ============================================================================
# Exercise 3: BatchNorm1d Module
# ============================================================================

class BatchNorm1d(Module):
    """
    1D Batch Normalization layer.
    
    Normalizes inputs over the batch dimension for 2D inputs (N, C)
    or over batch and length for 3D inputs (N, C, L).
    
    Args:
        num_features: Number of features/channels
        eps: Small constant for numerical stability
        momentum: Momentum for running stats (default: 0.1)
        affine: Whether to learn scale (gamma) and shift (beta)
    
    Example:
        bn = BatchNorm1d(64)
        x = Tensor(np.random.randn(32, 64))
        y = bn(x)
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, 
                 momentum: float = 0.1, affine: bool = True):
        """Initialize BatchNorm1d layer."""
        # TODO: Initialize parameters and buffers
        # HINT:
        # self.num_features = num_features
        # self.eps = eps
        # self.momentum = momentum
        # self.affine = affine
        # self._training = True
        # 
        # # Learnable parameters
        # if affine:
        #     self.gamma = Tensor(np.ones(num_features))
        #     self.beta = Tensor(np.zeros(num_features))
        # else:
        #     self.gamma = None
        #     self.beta = None
        # 
        # # Running statistics (not parameters, but state)
        # self.running_mean = np.zeros(num_features)
        # self.running_var = np.ones(num_features)
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self._training = True
        
        self.gamma = None  # Replace with Tensor
        self.beta = None   # Replace with Tensor
        self.running_mean = None  # Replace with numpy array
        self.running_var = None   # Replace with numpy array
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply batch normalization."""
        # TODO: Implement forward with gradient tracking
        # HINT:
        # gamma_data = self.gamma.data if self.gamma is not None else np.ones(self.num_features)
        # beta_data = self.beta.data if self.beta is not None else np.zeros(self.num_features)
        # 
        # result = batchnorm1d_forward(
        #     x.data, gamma_data, beta_data,
        #     self.running_mean, self.running_var,
        #     self._training, self.momentum, self.eps
        # )
        # 
        # if result is None:
        #     return None
        # 
        # out_data, cache = result
        # 
        # if self.affine:
        #     out = Tensor(out_data, (x, self.gamma, self.beta), 'batchnorm1d')
        # else:
        #     out = Tensor(out_data, (x,), 'batchnorm1d')
        # 
        # def _backward():
        #     backward_result = batchnorm1d_backward(out.grad, cache, self.eps)
        #     if backward_result is None:
        #         return
        #     dx, dgamma, dbeta = backward_result
        #     x.grad += dx
        #     if self.affine:
        #         self.gamma.grad += dgamma
        #         self.beta.grad += dbeta
        # 
        # out._backward = _backward
        # return out
        
        return None  # Replace with implementation
    
    def parameters(self) -> List[Tensor]:
        """Return learnable parameters."""
        if self.affine and self.gamma is not None and self.beta is not None:
            return [self.gamma, self.beta]
        return []
    
    def train(self):
        self._training = True
    
    def eval(self):
        self._training = False
    
    def __repr__(self):
        return f"BatchNorm1d({self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine})"


# ============================================================================
# Exercise 4: BatchNorm2d Forward Pass
# ============================================================================

def batchnorm2d_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                        running_mean: np.ndarray, running_var: np.ndarray,
                        training: bool, momentum: float = 0.1,
                        eps: float = 1e-5) -> Tuple[np.ndarray, dict]:
    """
    Compute 2D batch normalization forward pass.
    
    For convolutional layers, normalizes over (N, H, W) dimensions.
    
    Args:
        x: Input of shape (N, C, H, W)
        gamma: Scale parameter of shape (C,)
        beta: Shift parameter of shape (C,)
        running_mean: Running mean of shape (C,)
        running_var: Running variance of shape (C,)
        training: Whether in training mode
        momentum: Momentum for running stats
        eps: Numerical stability constant
    
    Returns:
        out: Normalized output of shape (N, C, H, W)
        cache: Values needed for backward
    """
    # TODO: Implement 2D batch normalization forward
    # HINT:
    # cache = {}
    # N, C, H, W = x.shape
    # 
    # if training:
    #     # Compute mean and var over (N, H, W) for each channel
    #     mean = np.mean(x, axis=(0, 2, 3))  # (C,)
    #     var = np.var(x, axis=(0, 2, 3))    # (C,)
    #     
    #     # Update running statistics
    #     running_mean[:] = momentum * running_mean + (1 - momentum) * mean
    #     running_var[:] = momentum * running_var + (1 - momentum) * var
    # else:
    #     mean = running_mean
    #     var = running_var
    # 
    # std = np.sqrt(var + eps)
    # 
    # # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
    # x_norm = (x - mean[None, :, None, None]) / std[None, :, None, None]
    # out = gamma[None, :, None, None] * x_norm + beta[None, :, None, None]
    # 
    # cache['x_norm'] = x_norm
    # cache['mean'] = mean
    # cache['var'] = var
    # cache['std'] = std
    # cache['gamma'] = gamma
    # cache['x'] = x
    # 
    # return out, cache
    
    pass  # Replace with implementation


# ============================================================================
# Exercise 5: BatchNorm2d Backward Pass
# ============================================================================

def batchnorm2d_backward(dy: np.ndarray, cache: dict, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 2D batch normalization backward pass.
    
    Args:
        dy: Upstream gradient of shape (N, C, H, W)
        cache: Cache from forward pass
        eps: Epsilon used in forward
    
    Returns:
        dx: Gradient w.r.t. input (N, C, H, W)
        dgamma: Gradient w.r.t. scale (C,)
        dbeta: Gradient w.r.t. shift (C,)
    """
    # TODO: Implement 2D batch normalization backward
    # HINT:
    # x_norm = cache['x_norm']
    # std = cache['std']
    # gamma = cache['gamma']
    # x = cache['x']
    # 
    # N, C, H, W = x.shape
    # m = N * H * W  # Total samples per channel
    # 
    # # Gradients for gamma and beta
    # dgamma = np.sum(dy * x_norm, axis=(0, 2, 3))
    # dbeta = np.sum(dy, axis=(0, 2, 3))
    # 
    # # Gradient for normalized input
    # dx_norm = dy * gamma[None, :, None, None]
    # 
    # # Gradient for input
    # dx = (1.0 / m) * (1.0 / std[None, :, None, None]) * (
    #     m * dx_norm
    #     - np.sum(dx_norm, axis=(0, 2, 3), keepdims=True).transpose(1, 0, 2, 3)
    #     - x_norm * np.sum(dx_norm * x_norm, axis=(0, 2, 3), keepdims=True).transpose(1, 0, 2, 3)
    # )
    # 
    # return dx, dgamma, dbeta
    
    pass  # Replace with implementation


# ============================================================================
# Exercise 6: BatchNorm2d Module
# ============================================================================

class BatchNorm2d(Module):
    """
    2D Batch Normalization layer for convolutional networks.
    
    Normalizes over (N, H, W) dimensions, keeping channels independent.
    
    Args:
        num_features: Number of channels (C)
        eps: Numerical stability constant
        momentum: Momentum for running stats
        affine: Whether to learn scale and shift
    
    Example:
        bn = BatchNorm2d(64)
        x = Tensor(np.random.randn(32, 64, 8, 8))
        y = bn(x)  # shape: (32, 64, 8, 8)
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True):
        """Initialize BatchNorm2d layer."""
        # TODO: Initialize parameters and buffers
        # HINT: Same structure as BatchNorm1d
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self._training = True
        
        self.gamma = None  # Replace with Tensor
        self.beta = None   # Replace with Tensor
        self.running_mean = None  # Replace with numpy array
        self.running_var = None   # Replace with numpy array
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply 2D batch normalization."""
        # TODO: Implement forward with gradient tracking
        # HINT: Similar to BatchNorm1d but use batchnorm2d functions
        
        return None  # Replace with implementation
    
    def parameters(self) -> List[Tensor]:
        """Return learnable parameters."""
        if self.affine and self.gamma is not None and self.beta is not None:
            return [self.gamma, self.beta]
        return []
    
    def train(self):
        self._training = True
    
    def eval(self):
        self._training = False
    
    def __repr__(self):
        return f"BatchNorm2d({self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine})"


# ============================================================================
# Exercise 7: Layer Normalization (Bonus)
# ============================================================================

class LayerNorm(Module):
    """
    Layer Normalization.
    
    Unlike BatchNorm, normalizes over feature dimensions (not batch).
    This is batch-size independent and commonly used in transformers.
    
    For input shape (N, C, ...), normalizes over (C, ...) for each sample.
    
    Args:
        normalized_shape: Shape to normalize over (e.g., [C] or [C, H, W])
        eps: Numerical stability constant
    """
    
    def __init__(self, normalized_shape, eps: float = 1e-5):
        """Initialize LayerNorm."""
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        
        # Learnable parameters
        self.gamma = Tensor(np.ones(normalized_shape))
        self.beta = Tensor(np.zeros(normalized_shape))
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization.
        
        Normalizes over the last len(normalized_shape) dimensions.
        """
        # TODO: Implement layer normalization
        # HINT:
        # # Determine axes to normalize over
        # ndim = len(self.normalized_shape)
        # axes = tuple(range(-ndim, 0))
        # 
        # # Compute statistics
        # mean = np.mean(x.data, axis=axes, keepdims=True)
        # var = np.var(x.data, axis=axes, keepdims=True)
        # std = np.sqrt(var + self.eps)
        # 
        # # Normalize
        # x_norm = (x.data - mean) / std
        # out_data = self.gamma.data * x_norm + self.beta.data
        # 
        # out = Tensor(out_data, (x, self.gamma, self.beta), 'layernorm')
        # 
        # def _backward():
        #     # Simplified backward (full derivation is complex)
        #     dgamma = np.sum(out.grad * x_norm, axis=tuple(range(x.ndim - ndim)))
        #     dbeta = np.sum(out.grad, axis=tuple(range(x.ndim - ndim)))
        #     
        #     # Input gradient
        #     n = np.prod(self.normalized_shape)
        #     dx_norm = out.grad * self.gamma.data
        #     dx = (1.0 / n) * (1.0 / std) * (
        #         n * dx_norm
        #         - np.sum(dx_norm, axis=axes, keepdims=True)
        #         - x_norm * np.sum(dx_norm * x_norm, axis=axes, keepdims=True)
        #     )
        #     
        #     x.grad += dx
        #     self.gamma.grad += dgamma
        #     self.beta.grad += dbeta
        # 
        # out._backward = _backward
        # return out
        
        return None  # Replace with implementation
    
    def parameters(self) -> List[Tensor]:
        return [self.gamma, self.beta]
    
    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"


# ============================================================================
# Test Functions
# ============================================================================

def test_batchnorm1d_forward():
    """Test BatchNorm1d forward pass."""
    results = {}
    
    try:
        np.random.seed(42)
        x = np.random.randn(32, 64)
        gamma = np.ones(64)
        beta = np.zeros(64)
        running_mean = np.zeros(64)
        running_var = np.ones(64)
        
        result = batchnorm1d_forward(x, gamma, beta, running_mean, running_var,
                                      training=True)
        
        if result is not None:
            out, cache = result
            results['shape'] = out.shape == x.shape
            results['normalized'] = np.allclose(np.mean(out, axis=0), 0, atol=1e-6) and \
                                   np.allclose(np.var(out, axis=0), 1, atol=1e-5)
        else:
            results['shape'] = False
            results['normalized'] = False
    except Exception as e:
        results['shape'] = False
        results['normalized'] = False
    
    return results


def test_batchnorm1d_backward():
    """Test BatchNorm1d backward pass."""
    results = {}
    
    try:
        np.random.seed(42)
        x = np.random.randn(8, 16)
        gamma = np.ones(16)
        beta = np.zeros(16)
        running_mean = np.zeros(16)
        running_var = np.ones(16)
        
        forward_result = batchnorm1d_forward(x, gamma, beta, running_mean, running_var,
                                              training=True)
        
        if forward_result is not None:
            out, cache = forward_result
            dy = np.random.randn(*out.shape)
            
            backward_result = batchnorm1d_backward(dy, cache)
            
            if backward_result is not None:
                dx, dgamma, dbeta = backward_result
                results['dx_shape'] = dx.shape == x.shape
                results['dgamma_shape'] = dgamma.shape == gamma.shape
                results['dbeta_shape'] = dbeta.shape == beta.shape
            else:
                results['dx_shape'] = False
                results['dgamma_shape'] = False
                results['dbeta_shape'] = False
        else:
            results['dx_shape'] = False
            results['dgamma_shape'] = False
            results['dbeta_shape'] = False
    except Exception as e:
        results['dx_shape'] = False
        results['dgamma_shape'] = False
        results['dbeta_shape'] = False
    
    return results


def test_batchnorm1d_module():
    """Test BatchNorm1d module."""
    results = {}
    
    try:
        np.random.seed(42)
        bn = BatchNorm1d(32)
        
        if bn.gamma is not None and bn.running_mean is not None:
            results['init'] = True
            
            x = Tensor(np.random.randn(16, 32))
            bn.train()
            y = bn(x)
            
            if y is not None:
                results['forward'] = y.shape == x.shape
                
                loss = y.sum()
                loss.backward()
                
                results['backward'] = bn.gamma.grad is not None and np.any(bn.gamma.grad != 0)
            else:
                results['forward'] = False
                results['backward'] = False
        else:
            results['init'] = False
            results['forward'] = False
            results['backward'] = False
    except Exception as e:
        results['init'] = False
        results['forward'] = False
        results['backward'] = False
    
    return results


def test_batchnorm2d_forward():
    """Test BatchNorm2d forward pass."""
    results = {}
    
    try:
        np.random.seed(42)
        x = np.random.randn(8, 16, 4, 4)
        gamma = np.ones(16)
        beta = np.zeros(16)
        running_mean = np.zeros(16)
        running_var = np.ones(16)
        
        result = batchnorm2d_forward(x, gamma, beta, running_mean, running_var,
                                      training=True)
        
        if result is not None:
            out, cache = result
            results['shape'] = out.shape == x.shape
            
            mean_per_channel = np.mean(out, axis=(0, 2, 3))
            var_per_channel = np.var(out, axis=(0, 2, 3))
            results['normalized'] = np.allclose(mean_per_channel, 0, atol=1e-6) and \
                                   np.allclose(var_per_channel, 1, atol=1e-5)
        else:
            results['shape'] = False
            results['normalized'] = False
    except Exception as e:
        results['shape'] = False
        results['normalized'] = False
    
    return results


def test_batchnorm2d_module():
    """Test BatchNorm2d module."""
    results = {}
    
    try:
        np.random.seed(42)
        bn = BatchNorm2d(16)
        
        if bn.gamma is not None and bn.running_mean is not None:
            results['init'] = True
            
            x = Tensor(np.random.randn(4, 16, 8, 8))
            bn.train()
            y = bn(x)
            
            if y is not None:
                results['forward'] = y.shape == x.shape
                
                loss = y.sum()
                loss.backward()
                
                results['backward'] = bn.gamma.grad is not None and np.any(bn.gamma.grad != 0)
            else:
                results['forward'] = False
                results['backward'] = False
        else:
            results['init'] = False
            results['forward'] = False
            results['backward'] = False
    except Exception as e:
        results['init'] = False
        results['forward'] = False
        results['backward'] = False
    
    return results


def test_running_stats():
    """Test that running statistics are updated during training."""
    results = {}
    
    try:
        np.random.seed(42)
        bn = BatchNorm1d(16, momentum=0.1)
        
        if bn.running_mean is None or bn.running_var is None:
            return {'running_mean': False, 'running_var': False}
        
        initial_mean = bn.running_mean.copy()
        initial_var = bn.running_var.copy()
        
        bn.train()
        for _ in range(5):
            x = Tensor(np.random.randn(32, 16))
            y = bn(x)
        
        results['running_mean'] = not np.allclose(bn.running_mean, initial_mean)
        results['running_var'] = not np.allclose(bn.running_var, initial_var)
    except Exception as e:
        results['running_mean'] = False
        results['running_var'] = False
    
    return results


def test_eval_mode():
    """Test that eval mode uses running statistics."""
    results = {}
    
    try:
        np.random.seed(42)
        bn = BatchNorm1d(16)
        
        if bn.running_mean is None:
            return {'uses_running_stats': False}
        
        bn.train()
        for _ in range(10):
            x = Tensor(np.random.randn(32, 16))
            y = bn(x)
        
        bn.eval()
        
        x_test = Tensor(np.random.randn(1, 16))
        y_test = bn(x_test)
        
        if y_test is not None:
            results['uses_running_stats'] = y_test.shape == x_test.shape
        else:
            results['uses_running_stats'] = False
    except Exception as e:
        results['uses_running_stats'] = False
    
    return results


def test_against_pytorch():
    """Test BatchNorm against PyTorch."""
    results = {}
    
    try:
        import torch
        import torch.nn as nn
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        our_bn = BatchNorm2d(16)
        
        if our_bn.gamma is None:
            return {'forward': False, 'backward': False}
        
        torch_bn = nn.BatchNorm2d(16)
        torch_bn.weight.data = torch.tensor(our_bn.gamma.data.copy())
        torch_bn.bias.data = torch.tensor(our_bn.beta.data.copy())
        
        x_np = np.random.randn(4, 16, 8, 8)
        
        our_bn.train()
        torch_bn.train()
        
        our_x = Tensor(x_np.copy())
        our_y = our_bn(our_x)
        
        torch_x = torch.tensor(x_np, requires_grad=True, dtype=torch.float64)
        torch_bn = torch_bn.double()
        torch_y = torch_bn(torch_x)
        
        if our_y is not None:
            results['forward'] = np.allclose(our_y.data, torch_y.detach().numpy(), rtol=1e-4, atol=1e-6)
        else:
            results['forward'] = False
            
    except ImportError:
        results['forward'] = True
    except Exception as e:
        results['forward'] = False
    
    return results


if __name__ == "__main__":
    print("Day 33: Batch Normalization")
    print("=" * 60)
    
    print("\nBatchNorm1d Forward:")
    bn1d_fwd = test_batchnorm1d_forward()
    for name, passed in bn1d_fwd.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nBatchNorm1d Backward:")
    bn1d_bwd = test_batchnorm1d_backward()
    for name, passed in bn1d_bwd.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nBatchNorm1d Module:")
    bn1d_mod = test_batchnorm1d_module()
    for name, passed in bn1d_mod.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nBatchNorm2d Forward:")
    bn2d_fwd = test_batchnorm2d_forward()
    for name, passed in bn2d_fwd.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nBatchNorm2d Module:")
    bn2d_mod = test_batchnorm2d_module()
    for name, passed in bn2d_mod.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRunning Statistics:")
    running_results = test_running_stats()
    for name, passed in running_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nEval Mode:")
    eval_results = test_eval_mode()
    for name, passed in eval_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nPyTorch Comparison:")
    pytorch_results = test_against_pytorch()
    for name, passed in pytorch_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day33.py for comprehensive tests!")
