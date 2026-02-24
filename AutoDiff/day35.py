"""
Day 35: Final Project - Complete Autodiff Library
=================================================
Estimated time: 5-6 hours
Prerequisites: Days 1-34 (Everything!)

Congratulations! You've made it to the final day!

Learning objectives:
- Assemble all components into a complete, usable autodiff library
- Train a CNN on MNIST-like synthetic data
- Understand the full pipeline from data to trained model
- Appreciate what PyTorch does under the hood

This file brings together everything you've built:
- Tensor with automatic differentiation (Days 1-10)
- Operations: add, mul, matmul, reshape, etc. (Days 11-15)
- Loss functions: MSE, Cross-Entropy (Days 16-20)
- Layers: Linear, Conv2d, BatchNorm, Pooling (Days 21-33)
- Optimizers: SGD with momentum (Day 34)
- Complete CNN architectures (Day 34)

Your task is to:
1. Complete the MiniTorch library by filling in any missing pieces
2. Create a data loader for synthetic MNIST-like data
3. Build and train a CNN to classify digits
4. Achieve reasonable accuracy on the test set

Mathematical background:
The complete forward-backward cycle:
1. Forward: x -> model -> logits -> loss
2. Backward: dloss/dlogits -> ... -> dloss/dweights
3. Update: weights = weights - lr * dloss/dweights

This is the essence of deep learning - you've built it from scratch!
"""

import numpy as np
from typing import Tuple, Optional, List, Iterator, Dict, Any
from collections import OrderedDict
import time


# ============================================================================
# Part 1: The Complete Tensor Class
# ============================================================================

class Tensor:
    """
    A multidimensional array with automatic differentiation.
    
    This is your PyTorch-like Tensor class that supports:
    - Basic arithmetic (+, -, *, /, @)
    - Reduction operations (sum, mean, max)
    - Shape operations (reshape, transpose, flatten)
    - Automatic gradient computation
    
    The core insight: every operation creates a new Tensor that
    remembers its parents and knows how to compute gradients.
    """
    
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
    
    @property
    def size(self):
        return self.data.size
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, grad_fn={self._op or 'None'})"
    
    def backward(self):
        """Compute gradients for all tensors in the computation graph."""
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
    
    def detach(self):
        """Return a copy detached from the computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)
    
    @staticmethod
    def _unbroadcast(grad, original_shape):
        """Sum out broadcast dimensions."""
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
            self.grad += Tensor._unbroadcast(out.grad, self.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += Tensor._unbroadcast(other.data * out.grad, self.shape)
            other.grad += Tensor._unbroadcast(self.data * out.grad, other.shape)
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return -self + other
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return Tensor(np.array(other)) / self
    
    def __pow__(self, n):
        out = Tensor(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    @property
    def T(self):
        out = Tensor(self.data.T, (self,), 'T')
        
        def _backward():
            self.grad += out.grad.T
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
                self.grad += np.broadcast_to(grad, self.shape).copy()
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
            count = np.prod([self.data.shape[i] for i in axis])
        
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
    
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape), (self,), 'reshape')
        original_shape = self.shape
        
        def _backward():
            self.grad += out.grad.reshape(original_shape)
        out._backward = _backward
        return out
    
    def flatten(self, start_dim=0):
        shape = self.shape
        new_shape = shape[:start_dim] + (-1,)
        return self.reshape(new_shape)
    
    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data + 1e-10), (self,), 'log')
        
        def _backward():
            self.grad += out.grad / (self.data + 1e-10)
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def max(self, axis=None, keepdims=False):
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'max')
        
        def _backward():
            if axis is None:
                mask = (self.data == result)
                self.grad += mask * out.grad / mask.sum()
            else:
                expanded = result if keepdims else np.expand_dims(result, axis=axis)
                mask = (self.data == expanded)
                grad = out.grad if keepdims else np.expand_dims(out.grad, axis=axis)
                self.grad += mask * grad / mask.sum(axis=axis, keepdims=True)
        out._backward = _backward
        return out


# ============================================================================
# Part 2: Neural Network Modules
# ============================================================================

class Parameter(Tensor):
    """A Tensor that is meant to be a learnable parameter."""
    
    def __init__(self, data, requires_grad=True):
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data, requires_grad=requires_grad)
    
    def __repr__(self):
        return f"Parameter(shape={self.shape})"


class Module:
    """Base class for all neural network modules."""
    
    _training = True
    
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
    
    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            if hasattr(self, '_parameters') and self._parameters is not None:
                self._parameters[name] = value
        elif isinstance(value, Module):
            if hasattr(self, '_modules') and self._modules is not None:
                self._modules[name] = value
        object.__setattr__(self, name, value)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def parameters(self) -> Iterator[Parameter]:
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()
    
    def named_parameters(self, prefix='') -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param
        for mod_name, module in self._modules.items():
            mod_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
            yield from module.named_parameters(prefix=mod_prefix)
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
    
    def train(self):
        self._training = True
        for module in self._modules.values():
            module.train()
        return self
    
    def eval(self):
        self._training = False
        for module in self._modules.values():
            module.eval()
        return self
    
    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())
    
    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        for name, module in self._modules.items():
            lines.append(f"  ({name}): {module}")
        lines.append(")")
        return "\n".join(lines)


# ============================================================================
# Part 3: Layers
# ============================================================================

class Linear(Module):
    """Fully connected layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        scale = np.sqrt(2.0 / in_features)
        self.weight = Parameter(np.random.randn(out_features, in_features) * scale)
        if bias:
            self.bias = Parameter(np.zeros(out_features))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features})"


class ReLU(Module):
    """ReLU activation."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    
    def __repr__(self):
        return "ReLU()"


class Flatten(Module):
    """Flatten layer."""
    
    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim
    
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim)
    
    def __repr__(self):
        return f"Flatten(start_dim={self.start_dim})"


class Dropout(Module):
    """Dropout layer."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x: Tensor) -> Tensor:
        if not self._training or self.p == 0:
            return x
        
        mask = np.random.binomial(1, 1 - self.p, x.shape) / (1 - self.p)
        out = Tensor(x.data * mask, (x,), 'dropout')
        
        def _backward():
            x.grad += out.grad * mask
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Dropout(p={self.p})"


class Sequential(Module):
    """Sequential container."""
    
    def __init__(self, *modules):
        super().__init__()
        for i, module in enumerate(modules):
            setattr(self, str(i), module)
    
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x
    
    def __len__(self):
        return len(self._modules)
    
    def __getitem__(self, idx):
        return list(self._modules.values())[idx]
    
    def __repr__(self):
        lines = ["Sequential("]
        for i, module in enumerate(self._modules.values()):
            lines.append(f"  ({i}): {module}")
        lines.append(")")
        return "\n".join(lines)


# ============================================================================
# Part 4: Convolution Layers (Simplified for final project)
# ============================================================================

def im2col(x, kH, kW, stride=1, padding=0):
    """Transform input to column matrix for efficient convolution."""
    N, C, H, W = x.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    
    col = np.zeros((N, C, kH, kW, H_out, W_out))
    for y in range(kH):
        y_max = y + stride * H_out
        for x_idx in range(kW):
            x_max = x_idx + stride * W_out
            col[:, :, y, x_idx, :, :] = x[:, :, y:y_max:stride, x_idx:x_max:stride]
    
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)


def col2im(col, x_shape, kH, kW, stride=1, padding=0):
    """Transform column matrix back to image format."""
    N, C, H, W = x_shape
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    
    col = col.reshape(N, H_out, W_out, C, kH, kW).transpose(0, 3, 4, 5, 1, 2)
    
    img = np.zeros((N, C, H_padded, W_padded))
    for y in range(kH):
        y_max = y + stride * H_out
        for x_idx in range(kW):
            x_max = x_idx + stride * W_out
            img[:, :, y:y_max:stride, x_idx:x_max:stride] += col[:, :, y, x_idx, :, :]
    
    if padding > 0:
        img = img[:, :, padding:-padding, padding:-padding]
    return img


class Conv2d(Module):
    """2D Convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.weight = Parameter(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale)
        if bias:
            self.bias = Parameter(np.zeros(out_channels))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        kH = kW = self.kernel_size
        
        H_out = (H + 2*self.padding - kH) // self.stride + 1
        W_out = (W + 2*self.padding - kW) // self.stride + 1
        
        col = im2col(x.data, kH, kW, self.stride, self.padding)
        weight_col = self.weight.data.reshape(self.out_channels, -1)
        
        out_data = col @ weight_col.T
        out_data = out_data.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        
        if self.bias is not None:
            out_data += self.bias.data.reshape(1, -1, 1, 1)
        
        children = (x, self.weight) if self.bias is None else (x, self.weight, self.bias)
        out = Tensor(out_data, children, 'conv2d')
        
        def _backward():
            C_out = self.out_channels
            dy = out.grad.transpose(0, 2, 3, 1).reshape(-1, C_out)
            
            if self.bias is not None:
                self.bias.grad += np.sum(out.grad, axis=(0, 2, 3))
            
            self.weight.grad += (dy.T @ col).reshape(self.weight.shape)
            
            dcol = dy @ weight_col
            x.grad += col2im(dcol, x.shape, kH, kW, self.stride, self.padding)
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Conv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size})"


class MaxPool2d(Module):
    """Max pooling layer."""
    
    def __init__(self, kernel_size: int, stride: int = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
    
    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        kH = kW = self.kernel_size
        s = self.stride
        
        H_out = (H - kH) // s + 1
        W_out = (W - kW) // s + 1
        
        out_data = np.zeros((N, C, H_out, W_out))
        mask = np.zeros_like(x.data)
        
        for h in range(H_out):
            for w in range(W_out):
                h_start, w_start = h * s, w * s
                window = x.data[:, :, h_start:h_start+kH, w_start:w_start+kW]
                out_data[:, :, h, w] = np.max(window, axis=(2, 3))
                
                window_flat = window.reshape(N, C, -1)
                max_idx = np.argmax(window_flat, axis=2)
                for n in range(N):
                    for c in range(C):
                        idx = max_idx[n, c]
                        h_idx = h_start + idx // kH
                        w_idx = w_start + idx % kW
                        mask[n, c, h_idx, w_idx] = 1
        
        out = Tensor(out_data, (x,), 'maxpool2d')
        
        def _backward():
            dx = np.zeros_like(x.data)
            for h in range(H_out):
                for w in range(W_out):
                    h_start, w_start = h * s, w * s
                    dx[:, :, h_start:h_start+kH, w_start:w_start+kW] += \
                        mask[:, :, h_start:h_start+kH, w_start:w_start+kW] * out.grad[:, :, h:h+1, w:w+1]
            x.grad += dx
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size})"


class BatchNorm2d(Module):
    """Batch normalization for 2D inputs."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = Parameter(np.ones(num_features))
        self.beta = Parameter(np.zeros(num_features))
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: Tensor) -> Tensor:
        if self._training:
            mean = np.mean(x.data, axis=(0, 2, 3))
            var = np.var(x.data, axis=(0, 2, 3))
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        std = np.sqrt(var + self.eps)
        x_norm = (x.data - mean[None, :, None, None]) / std[None, :, None, None]
        out_data = self.gamma.data[None, :, None, None] * x_norm + self.beta.data[None, :, None, None]
        
        out = Tensor(out_data, (x, self.gamma, self.beta), 'batchnorm2d')
        
        N, C, H, W = x.shape
        m = N * H * W
        
        def _backward():
            dx_norm = out.grad * self.gamma.data[None, :, None, None]
            
            self.gamma.grad += np.sum(out.grad * x_norm, axis=(0, 2, 3))
            self.beta.grad += np.sum(out.grad, axis=(0, 2, 3))
            
            x.grad += (1.0 / m) * (1.0 / std[None, :, None, None]) * (
                m * dx_norm
                - np.sum(dx_norm, axis=(0, 2, 3))[None, :, None, None]
                - x_norm * np.sum(dx_norm * x_norm, axis=(0, 2, 3))[None, :, None, None]
            )
        
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"BatchNorm2d({self.num_features})"


# ============================================================================
# Part 5: Loss Functions
# ============================================================================

def softmax(x):
    """Numerically stable softmax."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class CrossEntropyLoss(Module):
    """Cross-entropy loss for classification."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        N = logits.shape[0]
        probs = softmax(logits.data)
        probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
        
        loss = -np.mean(np.log(probs_clipped[np.arange(N), targets]))
        out = Tensor(loss, (logits,), 'cross_entropy')
        
        def _backward():
            grad = probs.copy()
            grad[np.arange(N), targets] -= 1
            grad /= N
            logits.grad += grad * out.grad
        
        out._backward = _backward
        return out


class MSELoss(Module):
    """Mean squared error loss."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        diff = pred - target
        return (diff * diff).mean()


# ============================================================================
# Part 6: Optimizers
# ============================================================================

class SGD:
    """SGD optimizer with momentum."""
    
    def __init__(self, parameters, lr: float = 0.01, momentum: float = 0, weight_decay: float = 0):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        if momentum > 0:
            self.velocities = [np.zeros_like(p.data) for p in self.parameters]
        else:
            self.velocities = None
    
    def step(self):
        for i, param in enumerate(self.parameters):
            grad = param.grad.copy()
            
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data
            
            if self.momentum > 0 and self.velocities is not None:
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                update = self.velocities[i]
            else:
                update = grad
            
            param.data -= self.lr * update
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


class Adam:
    """Adam optimizer."""
    
    def __init__(self, parameters, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            grad = param.grad
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


# ============================================================================
# Part 7: Data Loading
# ============================================================================

def generate_synthetic_mnist(n_samples: int = 1000, n_classes: int = 10, 
                              img_size: int = 28) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic MNIST-like data.
    
    Each class has a distinct pattern (simple for demonstration).
    
    Returns:
        images: Array of shape (n_samples, 1, img_size, img_size)
        labels: Array of shape (n_samples,)
    """
    # API hints:
    # - Create distinct pattern for each class (0-9)
    # - np.ogrid for circle mask: (x-c)^2 + (y-c)^2 < r^2
    # - Slicing for lines: img[h1:h2, w1:w2] += value
    # - Add small noise: np.random.randn(...) * 0.1 for variety
    # - Normalize at end: (images - mean) / (std + eps)
    # - Return tuple: (images, labels)
    
    return None


class DataLoader:
    """Simple data loader with batching."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(X)
    
    def __iter__(self):
        indices = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            batch_indices = indices[start:end]
            yield Tensor(self.X[batch_indices]), self.y[batch_indices]
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


# ============================================================================
# Part 8: CNN Model for Classification
# ============================================================================

class SimpleCNN(Module):
    """
    Simple CNN for image classification.
    
    Architecture:
        Conv(1->32, 3x3, padding=1) -> BN -> ReLU -> MaxPool
        Conv(32->64, 3x3, padding=1) -> BN -> ReLU -> MaxPool
        Flatten -> Linear(64*7*7 -> 128) -> ReLU -> Dropout
        Linear(128 -> num_classes)
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        
        # API hints:
        # - Two conv blocks: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        # - conv1: in_channels -> 32, conv2: 32 -> 64
        # - MaxPool2d(2, 2) halves spatial dims (28->14->7)
        # - FC layers: 64*7*7 -> 128 -> num_classes
        # - Dropout(0.5) for regularization
        
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # API hints:
        # - Block 1: conv1 -> bn1 -> relu -> pool
        # - Block 2: conv2 -> bn2 -> relu -> pool
        # - Classifier: flatten -> fc1 -> relu -> dropout -> fc2
        # - Output is logits (no softmax here)
        
        return None


# ============================================================================
# Part 9: Training Functions
# ============================================================================

def train_epoch(model: Module, train_loader: DataLoader, 
                loss_fn: Module, optimizer, device=None) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        
        logits = model(X_batch)
        if logits is None:
            continue
        
        loss = loss_fn(logits, y_batch)
        if loss is None:
            continue
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model: Module, data_loader: DataLoader, loss_fn: Module) -> Tuple[float, float]:
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in data_loader:
        logits = model(X_batch)
        if logits is None:
            continue
        
        loss = loss_fn(logits, y_batch)
        if loss is not None:
            total_loss += loss.data
        
        predictions = np.argmax(logits.data, axis=1)
        correct += np.sum(predictions == y_batch)
        total += len(y_batch)
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def train_model(model: Module, train_loader: DataLoader, val_loader: DataLoader,
                n_epochs: int = 10, lr: float = 0.001) -> Dict[str, List]:
    """Complete training loop."""
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{n_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2%}")
    
    return history


# ============================================================================
# Part 10: Main Execution
# ============================================================================

def main():
    """Main function to run the complete autodiff demonstration."""
    print("=" * 70)
    print("Day 35: Complete Autodiff Library - Final Project")
    print("=" * 70)
    
    print("\nGenerating synthetic MNIST data...")
    data = generate_synthetic_mnist(n_samples=1000)
    
    if data is None:
        print("ERROR: generate_synthetic_mnist not implemented!")
        print("Please complete the TODO sections.")
        return
    
    X_train, y_train = data
    print(f"Data shape: {X_train.shape}, Labels shape: {y_train.shape}")
    
    split = int(0.8 * len(X_train))
    X_val, y_val = X_train[split:], y_train[split:]
    X_train, y_train = X_train[:split], y_train[:split]
    
    train_loader = DataLoader(X_train, y_train, batch_size=32, shuffle=True)
    val_loader = DataLoader(X_val, y_val, batch_size=32, shuffle=False)
    
    print("\nBuilding model...")
    model = SimpleCNN(in_channels=1, num_classes=10)
    
    if model.conv1 is None:
        print("ERROR: SimpleCNN not implemented!")
        print("Please complete the TODO sections.")
        return
    
    print(f"Model parameters: {model.num_parameters()}")
    
    print("\nTraining...")
    history = train_model(model, train_loader, val_loader, n_epochs=5, lr=0.001)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nFinal Validation Accuracy: {history['val_accuracy'][-1]:.2%}")
    
    print("\n" + "=" * 70)
    print("Congratulations! You've built a complete autodiff library from scratch!")
    print("=" * 70)
    print("\nYou now understand:")
    print("  - How automatic differentiation works")
    print("  - How neural network layers are implemented")
    print("  - How backpropagation flows through the network")
    print("  - How optimizers update weights")
    print("\nThis is the foundation of PyTorch, TensorFlow, and other DL frameworks!")


# ============================================================================
# Test Functions
# ============================================================================

def test_tensor_ops():
    """Test basic tensor operations."""
    results = {}
    
    try:
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        results['add'] = np.allclose(c.data, [5, 7, 9])
        
        c.sum().backward()
        results['add_backward'] = np.allclose(a.grad, [1, 1, 1])
    except:
        results['add'] = False
        results['add_backward'] = False
    
    return results


def test_linear():
    """Test linear layer."""
    results = {}
    
    try:
        np.random.seed(42)
        linear = Linear(10, 5)
        x = Tensor(np.random.randn(4, 10))
        y = linear(x)
        
        results['forward'] = y.shape == (4, 5)
        
        y.sum().backward()
        results['backward'] = np.any(linear.weight.grad != 0)
    except:
        results['forward'] = False
        results['backward'] = False
    
    return results


def test_conv2d():
    """Test Conv2d layer."""
    results = {}
    
    try:
        np.random.seed(42)
        conv = Conv2d(3, 16, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = conv(x)
        
        results['forward'] = y.shape == (2, 16, 8, 8)
        
        y.sum().backward()
        results['backward'] = np.any(conv.weight.grad != 0)
    except:
        results['forward'] = False
        results['backward'] = False
    
    return results


def test_complete_model():
    """Test complete CNN model."""
    results = {}
    
    try:
        np.random.seed(42)
        model = SimpleCNN(in_channels=1, num_classes=10)
        
        if model.conv1 is None:
            return {'forward': False, 'backward': False}
        
        x = Tensor(np.random.randn(4, 1, 28, 28))
        y = model(x)
        
        if y is not None:
            results['forward'] = y.shape == (4, 10)
            
            loss_fn = CrossEntropyLoss()
            targets = np.array([0, 1, 2, 3])
            loss = loss_fn(y, targets)
            
            if loss is not None:
                loss.backward()
                results['backward'] = any(np.any(p.grad != 0) for p in model.parameters())
            else:
                results['backward'] = False
        else:
            results['forward'] = False
            results['backward'] = False
    except:
        results['forward'] = False
        results['backward'] = False
    
    return results


if __name__ == "__main__":
    print("Day 35: Final Project - Complete Autodiff Library")
    print("=" * 60)
    
    print("\nTensor Operations:")
    tensor_results = test_tensor_ops()
    for name, passed in tensor_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLinear Layer:")
    linear_results = test_linear()
    for name, passed in linear_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nConv2d Layer:")
    conv_results = test_conv2d()
    for name, passed in conv_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nComplete Model:")
    model_results = test_complete_model()
    for name, passed in model_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day35.py for comprehensive tests!")
    print("Run main() to train on synthetic MNIST!")
