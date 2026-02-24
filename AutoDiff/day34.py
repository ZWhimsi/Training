"""
Day 34: Complete CNN Module
===========================
Estimated time: 4-5 hours
Prerequisites: Days 31-33 (Conv2d, Pooling, BatchNorm)

Learning objectives:
- Combine Conv2d, pooling, and batch normalization into a complete CNN
- Implement a LeNet-style architecture from scratch
- Build a more modern CNN with residual connections
- Train a CNN on synthetic image data

Key concepts:
- CNN Architecture patterns:
  - Conv -> BatchNorm -> ReLU -> Pool (classic block)
  - Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm + skip (residual block)

- Common architectures:
  - LeNet-5: Simple 2 conv + 3 fc layers
  - VGG-style: Repeated conv blocks with increasing channels
  - ResNet-style: Residual connections for deeper networks

- Design choices:
  - Kernel size: 3x3 is most common (can capture patterns with less params)
  - Padding: same padding (p=k//2) preserves spatial dimensions
  - Stride: stride=2 for downsampling instead of pooling (modern)
  - BatchNorm: after every conv, before activation

Mathematical background:
- CNN reduces spatial dimensions while increasing channels
  - Input: (N, 3, 32, 32) - 3 channels, 32x32 spatial
  - After conv block: (N, 64, 16, 16) - more channels, smaller spatial
  - After flatten: (N, 64*4*4) = (N, 1024) features
  - Final: (N, num_classes) logits

- Parameter count calculation:
  - Conv2d(C_in, C_out, k): C_out * C_in * k * k + C_out (bias)
  - BatchNorm(C): 2 * C (gamma and beta)
  - Linear(in, out): in * out + out (bias)
"""

import numpy as np
from typing import Tuple, Optional, List, Union


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


def _unbroadcast(grad, original_shape):
    """Reverse broadcasting by summing over broadcast dimensions."""
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)
    for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
        if orig_dim == 1 and grad_dim > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# ============================================================================
# Base Module Class
# ============================================================================

class Module:
    """Base class for neural network modules."""
    
    _training = True
    
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
        for module in self._get_submodules():
            module.train()
    
    def eval(self):
        self._training = False
        for module in self._get_submodules():
            module.eval()
    
    def _get_submodules(self):
        """Get all submodules (override in containers)."""
        return []


# ============================================================================
# Exercise 1: Implement Core Layers (from previous days)
# ============================================================================

class Conv2d(Module):
    """2D Convolutional layer - implement based on Day 31."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        # API hints:
        # - He initialization: scale = sqrt(2.0 / fan_in), fan_in = in_channels * k * k
        # - weight shape: (out_channels, in_channels, kernel_size, kernel_size)
        # - bias shape: (out_channels,) or None if bias=False
        # - Use Tensor(...) to wrap numpy arrays
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = None  # Replace with Tensor
        self.bias = None    # Replace with Tensor
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution - implement im2col based convolution."""
        # API hints:
        # - Use im2col/col2im pattern from Day 31
        # - Or implement direct convolution with loops (slower but simpler)
        # - Track children for gradient computation
        # - Define _backward closure for gradient flow
        
        return None
    
    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight] if self.weight else []
    
    def __repr__(self):
        return f"Conv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size})"


class MaxPool2d(Module):
    """Max pooling layer - implement based on Day 32."""
    
    def __init__(self, kernel_size: int, stride: int = None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply max pooling."""
        # API hints:
        # - Extract windows, compute np.max over spatial dims
        # - Track max positions (mask) for backward
        # - Backward: gradient flows only to max positions
        
        return None
    
    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size})"


class BatchNorm2d(Module):
    """Batch normalization for 2D inputs - implement based on Day 33."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # API hints:
        # - gamma: Tensor of ones (scale parameter)
        # - beta: Tensor of zeros (shift parameter)
        # - running_mean/var: numpy arrays for inference mode
        
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply batch normalization."""
        # API hints:
        # - Training: compute batch mean/var over (N, H, W)
        # - Eval: use running_mean/running_var
        # - Normalize: (x - mean) / sqrt(var + eps)
        # - Scale and shift: gamma * x_norm + beta
        
        return None
    
    def parameters(self):
        if self.gamma is not None and self.beta is not None:
            return [self.gamma, self.beta]
        return []
    
    def __repr__(self):
        return f"BatchNorm2d({self.num_features})"


# ============================================================================
# Exercise 2: Basic Building Blocks
# ============================================================================

class ReLU(Module):
    """ReLU activation function."""
    
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(np.maximum(0, x.data), (x,), 'relu')
        
        def _backward():
            x.grad += (x.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "ReLU()"


class Flatten(Module):
    """Flatten spatial dimensions."""
    
    def forward(self, x: Tensor) -> Tensor:
        """Flatten (N, C, H, W) -> (N, C*H*W)"""
        N = x.shape[0]
        out_data = x.data.reshape(N, -1)
        out = Tensor(out_data, (x,), 'flatten')
        original_shape = x.shape
        
        def _backward():
            x.grad += out.grad.reshape(original_shape)
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "Flatten()"


class Linear(Module):
    """Fully connected layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(out_features, in_features) * scale)
        self.bias = Tensor(np.zeros(out_features)) if bias else None
    
    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def parameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        return [self.weight]
    
    def __repr__(self):
        return f"Linear({self.in_features}, {self.out_features})"


class Dropout(Module):
    """Dropout regularization."""
    
    def __init__(self, p: float = 0.5):
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
        self._modules = list(modules)
    
    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules:
            x = module(x)
        return x
    
    def parameters(self):
        params = []
        for module in self._modules:
            params.extend(module.parameters())
        return params
    
    def _get_submodules(self):
        return self._modules
    
    def __repr__(self):
        lines = ["Sequential("]
        for i, m in enumerate(self._modules):
            lines.append(f"  ({i}): {m}")
        lines.append(")")
        return "\n".join(lines)


# ============================================================================
# Exercise 3: ConvBlock - Basic CNN Building Block
# ============================================================================

class ConvBlock(Module):
    """
    Standard convolutional block: Conv -> BatchNorm -> ReLU
    
    This is the basic building block for most CNNs.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        """Initialize ConvBlock."""
        # API hints:
        # - Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # - BatchNorm2d(out_channels)
        # - ReLU() activation
        
        self.conv = None   # Replace
        self.bn = None     # Replace
        self.relu = None   # Replace
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply conv -> bn -> relu."""
        # API hints:
        # - Sequential application: x = conv(x), x = bn(x), x = relu(x)
        # - Check for None at each step (in case layers not implemented)
        # - Common pattern: Conv -> BatchNorm -> Activation
        
        return None
    
    def parameters(self):
        params = []
        if self.conv:
            params.extend(self.conv.parameters())
        if self.bn:
            params.extend(self.bn.parameters())
        return params
    
    def _get_submodules(self):
        modules = []
        if self.conv:
            modules.append(self.conv)
        if self.bn:
            modules.append(self.bn)
        return modules
    
    def __repr__(self):
        return f"ConvBlock(conv={self.conv}, bn={self.bn})"


# ============================================================================
# Exercise 4: ResidualBlock - Skip Connection Block
# ============================================================================

class ResidualBlock(Module):
    """
    Residual block with skip connection: out = F(x) + x
    
    If dimensions change, uses 1x1 conv to project skip connection.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for first convolution (for downsampling)
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """Initialize ResidualBlock."""
        # API hints:
        # - Main path: Conv2d(3x3) -> BN -> ReLU -> Conv2d(3x3) -> BN
        # - conv1 uses stride for downsampling, conv2 uses stride=1
        # - Skip connection: if dimensions change, use 1x1 conv to project
        # - Skip = Sequential(Conv2d(1x1, stride), BatchNorm2d) when needed
        # - Skip = None for identity shortcut (same dimensions)
        
        self.conv1 = None
        self.bn1 = None
        self.conv2 = None
        self.bn2 = None
        self.relu = None
        self.skip = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply residual block: out = F(x) + skip(x)."""
        # API hints:
        # - Store identity = x before main path
        # - Main path: conv1 -> bn1 -> relu -> conv2 -> bn2
        # - If skip exists: identity = skip(x)
        # - Final: out = F(x) + identity, then relu(out)
        # - The "+" is the residual connection that enables deep networks
        
        return None
    
    def parameters(self):
        params = []
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2, self.skip]:
            if layer:
                params.extend(layer.parameters())
        return params
    
    def _get_submodules(self):
        modules = []
        for layer in [self.conv1, self.bn1, self.conv2, self.bn2, self.skip]:
            if layer:
                if isinstance(layer, Sequential):
                    modules.extend(layer._modules)
                else:
                    modules.append(layer)
        return modules
    
    def __repr__(self):
        return f"ResidualBlock(conv1={self.conv1}, conv2={self.conv2})"


# ============================================================================
# Exercise 5: LeNet-style CNN
# ============================================================================

class LeNet(Module):
    """
    LeNet-style CNN for image classification.
    
    Architecture:
        Conv(1->6, 5x5) -> ReLU -> MaxPool
        Conv(6->16, 5x5) -> ReLU -> MaxPool
        Flatten
        Linear(16*4*4 -> 120) -> ReLU
        Linear(120 -> 84) -> ReLU
        Linear(84 -> num_classes)
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of output classes
    
    Expected input size: (N, C, 28, 28) or (N, C, 32, 32)
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        """Initialize LeNet."""
        # API hints:
        # - conv1: Conv2d(in_channels, 6, kernel_size=5, padding=2)
        # - conv2: Conv2d(6, 16, kernel_size=5)
        # - pool: MaxPool2d(kernel_size=2, stride=2)
        # - fc layers: 16*5*5 -> 120 -> 84 -> num_classes
        # - Calculate feature map size after convs and pools
        
        self.conv1 = None
        self.conv2 = None
        self.pool = None
        self.relu = None
        self.flatten = None
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through LeNet."""
        # API hints:
        # - Pattern: Conv -> ReLU -> Pool (repeat twice)
        # - Then Flatten -> FC -> ReLU -> FC -> ReLU -> FC
        # - Check for None at each step if layers might not be implemented
        # - Final output is logits (no softmax - applied in loss)
        
        return None
    
    def parameters(self):
        params = []
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]:
            if layer:
                params.extend(layer.parameters())
        return params
    
    def __repr__(self):
        return "LeNet()"


# ============================================================================
# Exercise 6: Modern CNN
# ============================================================================

class SimpleCNN(Module):
    """
    Simple modern CNN architecture.
    
    Architecture:
        ConvBlock(in->32, 3x3, stride=1)
        ConvBlock(32->64, 3x3, stride=2)  # Downsample
        ConvBlock(64->128, 3x3, stride=2) # Downsample
        GlobalAvgPool
        Linear(128 -> num_classes)
    
    Uses:
        - 3x3 kernels (efficient)
        - BatchNorm (stable training)
        - Stride for downsampling (no pooling needed)
        - Global average pooling (fewer params than FC)
    """
    
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        """Initialize SimpleCNN."""
        # API hints:
        # - ConvBlock(in, out, stride) for conv+bn+relu blocks
        # - stride=2 for downsampling (halves spatial dimensions)
        # - GlobalAvgPool reduces spatial dims to 1x1
        # - Linear(last_channels, num_classes) for classification
        
        self.block1 = None
        self.block2 = None
        self.block3 = None
        self.gap = None
        self.fc = None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        # API hints:
        # - Sequential: block1 -> block2 -> block3 -> gap -> fc
        # - Each ConvBlock includes conv+bn+relu
        # - GlobalAvgPool flattens spatial dims
        
        return None
    
    def parameters(self):
        params = []
        for layer in [self.block1, self.block2, self.block3, self.fc]:
            if layer:
                params.extend(layer.parameters())
        return params
    
    def __repr__(self):
        return "SimpleCNN()"


class GlobalAvgPool(Module):
    """Global average pooling - reduces (N, C, H, W) to (N, C)."""
    
    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        out_data = np.mean(x.data, axis=(2, 3))
        out = Tensor(out_data, (x,), 'global_avgpool')
        
        def _backward():
            x.grad += np.broadcast_to(out.grad[:, :, None, None] / (H * W), x.shape).copy()
        out._backward = _backward
        return out
    
    def __repr__(self):
        return "GlobalAvgPool()"


# ============================================================================
# Exercise 7: Cross-Entropy Loss
# ============================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class CrossEntropyLoss(Module):
    """
    Cross-entropy loss for classification.
    
    Combines softmax and negative log-likelihood.
    
    Args:
        logits: Raw model outputs of shape (N, C)
        targets: Class indices of shape (N,)
    
    Returns:
        Scalar loss
    """
    
    def forward(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        """Compute cross-entropy loss."""
        # API hints:
        # - softmax(logits) to get probabilities
        # - np.clip(probs, 1e-10, 1-1e-10) for numerical stability
        # - loss = -mean(log(probs[range(N), targets])) - negative log likelihood
        # - Backward: grad = probs - one_hot(targets), then grad /= N
        # - Simplified: grad[i, targets[i]] -= 1 for each sample
        
        return None
    
    def __repr__(self):
        return "CrossEntropyLoss()"


# ============================================================================
# Exercise 8: SGD Optimizer
# ============================================================================

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    
    Args:
        parameters: List of Tensor parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (0 for vanilla SGD)
        weight_decay: L2 regularization factor
    """
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01,
                 momentum: float = 0, weight_decay: float = 0):
        """Initialize SGD."""
        # API hints:
        # - Store parameters as list
        # - velocities: list of zero arrays matching param shapes (for momentum)
        # - Only create velocities if momentum > 0
        
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = None
    
    def step(self):
        """Perform one optimization step."""
        # API hints:
        # - For each param: grad = param.grad (optionally + weight_decay * param.data)
        # - With momentum: v = momentum * v + grad, update = v
        # - Without momentum: update = grad
        # - param.data -= lr * update
        
        pass
    
    def zero_grad(self):
        """Zero all parameter gradients."""
        for param in self.parameters:
            param.zero_grad()


# ============================================================================
# Test Functions
# ============================================================================

def test_conv_block():
    """Test ConvBlock."""
    results = {}
    
    try:
        np.random.seed(42)
        block = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        
        if block.conv is None:
            return {'forward': False, 'backward': False}
        
        x = Tensor(np.random.randn(2, 3, 8, 8))
        y = block(x)
        
        if y is not None:
            results['forward'] = y.shape == (2, 32, 8, 8)
            
            y.sum().backward()
            results['backward'] = any(np.any(p.grad != 0) for p in block.parameters())
        else:
            results['forward'] = False
            results['backward'] = False
    except Exception as e:
        results['forward'] = False
        results['backward'] = False
    
    return results


def test_residual_block():
    """Test ResidualBlock."""
    results = {}
    
    try:
        np.random.seed(42)
        
        block_same = ResidualBlock(32, 32, stride=1)
        block_down = ResidualBlock(32, 64, stride=2)
        
        if block_same.conv1 is None:
            return {'same_dims': False, 'downsample': False}
        
        x = Tensor(np.random.randn(2, 32, 8, 8))
        
        y_same = block_same(x)
        if y_same is not None:
            results['same_dims'] = y_same.shape == (2, 32, 8, 8)
        else:
            results['same_dims'] = False
        
        y_down = block_down(x)
        if y_down is not None:
            results['downsample'] = y_down.shape == (2, 64, 4, 4)
        else:
            results['downsample'] = False
    except Exception as e:
        results['same_dims'] = False
        results['downsample'] = False
    
    return results


def test_lenet():
    """Test LeNet architecture."""
    results = {}
    
    try:
        np.random.seed(42)
        model = LeNet(in_channels=1, num_classes=10)
        
        if model.conv1 is None:
            return {'forward': False, 'backward': False, 'params': False}
        
        x = Tensor(np.random.randn(2, 1, 28, 28))
        y = model(x)
        
        if y is not None:
            results['forward'] = y.shape == (2, 10)
            
            y.sum().backward()
            results['backward'] = any(np.any(p.grad != 0) for p in model.parameters())
            results['params'] = len(model.parameters()) > 0
        else:
            results['forward'] = False
            results['backward'] = False
            results['params'] = False
    except Exception as e:
        results['forward'] = False
        results['backward'] = False
        results['params'] = False
    
    return results


def test_simple_cnn():
    """Test SimpleCNN architecture."""
    results = {}
    
    try:
        np.random.seed(42)
        model = SimpleCNN(in_channels=3, num_classes=10)
        
        if model.block1 is None:
            return {'forward': False, 'backward': False}
        
        x = Tensor(np.random.randn(2, 3, 32, 32))
        y = model(x)
        
        if y is not None:
            results['forward'] = y.shape == (2, 10)
            
            y.sum().backward()
            results['backward'] = any(np.any(p.grad != 0) for p in model.parameters())
        else:
            results['forward'] = False
            results['backward'] = False
    except Exception as e:
        results['forward'] = False
        results['backward'] = False
    
    return results


def test_cross_entropy():
    """Test cross-entropy loss."""
    results = {}
    
    try:
        np.random.seed(42)
        loss_fn = CrossEntropyLoss()
        
        logits = Tensor(np.random.randn(4, 10))
        targets = np.array([0, 3, 5, 9])
        
        loss = loss_fn(logits, targets)
        
        if loss is not None:
            results['forward'] = loss.data.shape == ()
            
            loss.backward()
            results['backward'] = np.any(logits.grad != 0)
        else:
            results['forward'] = False
            results['backward'] = False
    except Exception as e:
        results['forward'] = False
        results['backward'] = False
    
    return results


def test_sgd():
    """Test SGD optimizer."""
    results = {}
    
    try:
        np.random.seed(42)
        
        w = Tensor(np.random.randn(3, 3))
        initial_w = w.data.copy()
        
        optimizer = SGD([w], lr=0.1)
        
        w.grad = np.ones_like(w.data)
        optimizer.step()
        
        results['updates'] = not np.allclose(w.data, initial_w)
        
        optimizer.zero_grad()
        results['zero_grad'] = np.allclose(w.grad, 0)
    except Exception as e:
        results['updates'] = False
        results['zero_grad'] = False
    
    return results


def test_training_loop():
    """Test a simple training loop."""
    results = {}
    
    try:
        np.random.seed(42)
        
        model = Sequential(
            Linear(10, 32),
            ReLU(),
            Linear(32, 5)
        )
        
        loss_fn = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.1)
        
        x = Tensor(np.random.randn(8, 10))
        targets = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        
        logits = model(x)
        if logits is None:
            return {'loss_decreases': False}
        
        initial_loss = loss_fn(logits, targets)
        if initial_loss is None:
            return {'loss_decreases': False}
        
        initial_loss_value = initial_loss.data
        
        for _ in range(10):
            optimizer.zero_grad()
            
            logits = model(x)
            if logits is None:
                return {'loss_decreases': False}
            
            loss = loss_fn(logits, targets)
            if loss is None:
                return {'loss_decreases': False}
            
            loss.backward()
            optimizer.step()
        
        final_loss_value = loss.data
        results['loss_decreases'] = final_loss_value < initial_loss_value
    except Exception as e:
        results['loss_decreases'] = False
    
    return results


if __name__ == "__main__":
    print("Day 34: Complete CNN Module")
    print("=" * 60)
    
    print("\nConvBlock:")
    conv_results = test_conv_block()
    for name, passed in conv_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nResidualBlock:")
    res_results = test_residual_block()
    for name, passed in res_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLeNet:")
    lenet_results = test_lenet()
    for name, passed in lenet_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSimpleCNN:")
    cnn_results = test_simple_cnn()
    for name, passed in cnn_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nCross-Entropy Loss:")
    ce_results = test_cross_entropy()
    for name, passed in ce_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSGD Optimizer:")
    sgd_results = test_sgd()
    for name, passed in sgd_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nTraining Loop:")
    train_results = test_training_loop()
    for name, passed in train_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day34.py for comprehensive tests!")
