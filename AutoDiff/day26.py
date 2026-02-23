"""
Day 26: SGD Optimizer
=====================
Estimated time: 2-3 hours
Prerequisites: Day 25 (Parameter and Module Base Class)

Learning objectives:
- Understand how optimizers update parameters
- Implement basic Stochastic Gradient Descent (SGD)
- Build the Optimizer base class pattern
- Learn the zero_grad -> forward -> backward -> step cycle

Key concepts:
- Optimizer: Algorithm that updates parameters using gradients
  - Receives list of parameters to optimize
  - step() applies the update rule
  - zero_grad() clears gradients before each iteration

- SGD update rule:
  - θ_{t+1} = θ_t - lr * ∇L(θ_t)
  - where lr is the learning rate

Mathematical background:
- Gradient descent finds minima by moving opposite to gradient
- Learning rate controls step size:
  - Too large: May overshoot and diverge
  - Too small: Slow convergence
- Stochastic: Uses mini-batches instead of full dataset
  - Faster iterations
  - Natural regularization from noise
  - Can escape shallow local minima

Training loop pattern:
    optimizer = SGD(model.parameters(), lr=0.01)
    for data, target in dataloader:
        optimizer.zero_grad()     # Clear old gradients
        output = model(data)      # Forward pass
        loss = criterion(output, target)
        loss.backward()           # Compute gradients
        optimizer.step()          # Update parameters
"""

import numpy as np
from typing import List, Iterator, Dict, Any, Optional
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
    """A Tensor that's marked as a learnable parameter."""
    
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
    """Linear layer: y = xW^T + b"""
    
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
# Exercise 1: Optimizer Base Class
# ============================================================================

class Optimizer:
    """
    Base class for all optimizers.
    
    Manages a list of parameter groups with their options.
    Provides interface for zero_grad() and step().
    
    Args:
        params: Iterable of Parameters or dicts containing parameters
                Dict format: {'params': [...], 'lr': 0.01, ...}
        defaults: Default options for all parameter groups
    
    Example:
        opt = SGD([p1, p2], lr=0.01)
        # or with parameter groups:
        opt = SGD([
            {'params': model.encoder.parameters(), 'lr': 0.01},
            {'params': model.decoder.parameters(), 'lr': 0.001}
        ], lr=0.01)
    """
    
    def __init__(self, params, defaults: Dict):
        """
        Initialize optimizer.
        
        Args:
            params: Parameters to optimize (iterable or list of dicts)
            defaults: Default hyperparameters
        """
        # TODO: Initialize param_groups list
        # HINT:
        # self.defaults = defaults
        # self.param_groups = []
        # 
        # params = list(params)  # Materialize iterator
        # if len(params) == 0:
        #     raise ValueError("optimizer got empty parameter list")
        # 
        # # Check if params is list of dicts (parameter groups) or list of Tensors
        # if isinstance(params[0], dict):
        #     for group in params:
        #         self.add_param_group(group)
        # else:
        #     self.add_param_group({'params': params})
        
        self.defaults = defaults
        self.param_groups = []  # Replace with initialization
    
    def add_param_group(self, param_group: Dict):
        """
        Add a parameter group to the optimizer.
        
        Args:
            param_group: Dict with 'params' key and optional hyperparameters
        """
        # TODO: Add parameter group with default values
        # HINT:
        # params = list(param_group['params'])
        # param_group['params'] = params
        # 
        # # Fill in missing values from defaults
        # for key, value in self.defaults.items():
        #     param_group.setdefault(key, value)
        # 
        # self.param_groups.append(param_group)
        
        pass  # Replace
    
    def zero_grad(self):
        """
        Clear gradients of all parameters.
        
        Called before each forward pass to prevent gradient accumulation.
        """
        # TODO: Zero gradients for all parameters
        # HINT:
        # for group in self.param_groups:
        #     for p in group['params']:
        #         p.zero_grad()
        
        pass  # Replace
    
    def step(self):
        """
        Perform a single optimization step.
        
        Override in subclasses to implement specific update rules.
        """
        raise NotImplementedError


# ============================================================================
# Exercise 2: Basic SGD Optimizer
# ============================================================================

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Update rule:
        θ = θ - lr * ∇L(θ)
    
    Args:
        params: Parameters to optimize
        lr: Learning rate (required)
    
    Example:
        optimizer = SGD(model.parameters(), lr=0.01)
        
        for data, target in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    """
    
    def __init__(self, params, lr: float):
        """
        Initialize SGD optimizer.
        
        Args:
            params: Parameters to optimize
            lr: Learning rate
        """
        # TODO: Initialize with lr as default
        # HINT:
        # if lr < 0.0:
        #     raise ValueError(f"Invalid learning rate: {lr}")
        # defaults = {'lr': lr}
        # super().__init__(params, defaults)
        
        pass  # Replace with initialization
    
    def step(self):
        """
        Perform SGD update step.
        
        For each parameter θ:
            θ = θ - lr * θ.grad
        """
        # TODO: Implement SGD update
        # HINT:
        # for group in self.param_groups:
        #     lr = group['lr']
        #     for p in group['params']:
        #         if p.grad is not None:
        #             p.data -= lr * p.grad
        
        pass  # Replace


# ============================================================================
# Exercise 3: SGD with Different Learning Rates
# ============================================================================

def create_optimizer_with_param_groups(model: Module, base_lr: float) -> SGD:
    """
    Create SGD with different learning rates for different layers.
    
    This is useful when you want:
    - Lower lr for pretrained layers (fine-tuning)
    - Higher lr for new layers
    - Different lr for weights vs biases
    
    Args:
        model: Model with layer1 and layer2 attributes
        base_lr: Base learning rate
    
    Returns:
        SGD optimizer with parameter groups
    
    Example output structure:
        [
            {'params': layer1_params, 'lr': base_lr * 0.1},
            {'params': layer2_params, 'lr': base_lr}
        ]
    """
    # TODO: Create optimizer with different lr per layer
    # HINT:
    # layer1_params = list(model.layer1.parameters())
    # layer2_params = list(model.layer2.parameters())
    # 
    # param_groups = [
    #     {'params': layer1_params, 'lr': base_lr * 0.1},  # Lower lr for layer1
    #     {'params': layer2_params, 'lr': base_lr}         # Base lr for layer2
    # ]
    # 
    # return SGD(param_groups, lr=base_lr)
    
    return None  # Replace


# ============================================================================
# Exercise 4: Simple Training Loop
# ============================================================================

def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error loss."""
    diff = pred - target
    return (diff * diff).mean()


def train_step(model: Module, optimizer: Optimizer, 
               x: Tensor, y: Tensor) -> float:
    """
    Perform a single training step.
    
    This is the fundamental pattern for all neural network training:
    1. Zero gradients
    2. Forward pass
    3. Compute loss
    4. Backward pass
    5. Update parameters
    
    Args:
        model: Neural network
        optimizer: Optimizer instance
        x: Input data
        y: Target labels
    
    Returns:
        Loss value as float
    """
    # TODO: Implement training step
    # HINT:
    # # 1. Clear gradients from previous iteration
    # optimizer.zero_grad()
    # 
    # # 2. Forward pass
    # pred = model(x)
    # 
    # # 3. Compute loss
    # loss = mse_loss(pred, y)
    # 
    # # 4. Backward pass - compute gradients
    # loss.backward()
    # 
    # # 5. Update parameters
    # optimizer.step()
    # 
    # return float(loss.data)
    
    return 0.0  # Replace


def train_loop(model: Module, optimizer: Optimizer,
               X: np.ndarray, Y: np.ndarray, 
               epochs: int = 100) -> List[float]:
    """
    Train model for multiple epochs.
    
    Args:
        model: Neural network
        optimizer: Optimizer instance
        X: Training inputs (numpy array)
        Y: Training targets (numpy array)
        epochs: Number of training iterations
    
    Returns:
        List of loss values per epoch
    """
    # TODO: Implement training loop
    # HINT:
    # losses = []
    # for epoch in range(epochs):
    #     x = Tensor(X)
    #     y = Tensor(Y)
    #     loss = train_step(model, optimizer, x, y)
    #     losses.append(loss)
    # return losses
    
    return []  # Replace


# ============================================================================
# Exercise 5: Optimizer State Management
# ============================================================================

class SGDWithState(Optimizer):
    """
    SGD with state dictionary support.
    
    The state dict allows:
    - Saving optimizer state (for checkpoints)
    - Loading optimizer state (resume training)
    - Storing per-parameter state (momentum buffers, etc.)
    """
    
    def __init__(self, params, lr: float):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {'lr': lr}
        self._state: Dict[Parameter, Dict] = {}  # Per-parameter state
        self._step_count = 0
        super().__init__(params, defaults)
    
    @property
    def state(self) -> Dict:
        """Return state dictionary."""
        return self._state
    
    def step(self):
        """Perform SGD step and track iteration count."""
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is not None:
                    p.data -= lr * p.grad
        self._step_count += 1
    
    def state_dict(self) -> Dict:
        """
        Return optimizer state as a dictionary.
        
        Returns:
            Dict containing param_groups and state
        """
        # TODO: Implement state dict generation
        # HINT:
        # return {
        #     'step_count': self._step_count,
        #     'param_groups': [
        #         {k: v for k, v in group.items() if k != 'params'}
        #         for group in self.param_groups
        #     ]
        # }
        
        return {}  # Replace
    
    def load_state_dict(self, state_dict: Dict):
        """
        Load optimizer state from dictionary.
        
        Args:
            state_dict: State dictionary from state_dict()
        """
        # TODO: Implement state dict loading
        # HINT:
        # self._step_count = state_dict.get('step_count', 0)
        # 
        # for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
        #     for key, value in saved_group.items():
        #         group[key] = value
        
        pass  # Replace


# ============================================================================
# Test Functions
# ============================================================================

def test_optimizer_base():
    """Test Optimizer base class."""
    results = {}
    
    try:
        params = [Parameter(np.random.randn(3, 4)) for _ in range(2)]
        opt = SGD(params, lr=0.01)
        
        results['creates'] = opt is not None
        results['has_groups'] = hasattr(opt, 'param_groups') and len(opt.param_groups) > 0
        
        if results['has_groups']:
            results['params_stored'] = len(opt.param_groups[0]['params']) == 2
        else:
            results['params_stored'] = False
    except Exception as e:
        results['creates'] = False
        results['has_groups'] = False
        results['params_stored'] = False
    
    return results


def test_sgd_step():
    """Test SGD step."""
    results = {}
    
    try:
        np.random.seed(42)
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        p.grad = np.array([0.1, 0.2, 0.3])
        
        opt = SGD([p], lr=0.1)
        old_data = p.data.copy()
        opt.step()
        
        expected = old_data - 0.1 * np.array([0.1, 0.2, 0.3])
        results['update'] = np.allclose(p.data, expected)
    except Exception as e:
        results['update'] = False
    
    return results


def test_zero_grad():
    """Test zero_grad."""
    results = {}
    
    try:
        p = Parameter(np.array([1.0, 2.0, 3.0]))
        p.grad = np.array([0.1, 0.2, 0.3])
        
        opt = SGD([p], lr=0.1)
        opt.zero_grad()
        
        results['zeroed'] = np.all(p.grad == 0)
    except Exception as e:
        results['zeroed'] = False
    
    return results


def test_training_step():
    """Test training step."""
    results = {}
    
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        opt = SGD(model.parameters(), lr=0.01)
        
        x = Tensor(np.random.randn(4, 4))
        y = Tensor(np.random.randn(4, 2))
        
        loss1 = train_step(model, opt, x, y)
        loss2 = train_step(model, opt, x, y)
        
        results['runs'] = loss1 > 0
        results['improves'] = loss2 < loss1
    except Exception as e:
        results['runs'] = False
        results['improves'] = False
    
    return results


def test_training_loop():
    """Test training loop."""
    results = {}
    
    try:
        np.random.seed(42)
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        
        opt = SGD(model.parameters(), lr=0.01)
        
        X = np.random.randn(10, 4)
        Y = np.random.randn(10, 2)
        
        losses = train_loop(model, opt, X, Y, epochs=50)
        
        results['returns_losses'] = len(losses) == 50
        results['loss_decreases'] = losses[-1] < losses[0] if losses else False
    except Exception as e:
        results['returns_losses'] = False
        results['loss_decreases'] = False
    
    return results


def test_param_groups():
    """Test parameter groups with different learning rates."""
    results = {}
    
    try:
        class TwoLayerNet(Module):
            def __init__(self):
                super().__init__()
                self.layer1 = Linear(4, 8)
                self.layer2 = Linear(8, 2)
            
            def forward(self, x):
                return self.layer2(ReLU()(self.layer1(x)))
        
        model = TwoLayerNet()
        opt = create_optimizer_with_param_groups(model, base_lr=0.1)
        
        if opt is None:
            results['creates'] = False
            results['different_lr'] = False
        else:
            results['creates'] = len(opt.param_groups) == 2
            if results['creates']:
                lr1 = opt.param_groups[0]['lr']
                lr2 = opt.param_groups[1]['lr']
                results['different_lr'] = lr1 != lr2
            else:
                results['different_lr'] = False
    except Exception as e:
        results['creates'] = False
        results['different_lr'] = False
    
    return results


if __name__ == "__main__":
    print("Day 26: SGD Optimizer")
    print("=" * 60)
    
    print("\nOptimizer Base:")
    base_results = test_optimizer_base()
    for name, passed in base_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSGD Step:")
    step_results = test_sgd_step()
    for name, passed in step_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nZero Grad:")
    zg_results = test_zero_grad()
    for name, passed in zg_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nTraining Step:")
    ts_results = test_training_step()
    for name, passed in ts_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nTraining Loop:")
    tl_results = test_training_loop()
    for name, passed in tl_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nParameter Groups:")
    pg_results = test_param_groups()
    for name, passed in pg_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day26.py for comprehensive tests!")
