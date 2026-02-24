"""
Day 30: Training Loop Infrastructure
====================================
Estimated time: 4-5 hours
Prerequisites: Days 25-29 (Complete training framework)

Learning objectives:
- Implement DataLoader for batched training
- Build complete training/validation loops
- Create checkpointing for saving/resuming training
- Implement early stopping and best model selection
- Build a complete mini deep learning framework

Key concepts:
- DataLoader: Provides batched, optionally shuffled data
  - Efficient memory usage (don't load all at once)
  - Shuffling for better generalization
  - Handles last incomplete batch

- Training Loop Components:
  - Forward pass: model(x)
  - Loss computation: criterion(pred, target)
  - Backward pass: loss.backward()
  - Optimizer step: optimizer.step()
  - Metrics tracking: accuracy, loss history

- Checkpointing:
  - Save model weights
  - Save optimizer state
  - Save epoch number
  - Resume training from checkpoint

- Early Stopping:
  - Monitor validation loss
  - Stop if no improvement for N epochs
  - Keep best model weights

Framework structure:
    dataset = Dataset(X, Y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = Sequential(...)
    optimizer = Adam(model.parameters())
    criterion = MSELoss()
    
    trainer = Trainer(model, optimizer, criterion)
    history = trainer.fit(train_loader, val_loader, epochs=100)
"""

import numpy as np
from typing import List, Iterator, Dict, Any, Optional, Tuple, Callable
from collections import OrderedDict
import math
import os


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
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        if self._parameters:
            for name, param in self._parameters.items():
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, param
        if recurse and self._modules:
            for mod_name, module in self._modules.items():
                mod_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from module.named_parameters(prefix=mod_prefix, recurse=True)
    
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
    
    def state_dict(self) -> Dict[str, np.ndarray]:
        state = OrderedDict()
        for name, param in self.named_parameters():
            state[name] = param.data.copy()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, np.ndarray]):
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data = state_dict[name].copy()


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
    """Base optimizer class."""
    
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
    
    def state_dict(self) -> Dict:
        return {
            'state': {k: dict(v) for k, v in self.state.items()},
            'param_groups': [
                {k: v for k, v in group.items() if k != 'params'}
                for group in self.param_groups
            ]
        }
    
    def load_state_dict(self, state_dict: Dict):
        for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
            for key, value in saved_group.items():
                group[key] = value


class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, params, lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        defaults = {'lr': lr, 'betas': betas, 'eps': eps}
        self._step_count = 0
        super().__init__(params, defaults)
    
    def step(self):
        self._step_count += 1
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_id = id(p)
                if param_id not in self.state:
                    self.state[param_id] = {
                        'exp_avg': np.zeros_like(p.data),
                        'exp_avg_sq': np.zeros_like(p.data)
                    }
                
                state = self.state[param_id]
                grad = p.grad
                
                state['exp_avg'] = beta1 * state['exp_avg'] + (1 - beta1) * grad
                state['exp_avg_sq'] = beta2 * state['exp_avg_sq'] + (1 - beta2) * grad ** 2
                
                bias_correction1 = 1 - beta1 ** self._step_count
                bias_correction2 = 1 - beta2 ** self._step_count
                
                m_hat = state['exp_avg'] / bias_correction1
                v_hat = state['exp_avg_sq'] / bias_correction2
                
                p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)


# ============================================================================
# Exercise 1: Dataset Class
# ============================================================================

class Dataset:
    """
    Simple dataset class holding features and labels.
    
    Provides:
    - Length via len()
    - Indexing via []
    - Iteration support
    
    Example:
        dataset = Dataset(X, Y)
        x, y = dataset[0]      # Get first sample
        for x, y in dataset:   # Iterate all samples
            ...
    """
    
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            Y: Label array of shape (n_samples, ...) 
        """
        # API hints:
        # - Validate: len(X) == len(Y)
        # - self.X = X, self.Y = Y
        
        self.X = X
        self.Y = Y
    
    def __len__(self) -> int:
        """Return number of samples."""
        # API hints:
        # - len(self.X) -> number of samples
        
        return 0  # Replace
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get sample by index."""
        # API hints:
        # - return (self.X[idx], self.Y[idx])
        
        return None, None  # Replace


# ============================================================================
# Exercise 2: DataLoader Class
# ============================================================================

class DataLoader:
    """
    DataLoader for batched iteration over a dataset.
    
    Features:
    - Batching: Groups samples into batches
    - Shuffling: Randomizes order each epoch
    - Drop last: Optionally skip incomplete last batch
    
    Example:
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for batch_x, batch_y in loader:
            # batch_x shape: (32, n_features)
            # batch_y shape: (32, ...)
            train_step(batch_x, batch_y)
    """
    
    def __init__(self, dataset: Dataset, batch_size: int = 32,
                 shuffle: bool = False, drop_last: bool = False):
        """
        Initialize DataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop incomplete last batch
        """
        # API hints:
        # - Store all args as instance attributes
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
    
    def __len__(self) -> int:
        """Return number of batches."""
        # API hints:
        # - n = len(self.dataset)
        # - drop_last: n // batch_size (floor division)
        # - keep_last: (n + batch_size - 1) // batch_size (ceiling division)
        
        return 0  # Replace
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Iterate over batches.
        
        Yields:
            Tuple of (batch_x, batch_y) arrays
        """
        # API hints:
        # - indices = np.arange(n)
        # - np.random.shuffle(indices) -> shuffle if needed
        # - range(0, n, batch_size) -> iterate in batch_size steps
        # - Handle last batch: drop if incomplete and drop_last=True
        # - self.dataset.X[batch_indices] -> batch slice
        # - yield (batch_x, batch_y)
        
        return iter([])  # Replace


# ============================================================================
# Exercise 3: Loss Functions
# ============================================================================

class MSELoss:
    """Mean Squared Error loss."""
    
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute MSE loss.
        
        MSE = mean((pred - target)²)
        """
        # API hints:
        # - diff = pred - target
        # - (diff * diff).mean() -> MSE
        
        return Tensor(np.array(0.0))  # Replace


class CrossEntropyLoss:
    """
    Cross-entropy loss for classification.
    
    Combines softmax and negative log likelihood.
    
    Args:
        pred: Raw logits of shape (batch, num_classes)
        target: Class indices of shape (batch,)
    """
    
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute cross-entropy loss.
        
        CE = -log(softmax(pred)[target])
        """
        # API hints:
        # - Stable softmax: subtract max before exp
        # - np.exp(logits - max_logits)
        # - softmax = exp_logits / sum(exp_logits)
        # - correct_probs = softmax[arange(batch), target]
        # - loss = -mean(log(correct_probs + eps))
        # - Backward: grad = softmax; grad[target] -= 1; grad /= batch_size
        
        return Tensor(np.array(0.0))  # Replace


# ============================================================================
# Exercise 4: Training Step Functions
# ============================================================================

def train_epoch(model: Module, dataloader: DataLoader,
                optimizer: Optimizer, criterion: Callable) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Neural network
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
    
    Returns:
        Dict with 'loss' and optionally 'accuracy'
    """
    # API hints:
    # - model.train() -> set training mode
    # - for batch_x, batch_y in dataloader: iterate batches
    # - optimizer.zero_grad() -> clear gradients
    # - Tensor(batch_x) -> convert to Tensor
    # - model(x) -> forward pass
    # - criterion(pred, y) -> compute loss
    # - loss.backward() -> backprop
    # - optimizer.step() -> update
    # - Track total_loss / n_batches
    
    return {}  # Replace


def validate_epoch(model: Module, dataloader: DataLoader,
                   criterion: Callable) -> Dict[str, float]:
    """
    Validate for one epoch (no gradient updates).
    
    Args:
        model: Neural network
        dataloader: Validation data loader
        criterion: Loss function
    
    Returns:
        Dict with 'loss' and optionally 'accuracy'
    """
    # API hints:
    # - model.eval() -> set eval mode
    # - Tensor(data, requires_grad=False) -> no grad tracking
    # - No zero_grad, backward, or step calls
    # - Just forward pass and loss computation
    
    return {}  # Replace


# ============================================================================
# Exercise 5: Checkpointing
# ============================================================================

def save_checkpoint(filepath: str, model: Module, optimizer: Optimizer,
                    epoch: int, loss: float, **kwargs):
    """
    Save training checkpoint.
    
    Args:
        filepath: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        **kwargs: Additional data to save
    """
    # API hints:
    # - model.state_dict() -> model weights
    # - optimizer.state_dict() -> optimizer state
    # - checkpoint = {'epoch': ..., 'loss': ..., 'model_state_dict': ...}
    # - np.savez(filepath, **checkpoint) -> save to file
    
    pass


def load_checkpoint(filepath: str, model: Module, 
                    optimizer: Optional[Optimizer] = None) -> Dict:
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load into
        optimizer: Optimizer to load into (optional)
    
    Returns:
        Dict with checkpoint metadata (epoch, loss, etc.)
    """
    # API hints:
    # - np.load(filepath, allow_pickle=True) -> load file
    # - loaded['model_state_dict'].item() -> get dict
    # - model.load_state_dict(...) -> restore weights
    # - Return {'epoch': ..., 'loss': ...}
    
    return {}  # Replace


# ============================================================================
# Exercise 6: Early Stopping
# ============================================================================

class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    
    Example:
        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        for epoch in range(1000):
            train_loss = train_epoch(...)
            val_loss = validate_epoch(...)
            
            if early_stopping(val_loss):
                print("Early stopping triggered!")
                break
            
            if early_stopping.is_best:
                save_checkpoint(...)
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0,
                 mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy
        """
        # API hints:
        # - Store all args as attributes
        # - self.counter = 0 -> tracks epochs without improvement
        # - self.best_value = None -> best metric seen
        # - self.is_best = False -> was last call the best?
        # - self.should_stop = False -> should training stop?
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.is_best = False
        self.should_stop = False
    
    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current metric value (e.g., validation loss)
        
        Returns:
            True if training should stop
        """
        # API hints:
        # - First call: set best_value, return False
        # - mode='min': improved if current < best - min_delta
        # - mode='max': improved if current > best + min_delta
        # - If improved: reset counter, update best_value, is_best=True
        # - If not improved: increment counter, is_best=False
        # - Return counter >= patience
        
        return False  # Replace


# ============================================================================
# Exercise 7: Complete Trainer Class
# ============================================================================

class Trainer:
    """
    Complete training orchestrator.
    
    Handles:
    - Training and validation loops
    - Early stopping
    - Checkpointing
    - History tracking
    - Learning rate scheduling
    
    Example:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir='./checkpoints'
        )
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,
            early_stopping_patience=10
        )
    """
    
    def __init__(self, model: Module, optimizer: Optimizer,
                 criterion: Callable, checkpoint_dir: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            model: Neural network
            optimizer: Optimizer
            criterion: Loss function
            checkpoint_dir: Directory for checkpoints
        """
        # API hints:
        # - Store all args as attributes
        # - self.history = {'train_loss': [], 'val_loss': []}
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoint_dir = checkpoint_dir
        self.history: Dict[str, List] = {'train_loss': [], 'val_loss': []}
    
    def fit(self, train_loader: DataLoader, 
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100,
            early_stopping_patience: Optional[int] = None,
            verbose: bool = True) -> Dict[str, List]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
        
        Returns:
            Training history dict
        """
        # API hints:
        # - EarlyStopping(patience=...) -> optional early stopping
        # - for epoch in range(epochs): main loop
        # - train_epoch(...) -> training step
        # - validate_epoch(...) -> validation step (if val_loader)
        # - self.history['train_loss'].append(...) -> track history
        # - save_checkpoint(...) -> save best model
        # - early_stopping(val_loss) -> check stopping criterion
        # - return self.history
        
        return self.history  # Replace
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        return validate_epoch(self.model, dataloader, self.criterion)


# ============================================================================
# Test Functions
# ============================================================================

def test_dataset():
    """Test Dataset class."""
    results = {}
    
    try:
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        
        dataset = Dataset(X, Y)
        
        results['length'] = len(dataset) == 100
        
        x, y = dataset[0]
        results['getitem'] = np.allclose(x, X[0]) and np.allclose(y, Y[0])
    except Exception as e:
        results['length'] = False
        results['getitem'] = False
    
    return results


def test_dataloader():
    """Test DataLoader class."""
    results = {}
    
    try:
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        results['length'] = len(loader) == 4
        
        batches = list(loader)
        results['iterates'] = len(batches) == 4
        results['batch_size'] = batches[0][0].shape[0] == 32
        results['last_batch'] = batches[-1][0].shape[0] == 4
    except Exception as e:
        results['length'] = False
        results['iterates'] = False
        results['batch_size'] = False
        results['last_batch'] = False
    
    return results


def test_dataloader_shuffle():
    """Test DataLoader shuffling."""
    results = {}
    
    try:
        np.random.seed(42)
        X = np.arange(100).reshape(-1, 1).astype(float)
        Y = np.arange(100).reshape(-1, 1).astype(float)
        dataset = Dataset(X, Y)
        
        loader = DataLoader(dataset, batch_size=100, shuffle=True)
        
        batch_x, _ = next(iter(loader))
        results['shuffles'] = not np.allclose(batch_x.flatten(), np.arange(100))
    except Exception as e:
        results['shuffles'] = False
    
    return results


def test_mse_loss():
    """Test MSE loss."""
    results = {}
    
    try:
        criterion = MSELoss()
        
        pred = Tensor(np.array([1.0, 2.0, 3.0]))
        target = Tensor(np.array([1.1, 2.2, 3.3]))
        
        loss = criterion(pred, target)
        expected = np.mean([0.01, 0.04, 0.09])
        
        results['correct'] = np.isclose(loss.data, expected, rtol=1e-4)
    except Exception as e:
        results['correct'] = False
    
    return results


def test_train_epoch():
    """Test train_epoch function."""
    results = {}
    
    try:
        np.random.seed(42)
        
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        loader = DataLoader(dataset, batch_size=32)
        
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        optimizer = Adam(model.parameters())
        criterion = MSELoss()
        
        metrics = train_epoch(model, loader, optimizer, criterion)
        
        results['returns_loss'] = 'loss' in metrics
        results['loss_positive'] = metrics.get('loss', 0) > 0
    except Exception as e:
        results['returns_loss'] = False
        results['loss_positive'] = False
    
    return results


def test_early_stopping():
    """Test EarlyStopping class."""
    results = {}
    
    try:
        es = EarlyStopping(patience=3)
        
        results['no_stop_initially'] = not es(1.0)
        results['no_stop_improving'] = not es(0.9)
        results['no_stop_patience_1'] = not es(0.95)
        results['no_stop_patience_2'] = not es(0.95)
        results['stops_patience_3'] = es(0.95)
    except Exception as e:
        results['no_stop_initially'] = False
        results['no_stop_improving'] = False
        results['no_stop_patience_1'] = False
        results['no_stop_patience_2'] = False
        results['stops_patience_3'] = False
    
    return results


def test_trainer():
    """Test Trainer class."""
    results = {}
    
    try:
        np.random.seed(42)
        
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        loader = DataLoader(dataset, batch_size=32)
        
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        optimizer = Adam(model.parameters())
        criterion = MSELoss()
        
        trainer = Trainer(model, optimizer, criterion)
        history = trainer.fit(loader, epochs=10, verbose=False)
        
        results['returns_history'] = 'train_loss' in history
        results['loss_decreases'] = history['train_loss'][-1] < history['train_loss'][0]
    except Exception as e:
        results['returns_history'] = False
        results['loss_decreases'] = False
    
    return results


if __name__ == "__main__":
    print("Day 30: Training Loop Infrastructure")
    print("=" * 60)
    
    print("\nDataset:")
    ds_results = test_dataset()
    for name, passed in ds_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nDataLoader:")
    dl_results = test_dataloader()
    for name, passed in dl_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nDataLoader Shuffle:")
    dls_results = test_dataloader_shuffle()
    for name, passed in dls_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nMSE Loss:")
    mse_results = test_mse_loss()
    for name, passed in mse_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nTrain Epoch:")
    te_results = test_train_epoch()
    for name, passed in te_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nEarly Stopping:")
    es_results = test_early_stopping()
    for name, passed in es_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nTrainer:")
    tr_results = test_trainer()
    for name, passed in tr_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day30.py for comprehensive tests!")
    print("\nCongratulations! You've completed the training framework!")
