"""
Day 7: Optimizers
=================
Estimated time: 1-2 hours
Prerequisites: Day 6 (loss functions)

Learning objectives:
- Understand gradient descent and variants
- Implement SGD, SGD with momentum
- Learn about Adam optimizer
- Understand learning rate and momentum
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any


# ============================================================================
# Exercise 1: Vanilla SGD
# ============================================================================

class SGD_Manual:
    """
    Implement vanilla Stochastic Gradient Descent.
    
    Update rule: param = param - lr * grad
    """
    def __init__(self, params: List[torch.Tensor], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr
    
    def zero_grad(self):
        """Set all gradients to zero."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """
        Perform one optimization step.
        
        TODO: Update each parameter
        HINT: 
        with torch.no_grad():
            for param in self.params:
                if param.grad is not None:
                    param -= self.lr * param.grad
        """
        pass  # Replace


# ============================================================================
# Exercise 2: SGD with Momentum
# ============================================================================

class SGD_Momentum:
    """
    SGD with momentum.
    
    v = momentum * v + grad
    param = param - lr * v
    """
    def __init__(self, params: List[torch.Tensor], lr: float = 0.01, momentum: float = 0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        # Initialize velocity for each parameter
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """
        TODO: Update with momentum
        HINT:
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is not None:
                    self.v[i] = self.momentum * self.v[i] + param.grad
                    param -= self.lr * self.v[i]
        """
        pass  # Replace


# ============================================================================
# Exercise 3: Adam Optimizer
# ============================================================================

class Adam_Manual:
    """
    Adam optimizer.
    
    m = β1 * m + (1 - β1) * grad           (first moment)
    v = β2 * v + (1 - β2) * grad²          (second moment)
    m_hat = m / (1 - β1^t)                  (bias correction)
    v_hat = v / (1 - β2^t)                  (bias correction)
    param = param - lr * m_hat / (√v_hat + ε)
    """
    def __init__(self, params: List[torch.Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        
        # Initialize first and second moment estimates
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        """
        TODO: Implement Adam update
        HINT:
        self.t += 1
        with torch.no_grad():
            for i, param in enumerate(self.params):
                if param.grad is not None:
                    g = param.grad
                    self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                    self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
                    m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                    param -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        """
        pass  # Replace


# ============================================================================
# Exercise 4: Learning Rate Scheduler
# ============================================================================

class StepLRScheduler:
    """
    Step learning rate scheduler.
    Multiplies lr by gamma every step_size epochs.
    """
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0
        self.base_lr = optimizer.lr
    
    def step(self):
        """
        TODO: Update learning rate if needed
        HINT:
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
        """
        pass  # Replace
    
    def get_lr(self):
        return self.optimizer.lr


# ============================================================================
# Exercise 5: Test Optimizer on Simple Problem
# ============================================================================

def train_simple_model(optimizer_class, n_steps: int = 100):
    """
    Train a simple linear model to fit y = 2x + 1
    
    Returns final loss.
    """
    # Simple model: y = w*x + b
    w = torch.tensor([0.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    
    # Training data
    x = torch.linspace(-1, 1, 20).unsqueeze(1)
    y_true = 2 * x + 1
    
    # Create optimizer
    if optimizer_class == 'sgd':
        opt = SGD_Manual([w, b], lr=0.1)
    elif optimizer_class == 'momentum':
        opt = SGD_Momentum([w, b], lr=0.1, momentum=0.9)
    elif optimizer_class == 'adam':
        opt = Adam_Manual([w, b], lr=0.1)
    else:
        return None
    
    # Training loop
    losses = []
    for _ in range(n_steps):
        opt.zero_grad()
        y_pred = w * x + b
        loss = ((y_pred - y_true) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    
    return losses[-1], w.item(), b.item()


if __name__ == "__main__":
    print("Day 7: Optimizers")
    print("=" * 50)
    
    print("\nTraining y = 2x + 1:")
    for name in ['sgd', 'momentum', 'adam']:
        result = train_simple_model(name)
        if result is not None:
            loss, w, b = result
            print(f"  {name}: loss={loss:.6f}, w={w:.3f} (target=2), b={b:.3f} (target=1)")
        else:
            print(f"  {name}: NOT IMPLEMENTED")
    
    print("\nRun test_day07.py to verify all implementations!")
