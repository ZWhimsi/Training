"""
Day 21: Cross-Entropy Loss
==========================
Estimated time: 2-3 hours
Prerequisites: Days 15-20 (reductions, exp/log, softmax concepts)

Learning objectives:
- Implement cross-entropy loss with gradient
- Understand numerical stability in log computations
- Implement softmax + cross-entropy (log-softmax trick)
- Build loss functions used in classification

Key concepts:
- Cross-Entropy: H(p, q) = -sum(p * log(q))
  - For classification: -log(q[true_class])
  
- Softmax: softmax(x)_i = exp(x_i) / sum(exp(x))
  - Converts logits to probabilities
  
- Log-Softmax trick: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
  - More numerically stable than log(softmax(x))
  
- Cross-Entropy Loss gradient w.r.t. logits:
  - d(CE)/d(z) = softmax(z) - one_hot(target)
  - This elegant result makes backprop efficient!

Mathematical background:
- Binary Cross-Entropy: -[y*log(p) + (1-y)*log(1-p)]
- Multi-class Cross-Entropy: -sum_i(y_i * log(p_i))
- With softmax: -log(exp(z_c) / sum(exp(z))) where c is true class
"""

import numpy as np
from typing import Tuple, Optional, Union


class Tensor:
    """Tensor class with cross-entropy operations."""
    
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
    def size(self):
        return self.data.size
    
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
    
    @staticmethod
    def unbroadcast(grad, original_shape):
        """Reduce gradient to original shape."""
        while grad.ndim > len(original_shape):
            grad = grad.sum(axis=0)
        for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
            if orig_dim == 1 and grad_dim > 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    
    # Basic operations (provided)
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
    
    def sum(self, axis=None, keepdims=False):
        """Sum reduction."""
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
        """Mean reduction."""
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
        """Exponential function."""
        out = Tensor(np.exp(self.data), (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        """Natural logarithm."""
        out = Tensor(np.log(self.data), (self,), 'log')
        
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def max(self, axis=None, keepdims=False):
        """Max reduction."""
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(result, (self,), 'max')
        
        def _backward():
            if axis is None:
                mask = (self.data == out.data)
            else:
                expanded = out.data if keepdims else np.expand_dims(out.data, axis=axis)
                mask = (self.data == expanded)
            mask = mask / mask.sum(axis=axis, keepdims=True)
            grad = out.grad if keepdims else np.expand_dims(out.grad, axis=axis)
            self.grad += mask * np.broadcast_to(grad, self.shape)
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 1: Numerically Stable Log-Softmax
    # ========================================================================
    
    def log_softmax(self, axis=-1):
        """
        Compute log-softmax along specified axis.
        
        log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
        
        This is more stable than log(softmax(x)) because:
        1. Subtracting max prevents exp overflow
        2. We never explicitly compute softmax then take log
        
        Args:
            axis: Axis along which to compute log-softmax (default: -1)
        
        Returns:
            Tensor with log-softmax values
        
        Gradient: 
            d(log_softmax)/d(x) = I - softmax(x)
            where I is identity for the selected element
        """
        # API hints:
        # - np.max(arr, axis, keepdims=True) -> get max along axis
        # - np.exp(arr) -> element-wise exponential
        # - np.sum(arr, axis, keepdims=True) -> sum along axis
        # - np.log(arr) -> element-wise natural log
        # - Tensor(data, (self,), 'op') -> create output tensor
        # - Backward: grad_sum = np.sum(out.grad, axis, keepdims=True)
        # - Formula: log_softmax = x_shifted - log(sum(exp(x_shifted)))
        
        return None
    
    # ========================================================================
    # Exercise 2: Softmax Function
    # ========================================================================
    
    def softmax(self, axis=-1):
        """
        Compute softmax along specified axis.
        
        softmax(x)_i = exp(x_i) / sum(exp(x))
        
        Uses log-softmax for stability: softmax = exp(log_softmax)
        
        Args:
            axis: Axis along which to compute softmax (default: -1)
        
        Returns:
            Tensor with softmax probabilities (sum to 1 along axis)
        """
        # API hints:
        # - self.log_softmax(axis) -> compute log-softmax
        # - tensor.exp() -> element-wise exponential
        # - Formula: softmax = exp(log_softmax(x))
        
        return None
    
    # ========================================================================
    # Exercise 3: Binary Cross-Entropy Loss
    # ========================================================================
    
    def binary_cross_entropy(self, target, eps=1e-7):
        """
        Binary cross-entropy loss.
        
        BCE = -[y * log(p) + (1-y) * log(1-p)]
        
        Args:
            target: Target tensor with values in [0, 1]
            eps: Small value for numerical stability
        
        Returns:
            Tensor with BCE loss (mean over all elements)
        
        Gradient w.r.t. self (predicted probability p):
            d(BCE)/d(p) = -y/p + (1-y)/(1-p)
                        = (p - y) / (p * (1-p))
        """
        if isinstance(target, (int, float, list, np.ndarray)):
            target = Tensor(target, requires_grad=False)
        
        # API hints:
        # - np.clip(arr, min, max) -> clamp values for stability
        # - np.log(arr) -> element-wise natural log
        # - np.mean(arr) -> compute mean
        # - Formula: BCE = -[y * log(p) + (1-y) * log(1-p)]
        # - Backward gradient: (p - y) / (p * (1-p)) / n
        
        return None
    
    # ========================================================================
    # Exercise 4: Cross-Entropy Loss (from logits)
    # ========================================================================
    
    def cross_entropy_loss(self, target):
        """
        Cross-entropy loss from logits.
        
        CE = -log(softmax(x)[target_class])
           = -x[target_class] + log(sum(exp(x)))
        
        Args:
            target: Integer class indices (not one-hot!)
                   Shape: (batch_size,) for logits shape (batch_size, num_classes)
        
        Returns:
            Tensor with mean cross-entropy loss
        
        Gradient w.r.t. logits:
            d(CE)/d(z) = softmax(z) - one_hot(target)
            
        This is the famous result that makes classification efficient!
        """
        if isinstance(target, (list, np.ndarray)):
            target = np.array(target, dtype=np.int64)
        
        # API hints:
        # - np.max(arr, axis=-1, keepdims=True) -> max for stability
        # - np.exp(arr), np.log(arr), np.sum(arr, axis, keepdims)
        # - np.arange(batch_size) -> indices for batch selection
        # - arr[np.arange(n), indices] -> select elements by index
        # - np.mean(arr) -> compute mean loss
        # - Backward: grad = softmax - one_hot(target), scaled by 1/batch_size
        
        return None
    
    # ========================================================================
    # Exercise 5: Negative Log-Likelihood Loss
    # ========================================================================
    
    def nll_loss(self, target):
        """
        Negative log-likelihood loss (expects log-probabilities as input).
        
        NLL = -log_probs[target_class]
        
        Args:
            target: Integer class indices
        
        Returns:
            Tensor with mean NLL loss
        
        Note: Use this with log_softmax output:
            loss = logits.log_softmax().nll_loss(target)
        """
        if isinstance(target, (list, np.ndarray)):
            target = np.array(target, dtype=np.int64)
        
        # API hints:
        # - self.data.shape[0] -> batch_size
        # - np.arange(batch_size) -> batch indices
        # - arr[np.arange(n), target] -> select target class values
        # - np.mean(arr) -> compute mean
        # - np.zeros_like(arr) -> create zero gradient array
        # - Backward: grad is -1/batch_size at target positions, 0 elsewhere
        
        return None


# ============================================================================
# Test Functions
# ============================================================================

def test_log_softmax():
    """Test log-softmax implementation."""
    results = {}
    
    x = Tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    log_sm = x.log_softmax()
    
    if log_sm is not None and log_sm.data is not None:
        # log-softmax should sum to 0 when exp'd
        probs = np.exp(log_sm.data)
        results['sums_to_1'] = np.allclose(probs.sum(axis=-1), [1, 1])
        
        # Check specific values
        expected = np.array([[-2.40760596, -1.40760596, -0.40760596],
                           [-1.09861229, -1.09861229, -1.09861229]])
        results['values'] = np.allclose(log_sm.data, expected, rtol=1e-5)
    else:
        results['sums_to_1'] = False
        results['values'] = False
    
    return results


def test_softmax():
    """Test softmax implementation."""
    results = {}
    
    x = Tensor([[1.0, 2.0, 3.0]])
    sm = x.softmax()
    
    if sm is not None and sm.data is not None:
        results['sums_to_1'] = np.allclose(sm.data.sum(), 1)
        results['positive'] = np.all(sm.data > 0)
        results['ordered'] = sm.data[0, 2] > sm.data[0, 1] > sm.data[0, 0]
    else:
        results['sums_to_1'] = False
        results['positive'] = False
        results['ordered'] = False
    
    return results


def test_binary_cross_entropy():
    """Test binary cross-entropy."""
    results = {}
    
    # Perfect prediction
    pred = Tensor([0.9, 0.1])
    target = Tensor([1.0, 0.0])
    loss = pred.binary_cross_entropy(target)
    
    if loss is not None and loss.data is not None:
        results['perfect_low'] = loss.data < 0.2
        
        # Test gradient
        loss.backward()
        results['has_grad'] = np.any(pred.grad != 0)
    else:
        results['perfect_low'] = False
        results['has_grad'] = False
    
    # Bad prediction should have higher loss
    pred2 = Tensor([0.1, 0.9])
    loss2 = pred2.binary_cross_entropy(target)
    
    if loss2 is not None and loss2.data is not None:
        results['bad_high'] = loss2.data > 1.0
    else:
        results['bad_high'] = False
    
    return results


def test_cross_entropy_loss():
    """Test cross-entropy loss from logits."""
    results = {}
    
    # Batch of 2, 3 classes
    logits = Tensor([[2.0, 1.0, 0.1],   # Should predict class 0
                     [0.1, 0.2, 3.0]])  # Should predict class 2
    target = np.array([0, 2])  # Correct classes
    
    loss = logits.cross_entropy_loss(target)
    
    if loss is not None and loss.data is not None:
        results['forward'] = loss.data > 0  # Loss should be positive
        
        loss.backward()
        # Gradient should be softmax - one_hot
        results['has_grad'] = np.any(logits.grad != 0)
        
        # Check gradient sums to 0 per sample (property of softmax - one_hot)
        grad_sum = logits.grad.sum(axis=-1)
        results['grad_sums_zero'] = np.allclose(grad_sum, 0, atol=1e-6)
    else:
        results['forward'] = False
        results['has_grad'] = False
        results['grad_sums_zero'] = False
    
    return results


def test_nll_loss():
    """Test NLL loss."""
    results = {}
    
    # Log probabilities (already log-softmax output)
    log_probs = Tensor([[-0.5, -1.5, -2.5],
                        [-2.0, -0.5, -1.5]])
    target = np.array([0, 1])
    
    loss = log_probs.nll_loss(target)
    
    if loss is not None and loss.data is not None:
        # Should be -mean([-0.5, -0.5]) = 0.5
        results['forward'] = np.allclose(loss.data, 0.5)
        
        loss.backward()
        results['has_grad'] = np.any(log_probs.grad != 0)
    else:
        results['forward'] = False
        results['has_grad'] = False
    
    return results


def test_cross_entropy_gradient():
    """Test the elegant gradient: softmax - one_hot."""
    results = {}
    
    logits = Tensor([[1.0, 2.0, 3.0]])
    target = np.array([1])  # True class is 1
    
    loss = logits.cross_entropy_loss(target)
    
    if loss is not None:
        loss.backward()
        
        # Compute expected gradient
        exp_x = np.exp(logits.data - logits.data.max())
        softmax = exp_x / exp_x.sum()
        one_hot = np.array([[0, 1, 0]])
        expected_grad = softmax - one_hot
        
        results['gradient_correct'] = np.allclose(logits.grad, expected_grad, rtol=1e-5)
    else:
        results['gradient_correct'] = False
    
    return results


if __name__ == "__main__":
    print("Day 21: Cross-Entropy Loss")
    print("=" * 60)
    
    print("\nLog-Softmax:")
    log_sm_results = test_log_softmax()
    for name, passed in log_sm_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSoftmax:")
    sm_results = test_softmax()
    for name, passed in sm_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nBinary Cross-Entropy:")
    bce_results = test_binary_cross_entropy()
    for name, passed in bce_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nCross-Entropy Loss:")
    ce_results = test_cross_entropy_loss()
    for name, passed in ce_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nNLL Loss:")
    nll_results = test_nll_loss()
    for name, passed in nll_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nCross-Entropy Gradient (softmax - one_hot):")
    grad_results = test_cross_entropy_gradient()
    for name, passed in grad_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day21.py for comprehensive tests!")
