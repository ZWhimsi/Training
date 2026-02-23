"""
Day 20: Softmax Implementation
==============================
Estimated time: 3-4 hours
Prerequisites: Days 12-19 (Tensor class, exp, log, logsumexp)

Learning objectives:
- Implement softmax with numerical stability
- Understand the Jacobian-vector product for softmax
- Implement log_softmax efficiently
- Build cross-entropy loss with proper gradients
- Understand the softmax-cross-entropy fusion optimization

Mathematical background:
========================

Softmax function:
For input x of shape (batch, classes):
    softmax(x)[i,j] = exp(x[i,j]) / sum_k(exp(x[i,k]))

Properties:
- Output sums to 1 along axis (probability distribution)
- Output is in (0, 1) for all elements
- Invariant to adding constant: softmax(x) = softmax(x + c)

Jacobian of softmax:
For a single sample, softmax has a dense Jacobian:
    ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
    
Where δ_ij is Kronecker delta (1 if i==j, 0 otherwise)

In matrix form for output s = softmax(x):
    J = diag(s) - s @ s.T

Jacobian-Vector Product (JVP) for backward:
Instead of forming the full Jacobian, we compute:
    grad_x = J.T @ grad_out 
           = (diag(s) - s @ s.T).T @ g
           = s * g - s * (s.T @ g)
           = s * (g - sum(s * g))

This is O(n) instead of O(n²)!

Log-Softmax:
    log_softmax(x) = log(softmax(x))
                   = x - logsumexp(x)

More numerically stable than log(softmax(x))

Gradient of log_softmax:
    d(log_softmax)/dx = I - softmax(x) (broadcast)
    Or per sample: grad = upstream - softmax * sum(upstream)

Cross-Entropy Loss:
For targets y (one-hot or class indices) and predictions p = softmax(x):
    CE = -sum(y * log(p)) = -sum(y * log_softmax(x))

For class indices:
    CE = -log_softmax(x)[target_class]

Combined softmax + cross-entropy gradient:
    d(CE)/dx = softmax(x) - y
    
This beautiful result is why we fuse them!
"""

import numpy as np
from typing import Tuple, Optional, Union


class Tensor:
    """Tensor class with softmax and cross-entropy."""
    
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
                self.grad += np.broadcast_to(grad, self.shape).copy()
        
        out._backward = _backward
        return out
    
    def exp(self):
        result = np.exp(self.data)
        out = Tensor(result, (self,), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        result = np.log(self.data)
        out = Tensor(result, (self,), 'log')
        
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 1: Softmax Forward Pass
    # ========================================================================
    
    def softmax(self, axis=-1):
        """
        Compute softmax along specified axis.
        
        softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
        
        Args:
            axis: Axis along which to compute softmax (default: -1, last axis)
        
        Returns:
            Tensor with softmax probabilities
        
        Numerical stability:
            Subtract max before exp to prevent overflow.
            softmax(x) = softmax(x - max(x))
        """
        # TODO: Implement forward pass (numerically stable)
        # HINT:
        # shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        # exp_shifted = np.exp(shifted)
        # result = exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)
        out = None  # Replace: Tensor(result, (self,), 'softmax')
        
        # TODO: Implement backward pass using Jacobian-vector product
        def _backward():
            # Efficient JVP: grad = s * (g - sum(s * g))
            # Where s = softmax output, g = upstream gradient
            # HINT:
            # s = out.data
            # g = out.grad
            # # Sum of (softmax * gradient) along axis
            # sum_sg = np.sum(s * g, axis=axis, keepdims=True)
            # self.grad += s * (g - sum_sg)
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 2: Log-Softmax (More Stable)
    # ========================================================================
    
    def log_softmax(self, axis=-1):
        """
        Compute log(softmax(x)) in a numerically stable way.
        
        log_softmax(x) = x - logsumexp(x)
        
        Args:
            axis: Axis along which to compute
        
        Returns:
            Tensor with log-softmax values
        
        This is more stable than log(softmax(x)) for very negative values.
        """
        # TODO: Implement forward pass
        # HINT:
        # # Compute logsumexp for stability
        # max_val = np.max(self.data, axis=axis, keepdims=True)
        # shifted = self.data - max_val
        # logsumexp = max_val + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
        # result = self.data - logsumexp
        out = None  # Replace: Tensor(result, (self,), 'log_softmax')
        
        # TODO: Implement backward pass
        def _backward():
            # d(log_softmax)/dx = I - softmax (broadcast)
            # grad = g - softmax * sum(g)
            # HINT:
            # softmax = np.exp(out.data)  # exp(log_softmax) = softmax
            # sum_g = np.sum(out.grad, axis=axis, keepdims=True)
            # self.grad += out.grad - softmax * sum_g
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 3: Cross-Entropy Loss (with class indices)
    # ========================================================================
    
    def cross_entropy(self, targets, axis=-1):
        """
        Compute cross-entropy loss with class indices.
        
        For each sample i:
            loss[i] = -log_softmax(x[i])[targets[i]]
        
        Args:
            targets: Class indices (numpy array of ints)
            axis: Axis for classes (default: -1)
        
        Returns:
            Tensor with per-sample losses
        
        This fuses softmax + negative log likelihood for numerical stability.
        """
        # Ensure targets is numpy array
        if isinstance(targets, Tensor):
            targets = targets.data.astype(int)
        else:
            targets = np.array(targets, dtype=int)
        
        # TODO: Implement forward pass
        # HINT:
        # # Compute log_softmax
        # max_val = np.max(self.data, axis=axis, keepdims=True)
        # shifted = self.data - max_val
        # logsumexp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
        # log_softmax = shifted - logsumexp
        # 
        # # Select the log probability of the correct class
        # # For 2D: log_softmax[range(batch), targets]
        # if self.ndim == 2:
        #     loss = -log_softmax[np.arange(len(targets)), targets]
        # else:
        #     loss = -np.take_along_axis(log_softmax, targets.reshape(-1, 1), axis=axis).squeeze(axis)
        out = None  # Replace: Tensor(loss, (self,), 'cross_entropy')
        
        # TODO: Implement backward pass
        def _backward():
            # Beautiful result: grad = softmax - one_hot(targets)
            # HINT:
            # # Compute softmax
            # max_val = np.max(self.data, axis=axis, keepdims=True)
            # shifted = self.data - max_val
            # exp_shifted = np.exp(shifted)
            # softmax = exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)
            # 
            # # Create one-hot encoding
            # one_hot = np.zeros_like(self.data)
            # if self.ndim == 2:
            #     one_hot[np.arange(len(targets)), targets] = 1
            # else:
            #     np.put_along_axis(one_hot, targets.reshape(-1, 1), 1, axis=axis)
            # 
            # # Gradient: softmax - one_hot, scaled by upstream
            # self.grad += (softmax - one_hot) * out.grad.reshape(-1, 1)
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 4: Cross-Entropy with One-Hot Targets
    # ========================================================================
    
    def cross_entropy_onehot(self, targets, axis=-1):
        """
        Compute cross-entropy loss with one-hot encoded targets.
        
        loss = -sum(targets * log_softmax(x), axis=axis)
        
        Args:
            targets: One-hot encoded targets (Tensor or numpy array)
            axis: Axis for classes
        
        Returns:
            Tensor with per-sample losses
        """
        if isinstance(targets, Tensor):
            targets_data = targets.data
        else:
            targets_data = np.array(targets, dtype=np.float64)
        
        # TODO: Implement forward pass
        # HINT:
        # # Compute log_softmax
        # max_val = np.max(self.data, axis=axis, keepdims=True)
        # shifted = self.data - max_val
        # logsumexp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
        # log_softmax = shifted - logsumexp
        # 
        # # Cross-entropy: -sum(target * log_softmax)
        # loss = -np.sum(targets_data * log_softmax, axis=axis)
        out = None  # Replace: Tensor(loss, (self,), 'cross_entropy_onehot')
        
        # TODO: Implement backward pass
        def _backward():
            # grad = (softmax - targets) * upstream
            # HINT:
            # max_val = np.max(self.data, axis=axis, keepdims=True)
            # shifted = self.data - max_val
            # exp_shifted = np.exp(shifted)
            # softmax = exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)
            # 
            # grad = out.grad
            # if grad.ndim < self.ndim:
            #     grad = np.expand_dims(grad, axis=axis)
            # self.grad += (softmax - targets_data) * grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 5: Softmax Temperature (for diversity control)
    # ========================================================================
    
    def softmax_temperature(self, temperature=1.0, axis=-1):
        """
        Softmax with temperature scaling.
        
        softmax_T(x) = softmax(x / T)
        
        Args:
            temperature: Temperature parameter
                - T > 1: Softer distribution (more uniform)
                - T < 1: Sharper distribution (more peaked)
                - T = 1: Standard softmax
            axis: Axis for softmax
        
        Returns:
            Tensor with temperature-scaled softmax
        """
        # TODO: Implement forward pass
        # HINT:
        # scaled = self.data / temperature
        # shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
        # exp_shifted = np.exp(shifted)
        # result = exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)
        out = None  # Replace: Tensor(result, (self,), 'softmax_temp')
        
        # TODO: Implement backward pass
        def _backward():
            # Same as softmax, but chain rule adds 1/T
            # HINT:
            # s = out.data
            # g = out.grad
            # sum_sg = np.sum(s * g, axis=axis, keepdims=True)
            # self.grad += (s * (g - sum_sg)) / temperature
            pass  # Replace
        
        out._backward = _backward
        return out


# ============================================================================
# Test Functions
# ============================================================================

def test_softmax_basic():
    """Test basic softmax."""
    results = {}
    
    x = Tensor([[1.0, 2.0, 3.0]])
    y = x.softmax()
    
    if y is not None and y.data is not None:
        # Check sum = 1
        results['sum_one'] = np.allclose(np.sum(y.data), 1.0)
        # Check positive
        results['positive'] = np.all(y.data > 0)
        # Check values
        expected = np.exp([1, 2, 3]) / np.sum(np.exp([1, 2, 3]))
        results['values'] = np.allclose(y.data[0], expected)
    else:
        results['sum_one'] = False
        results['positive'] = False
        results['values'] = False
    
    return results


def test_softmax_stability():
    """Test softmax numerical stability."""
    results = {}
    
    # Large values that would overflow naive implementation
    x = Tensor([[1000.0, 1001.0, 1002.0]])
    y = x.softmax()
    
    if y is not None and y.data is not None:
        results['no_nan'] = not np.any(np.isnan(y.data))
        results['no_inf'] = not np.any(np.isinf(y.data))
        results['sum_one'] = np.allclose(np.sum(y.data), 1.0)
    else:
        results['no_nan'] = False
        results['no_inf'] = False
        results['sum_one'] = False
    
    return results


def test_softmax_gradient():
    """Test softmax gradient (Jacobian-vector product)."""
    results = {}
    
    x = Tensor([[1.0, 2.0, 3.0]])
    y = x.softmax()
    y.backward()
    
    if x.grad is not None:
        # With upstream grad = 1, gradient should sum to 0
        # (softmax outputs sum to 1, so gradients must sum to 0)
        results['sum_zero'] = np.allclose(np.sum(x.grad), 0, atol=1e-10)
        results['shape'] = x.grad.shape == x.shape
    else:
        results['sum_zero'] = False
        results['shape'] = False
    
    return results


def test_softmax_batch():
    """Test softmax on batched input."""
    results = {}
    
    x = Tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])  # (2, 3)
    y = x.softmax(axis=-1)
    
    if y is not None and y.data is not None:
        # Each row should sum to 1
        results['row_sums'] = np.allclose(np.sum(y.data, axis=1), [1, 1])
        # Uniform input should give uniform output
        results['uniform'] = np.allclose(y.data[1], [1/3, 1/3, 1/3])
    else:
        results['row_sums'] = False
        results['uniform'] = False
    
    return results


def test_log_softmax():
    """Test log_softmax."""
    results = {}
    
    x = Tensor([[1.0, 2.0, 3.0]])
    y = x.log_softmax()
    
    if y is not None and y.data is not None:
        # exp(log_softmax) should give softmax
        softmax = np.exp(y.data)
        results['exp_sum_one'] = np.allclose(np.sum(softmax), 1.0)
        
        # Compare with softmax
        y2 = x.softmax()
        if y2 is not None:
            results['matches_softmax'] = np.allclose(np.exp(y.data), y2.data)
        else:
            results['matches_softmax'] = False
    else:
        results['exp_sum_one'] = False
        results['matches_softmax'] = False
    
    return results


def test_log_softmax_stability():
    """Test log_softmax stability."""
    results = {}
    
    # Very negative values
    x = Tensor([[-1000.0, -1001.0, -1002.0]])
    y = x.log_softmax()
    
    if y is not None and y.data is not None:
        results['no_nan'] = not np.any(np.isnan(y.data))
        results['no_inf'] = not np.any(np.isinf(y.data))
        results['finite'] = np.all(np.isfinite(y.data))
    else:
        results['no_nan'] = False
        results['no_inf'] = False
        results['finite'] = False
    
    return results


def test_cross_entropy():
    """Test cross-entropy loss."""
    results = {}
    
    # Predictions and targets
    logits = Tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])  # (2, 3)
    targets = np.array([2, 0])  # Correct classes
    
    loss = logits.cross_entropy(targets)
    
    if loss is not None and loss.data is not None:
        # Loss should be positive
        results['positive'] = np.all(loss.data > 0)
        
        # Compute expected loss manually
        log_sm = logits.data - np.log(np.sum(np.exp(logits.data), axis=1, keepdims=True))
        expected = -log_sm[np.arange(2), targets]
        results['values'] = np.allclose(loss.data, expected)
    else:
        results['positive'] = False
        results['values'] = False
    
    return results


def test_cross_entropy_gradient():
    """Test cross-entropy gradient (softmax - one_hot)."""
    results = {}
    
    logits = Tensor([[0.0, 0.0, 1.0]])  # Slightly prefer class 2
    targets = np.array([2])  # Target is class 2
    
    loss = logits.cross_entropy(targets)
    
    if loss is not None:
        loss.backward()
        
        if logits.grad is not None:
            # Gradient = softmax - one_hot
            softmax = np.exp(logits.data) / np.sum(np.exp(logits.data))
            one_hot = np.array([[0, 0, 1]])
            expected_grad = softmax - one_hot
            
            results['gradient'] = np.allclose(logits.grad, expected_grad)
            # Gradient should sum to 0
            results['sum_zero'] = np.allclose(np.sum(logits.grad), 0, atol=1e-10)
        else:
            results['gradient'] = False
            results['sum_zero'] = False
    else:
        results['gradient'] = False
        results['sum_zero'] = False
    
    return results


def test_cross_entropy_onehot():
    """Test cross-entropy with one-hot targets."""
    results = {}
    
    logits = Tensor([[1.0, 2.0, 3.0]])
    one_hot = np.array([[0, 0, 1]])  # Class 2
    
    loss = logits.cross_entropy_onehot(one_hot)
    
    if loss is not None and loss.data is not None:
        # Should match cross_entropy with index
        loss2 = logits.cross_entropy(np.array([2]))
        
        if loss2 is not None:
            results['matches_index'] = np.allclose(loss.data, loss2.data)
        else:
            results['matches_index'] = False
    else:
        results['matches_index'] = False
    
    return results


def test_softmax_temperature():
    """Test temperature-scaled softmax."""
    results = {}
    
    x = Tensor([[1.0, 2.0, 3.0]])
    
    # T=1: Standard softmax
    y1 = x.softmax_temperature(temperature=1.0)
    y_std = x.softmax()
    
    if y1 is not None and y_std is not None:
        results['t1_matches'] = np.allclose(y1.data, y_std.data)
    else:
        results['t1_matches'] = False
    
    # T>1: More uniform
    y_high = x.softmax_temperature(temperature=10.0)
    if y_high is not None and y_std is not None:
        # Higher temp should have higher entropy (more uniform)
        entropy_high = -np.sum(y_high.data * np.log(y_high.data + 1e-10))
        entropy_std = -np.sum(y_std.data * np.log(y_std.data + 1e-10))
        results['high_t_uniform'] = entropy_high > entropy_std
    else:
        results['high_t_uniform'] = False
    
    # T<1: More peaked
    y_low = x.softmax_temperature(temperature=0.1)
    if y_low is not None and y_std is not None:
        # Lower temp should be more peaked
        results['low_t_peaked'] = np.max(y_low.data) > np.max(y_std.data)
    else:
        results['low_t_peaked'] = False
    
    return results


if __name__ == "__main__":
    print("Day 20: Softmax Implementation")
    print("=" * 60)
    
    print("\nBasic Softmax:")
    basic_results = test_softmax_basic()
    for name, passed in basic_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSoftmax Stability:")
    stability_results = test_softmax_stability()
    for name, passed in stability_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSoftmax Gradient:")
    grad_results = test_softmax_gradient()
    for name, passed in grad_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nBatch Softmax:")
    batch_results = test_softmax_batch()
    for name, passed in batch_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLog-Softmax:")
    logsm_results = test_log_softmax()
    for name, passed in logsm_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLog-Softmax Stability:")
    logsm_stab = test_log_softmax_stability()
    for name, passed in logsm_stab.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nCross-Entropy Loss:")
    ce_results = test_cross_entropy()
    for name, passed in ce_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nCross-Entropy Gradient:")
    ce_grad = test_cross_entropy_gradient()
    for name, passed in ce_grad.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nCross-Entropy One-Hot:")
    ce_onehot = test_cross_entropy_onehot()
    for name, passed in ce_onehot.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSoftmax Temperature:")
    temp_results = test_softmax_temperature()
    for name, passed in temp_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day20.py for comprehensive tests!")
