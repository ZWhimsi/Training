"""
Day 19: Exp and Log Operations with Numerical Stability
========================================================
Estimated time: 2-3 hours
Prerequisites: Days 12-18 (Tensor class, max operations)

Learning objectives:
- Implement exp and log with proper gradients
- Understand numerical stability issues (overflow/underflow)
- Implement stable logsumexp operation
- Build sigmoid and tanh activations from primitives
- Learn about log-domain computations

Mathematical background:
========================

Exponential function:
- y = exp(x)
- dy/dx = exp(x) = y
- Key property: exp(a+b) = exp(a) * exp(b)

Natural logarithm:
- y = log(x)
- dy/dx = 1/x
- Key property: log(a*b) = log(a) + log(b)
- Domain: x > 0 only!

Numerical stability issues:
- exp(x) overflows for large x (x > 709 for float64)
- exp(x) underflows to 0 for very negative x
- log(x) returns -inf for x = 0
- log(sum(exp(x))) is numerically unstable

LogSumExp trick (stable softmax denominator):
- logsumexp(x) = log(sum(exp(x)))
- Naive: overflow for large x values
- Stable: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
- This subtracts max before exp, preventing overflow

Sigmoid function:
- sigmoid(x) = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))
- d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
- Range: (0, 1)
- Numerically stable: use exp(-abs(x)) and handle sign

Tanh function:
- tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
- tanh(x) = 2 * sigmoid(2x) - 1
- d(tanh)/dx = 1 - tanh(x)²
- Range: (-1, 1)
"""

import numpy as np
from typing import Tuple, Optional


class Tensor:
    """Tensor class with exp, log, and activation functions."""
    
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
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        out = Tensor(self.data / other.data, (self, other), '/')
        
        def _backward():
            self.grad += Tensor.unbroadcast((1/other.data) * out.grad, self.shape)
            other.grad += Tensor.unbroadcast((-self.data/(other.data**2)) * out.grad, other.shape)
        out._backward = _backward
        return out
    
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
                    if isinstance(axis, int):
                        grad = np.expand_dims(grad, axis=axis)
                    else:
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                self.grad += np.broadcast_to(grad, self.shape).copy()
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 1: Exponential Function
    # ========================================================================
    
    def exp(self):
        """
        Element-wise exponential: y = e^x
        
        Returns:
            Tensor with exp applied
        
        Gradient: dy/dx = e^x = y (output equals gradient!)
        
        Warning: exp(x) overflows for x > ~709 (float64)
        """
        # TODO: Implement forward pass
        # HINT: result = np.exp(self.data)
        out = None  # Replace: Tensor(result, (self,), 'exp')
        
        # TODO: Implement backward pass
        def _backward():
            # d/dx(e^x) = e^x = out.data
            # HINT: self.grad += out.data * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 2: Natural Logarithm
    # ========================================================================
    
    def log(self):
        """
        Element-wise natural logarithm: y = ln(x)
        
        Returns:
            Tensor with log applied
        
        Gradient: dy/dx = 1/x
        
        Warning: Only valid for x > 0. log(0) = -inf, log(negative) = NaN
        """
        # TODO: Implement forward pass
        # HINT: result = np.log(self.data)
        out = None  # Replace: Tensor(result, (self,), 'log')
        
        # TODO: Implement backward pass
        def _backward():
            # d/dx(ln(x)) = 1/x
            # HINT: self.grad += (1 / self.data) * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 3: LogSumExp (Numerically Stable)
    # ========================================================================
    
    def logsumexp(self, axis=None, keepdims=False):
        """
        Compute log(sum(exp(x))) in a numerically stable way.
        
        Uses the identity: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
        
        Args:
            axis: Axis along which to compute
            keepdims: Keep reduced dimensions
        
        Returns:
            Tensor with logsumexp result
        
        Gradient: d(logsumexp)/dx = softmax(x)
                  = exp(x) / sum(exp(x))
                  = exp(x - logsumexp(x))
        """
        # TODO: Implement forward pass (numerically stable)
        # HINT:
        # max_val = np.max(self.data, axis=axis, keepdims=True)
        # shifted = self.data - max_val
        # exp_shifted = np.exp(shifted)
        # sum_exp = np.sum(exp_shifted, axis=axis, keepdims=keepdims)
        # result = np.squeeze(max_val, axis=axis) + np.log(sum_exp) if not keepdims else max_val + np.log(sum_exp)
        out = None  # Replace: Tensor(result, (self,), 'logsumexp')
        
        # TODO: Implement backward pass
        def _backward():
            # Gradient is softmax(x) * upstream_grad
            # HINT:
            # lse = out.data if keepdims else (np.expand_dims(out.data, axis=axis) if axis is not None else out.data)
            # softmax = np.exp(self.data - lse)
            # grad = out.grad if keepdims else (np.expand_dims(out.grad, axis=axis) if axis is not None else out.grad)
            # self.grad += softmax * grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 4: Sigmoid Activation
    # ========================================================================
    
    def sigmoid(self):
        """
        Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
        
        Returns:
            Tensor with sigmoid applied
        
        Gradient: dσ/dx = σ(x) * (1 - σ(x))
        
        Numerically stable implementation:
        - For x >= 0: σ(x) = 1 / (1 + exp(-x))
        - For x < 0:  σ(x) = exp(x) / (1 + exp(x))
        """
        # TODO: Implement forward pass (numerically stable)
        # HINT:
        # positive = self.data >= 0
        # result = np.where(
        #     positive,
        #     1 / (1 + np.exp(-self.data)),
        #     np.exp(self.data) / (1 + np.exp(self.data))
        # )
        out = None  # Replace: Tensor(result, (self,), 'sigmoid')
        
        # TODO: Implement backward pass
        def _backward():
            # dσ/dx = σ(x) * (1 - σ(x)) = out.data * (1 - out.data)
            # HINT: self.grad += out.data * (1 - out.data) * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 5: Tanh Activation
    # ========================================================================
    
    def tanh(self):
        """
        Hyperbolic tangent: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        
        Returns:
            Tensor with tanh applied
        
        Gradient: d(tanh)/dx = 1 - tanh(x)²
        
        Alternative: tanh(x) = 2*sigmoid(2x) - 1
        """
        # TODO: Implement forward pass
        # HINT: result = np.tanh(self.data)
        out = None  # Replace: Tensor(result, (self,), 'tanh')
        
        # TODO: Implement backward pass
        def _backward():
            # d(tanh)/dx = 1 - tanh²(x) = 1 - out.data²
            # HINT: self.grad += (1 - out.data ** 2) * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 6: Softplus Activation
    # ========================================================================
    
    def softplus(self, beta=1.0, threshold=20.0):
        """
        Softplus: softplus(x) = (1/beta) * log(1 + exp(beta * x))
        
        Smooth approximation to ReLU.
        
        Args:
            beta: Sharpness parameter (default 1.0)
            threshold: Above this, return linear (for stability)
        
        Returns:
            Tensor with softplus applied
        
        Gradient: d(softplus)/dx = sigmoid(beta * x)
        
        Numerically stable:
        - For large x: softplus(x) ≈ x
        - For small x: Use log1p for better precision
        """
        # TODO: Implement forward pass (numerically stable)
        # HINT:
        # scaled = beta * self.data
        # result = np.where(
        #     scaled > threshold,
        #     self.data,  # Linear for large values
        #     np.log1p(np.exp(scaled)) / beta
        # )
        out = None  # Replace: Tensor(result, (self,), 'softplus')
        
        # TODO: Implement backward pass
        def _backward():
            # d(softplus)/dx = sigmoid(beta * x)
            # HINT:
            # scaled = beta * self.data
            # sigmoid_grad = 1 / (1 + np.exp(-scaled))
            # sigmoid_grad = np.where(scaled > threshold, 1.0, sigmoid_grad)
            # self.grad += sigmoid_grad * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 7: Log-Sigmoid (Numerically Stable)
    # ========================================================================
    
    def log_sigmoid(self):
        """
        Log of sigmoid: log(σ(x)) = log(1/(1+e^-x)) = -log(1+e^-x)
        
        Useful for binary cross-entropy loss.
        
        Numerically stable:
        - For x >= 0: log_sigmoid(x) = -log(1 + exp(-x))
        - For x < 0:  log_sigmoid(x) = x - log(1 + exp(x))
        
        Gradient: d(log_sigmoid)/dx = 1 - sigmoid(x) = sigmoid(-x)
        """
        # TODO: Implement forward pass (numerically stable)
        # HINT:
        # result = np.where(
        #     self.data >= 0,
        #     -np.log1p(np.exp(-self.data)),
        #     self.data - np.log1p(np.exp(self.data))
        # )
        out = None  # Replace: Tensor(result, (self,), 'log_sigmoid')
        
        # TODO: Implement backward pass
        def _backward():
            # d(log_sigmoid)/dx = sigmoid(-x) = 1 / (1 + exp(x))
            # HINT:
            # sigmoid_neg_x = np.where(
            #     self.data >= 0,
            #     np.exp(-self.data) / (1 + np.exp(-self.data)),
            #     1 / (1 + np.exp(self.data))
            # )
            # self.grad += sigmoid_neg_x * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out


# ============================================================================
# Test Functions
# ============================================================================

def test_exp():
    """Test exponential function."""
    results = {}
    
    x = Tensor([0.0, 1.0, 2.0])
    y = x.exp()
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, [1, np.e, np.e**2])
        y.backward()
        # d(e^x)/dx = e^x
        results['grad'] = np.allclose(x.grad, [1, np.e, np.e**2]) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


def test_log():
    """Test natural logarithm."""
    results = {}
    
    x = Tensor([1.0, np.e, np.e**2])
    y = x.log()
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, [0, 1, 2])
        y.backward()
        # d(ln(x))/dx = 1/x
        results['grad'] = np.allclose(x.grad, [1, 1/np.e, 1/(np.e**2)]) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


def test_exp_log_inverse():
    """Test that exp and log are inverses."""
    results = {}
    
    x = Tensor([1.0, 2.0, 3.0])
    y = x.exp().log()  # Should be back to x
    
    if y is not None and y.data is not None:
        results['exp_log'] = np.allclose(y.data, x.data)
        y.backward()
        # Chain rule: d(log(exp(x)))/dx = 1
        results['grad'] = np.allclose(x.grad, [1, 1, 1]) if x.grad is not None else False
    else:
        results['exp_log'] = False
        results['grad'] = False
    
    return results


def test_logsumexp():
    """Test logsumexp."""
    results = {}
    
    x = Tensor([1.0, 2.0, 3.0])
    y = x.logsumexp()
    
    # logsumexp([1,2,3]) = log(e + e² + e³)
    expected = np.log(np.exp(1) + np.exp(2) + np.exp(3))
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, expected)
        y.backward()
        # Gradient is softmax
        softmax = np.exp(x.data) / np.sum(np.exp(x.data))
        results['grad'] = np.allclose(x.grad, softmax) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


def test_logsumexp_stability():
    """Test logsumexp numerical stability."""
    results = {}
    
    # Large values that would overflow with naive exp
    x = Tensor([1000.0, 1001.0, 1002.0])
    y = x.logsumexp()
    
    # Should be approximately 1002 + log(e^-2 + e^-1 + 1)
    expected = 1002 + np.log(np.exp(-2) + np.exp(-1) + 1)
    
    if y is not None and y.data is not None:
        results['no_overflow'] = np.isfinite(y.data)
        results['correct'] = np.allclose(y.data, expected)
    else:
        results['no_overflow'] = False
        results['correct'] = False
    
    return results


def test_sigmoid():
    """Test sigmoid activation."""
    results = {}
    
    x = Tensor([0.0, 1.0, -1.0])
    y = x.sigmoid()
    
    expected = 1 / (1 + np.exp(-x.data))
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, expected)
        y.backward()
        # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
        expected_grad = expected * (1 - expected)
        results['grad'] = np.allclose(x.grad, expected_grad) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


def test_sigmoid_extreme():
    """Test sigmoid at extreme values."""
    results = {}
    
    x = Tensor([-100.0, 0.0, 100.0])
    y = x.sigmoid()
    
    if y is not None and y.data is not None:
        results['no_nan'] = not np.any(np.isnan(y.data))
        results['range'] = np.all((y.data >= 0) & (y.data <= 1))
        results['extreme_low'] = np.allclose(y.data[0], 0, atol=1e-10)
        results['extreme_high'] = np.allclose(y.data[2], 1, atol=1e-10)
    else:
        results['no_nan'] = False
        results['range'] = False
        results['extreme_low'] = False
        results['extreme_high'] = False
    
    return results


def test_tanh():
    """Test tanh activation."""
    results = {}
    
    x = Tensor([0.0, 1.0, -1.0])
    y = x.tanh()
    
    expected = np.tanh(x.data)
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, expected)
        y.backward()
        # d(tanh)/dx = 1 - tanh²
        expected_grad = 1 - expected ** 2
        results['grad'] = np.allclose(x.grad, expected_grad) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


def test_softplus():
    """Test softplus activation."""
    results = {}
    
    x = Tensor([-2.0, 0.0, 2.0])
    y = x.softplus()
    
    expected = np.log(1 + np.exp(x.data))
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, expected)
        y.backward()
        # d(softplus)/dx = sigmoid(x)
        expected_grad = 1 / (1 + np.exp(-x.data))
        results['grad'] = np.allclose(x.grad, expected_grad) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


def test_chain_rule():
    """Test chain rule with multiple operations."""
    results = {}
    
    # f(x) = sum(sigmoid(x)^2)
    x = Tensor([0.0, 1.0, -1.0])
    sig = x.sigmoid()
    sq = sig ** 2
    loss = sq.sum()
    
    if loss is not None:
        loss.backward()
        # d/dx = 2*sigmoid(x)*sigmoid(x)*(1-sigmoid(x))
        sig_val = 1 / (1 + np.exp(-x.data))
        expected = 2 * sig_val * sig_val * (1 - sig_val)
        results['grad'] = np.allclose(x.grad, expected) if x.grad is not None else False
    else:
        results['grad'] = False
    
    return results


if __name__ == "__main__":
    print("Day 19: Exp and Log Operations")
    print("=" * 60)
    
    print("\nExp Function:")
    exp_results = test_exp()
    for name, passed in exp_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLog Function:")
    log_results = test_log()
    for name, passed in log_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nExp-Log Inverse:")
    inv_results = test_exp_log_inverse()
    for name, passed in inv_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLogSumExp:")
    lse_results = test_logsumexp()
    for name, passed in lse_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nLogSumExp Stability:")
    stability_results = test_logsumexp_stability()
    for name, passed in stability_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSigmoid:")
    sigmoid_results = test_sigmoid()
    for name, passed in sigmoid_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSigmoid Extreme Values:")
    extreme_results = test_sigmoid_extreme()
    for name, passed in extreme_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nTanh:")
    tanh_results = test_tanh()
    for name, passed in tanh_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nSoftplus:")
    softplus_results = test_softplus()
    for name, passed in softplus_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nChain Rule:")
    chain_results = test_chain_rule()
    for name, passed in chain_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day19.py for comprehensive tests!")
