"""
Day 14: Tensor Addition and Multiplication with Gradients
=========================================================
Estimated time: 2-3 hours
Prerequisites: Days 12-13 (Tensor class, broadcasting)

Learning objectives:
- Implement complete element-wise operations with gradients
- Handle subtraction and division for tensors
- Implement power operations with gradients
- Build complex expressions and verify gradients

Key operations:
- Addition: z = x + y, dz/dx = 1, dz/dy = 1
- Subtraction: z = x - y, dz/dx = 1, dz/dy = -1
- Multiplication: z = x * y, dz/dx = y, dz/dy = x
- Division: z = x / y, dz/dx = 1/y, dz/dy = -x/y²
- Power: z = x^n, dz/dx = n * x^(n-1)

All operations must handle broadcasting correctly!
"""

import numpy as np
from typing import Tuple


class Tensor:
    """Tensor class with complete arithmetic operations."""
    
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
        """Reduce gradient to original shape by summing along broadcast dimensions."""
        while grad.ndim > len(original_shape):
            grad = grad.sum(axis=0)
        
        for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
            if orig_dim == 1 and grad_dim > 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad
    
    # ========================================================================
    # Exercise 1: Addition (complete with broadcasting)
    # ========================================================================
    
    def __add__(self, other):
        """
        Element-wise addition: z = self + other
        
        Gradient: dz/d(self) = 1, dz/d(other) = 1
        """
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
    
    # ========================================================================
    # Exercise 2: Negation
    # ========================================================================
    
    def __neg__(self):
        """
        Negation: z = -self
        
        Gradient: dz/d(self) = -1
        """
        # API hints:
        # - -self.data -> negate data
        # - Tensor(data, children, op) -> create output
        # - Gradient of negation is -1
        
        # TODO: Implement negation
        out = None  # Replace
        
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 3: Subtraction
    # ========================================================================
    
    def __sub__(self, other):
        """
        Subtraction: z = self - other
        
        Gradient: dz/d(self) = 1, dz/d(other) = -1
        """
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        
        # API hints:
        # - self.data - other.data -> element-wise subtraction
        # - Tensor(data, children, op) -> create output
        # - dz/d(self) = 1, dz/d(other) = -1
        # - Tensor.unbroadcast(grad, shape) -> handle broadcasting
        
        # TODO: Implement subtraction forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        """Handle other - self when other is not a Tensor."""
        return Tensor(np.array(other)) - self
    
    # ========================================================================
    # Exercise 4: Multiplication (complete with broadcasting)
    # ========================================================================
    
    def __mul__(self, other):
        """
        Element-wise multiplication: z = self * other
        
        Gradient: dz/d(self) = other, dz/d(other) = self
        """
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            self_grad = other.data * out.grad
            other_grad = self.data * out.grad
            self.grad += Tensor.unbroadcast(self_grad, self.shape)
            other.grad += Tensor.unbroadcast(other_grad, other.shape)
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    # ========================================================================
    # Exercise 5: Division
    # ========================================================================
    
    def __truediv__(self, other):
        """
        Element-wise division: z = self / other
        
        Gradient: dz/d(self) = 1/other
                  dz/d(other) = -self/other²
        """
        if isinstance(other, (int, float)):
            other = Tensor(np.array(other))
        
        # API hints:
        # - self.data / other.data -> element-wise division
        # - dz/d(self) = 1/other
        # - dz/d(other) = -self/other^2
        # - Tensor.unbroadcast(grad, shape) -> handle broadcasting
        
        # TODO: Implement division forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        """Handle other / self when other is not a Tensor."""
        return Tensor(np.array(other)) / self
    
    # ========================================================================
    # Exercise 6: Power (element-wise)
    # ========================================================================
    
    def __pow__(self, n):
        """
        Element-wise power: z = self^n (n is a scalar)
        
        Gradient: dz/d(self) = n * self^(n-1)
        
        Note: n must be a number, not a Tensor
        """
        assert isinstance(n, (int, float)), "Exponent must be a scalar"
        
        # API hints:
        # - self.data ** n -> element-wise power
        # - Power rule: d/dx(x^n) = n * x^(n-1)
        # - self.grad += local_grad * out.grad
        
        # TODO: Implement power forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 7: Element-wise exp
    # ========================================================================
    
    def exp(self):
        """
        Element-wise exponential: z = e^self
        
        Gradient: dz/d(self) = e^self = z
        """
        # API hints:
        # - np.exp(self.data) -> element-wise exponential
        # - d/dx(e^x) = e^x = out.data
        # - self.grad += local_grad * out.grad
        
        # TODO: Implement exp forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 8: Element-wise log
    # ========================================================================
    
    def log(self):
        """
        Element-wise natural logarithm: z = ln(self)
        
        Gradient: dz/d(self) = 1/self
        
        Note: Only valid for positive inputs!
        """
        # API hints:
        # - np.log(self.data) -> element-wise natural log
        # - d/dx(ln(x)) = 1/x
        # - self.grad += local_grad * out.grad
        
        # TODO: Implement log forward pass
        out = None  # Replace
        
        # TODO: Implement backward pass
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out


# ============================================================================
# Exercise 9: Build and Test Expressions
# ============================================================================

def test_subtraction():
    """Test subtraction operation and gradients."""
    results = {}
    
    # Basic subtraction
    a = Tensor([5.0, 6.0, 7.0])
    b = Tensor([1.0, 2.0, 3.0])
    c = a - b
    
    if c is not None and c.data is not None:
        results['forward'] = np.allclose(c.data, [4, 4, 4])
        c.backward()
        results['grad_a'] = np.allclose(a.grad, [1, 1, 1]) if a.grad is not None else False
        results['grad_b'] = np.allclose(b.grad, [-1, -1, -1]) if b.grad is not None else False
    else:
        results['forward'] = False
        results['grad_a'] = False
        results['grad_b'] = False
    
    return results


def test_division():
    """Test division operation and gradients."""
    results = {}
    
    # Division: a / b at a=6, b=2
    a = Tensor([6.0])
    b = Tensor([2.0])
    c = a / b  # = 3
    
    if c is not None and c.data is not None:
        results['forward'] = np.allclose(c.data, [3.0])
        c.backward()
        # dc/da = 1/b = 0.5
        # dc/db = -a/b² = -6/4 = -1.5
        results['grad_a'] = np.allclose(a.grad, [0.5]) if a.grad is not None else False
        results['grad_b'] = np.allclose(b.grad, [-1.5]) if b.grad is not None else False
    else:
        results['forward'] = False
        results['grad_a'] = False
        results['grad_b'] = False
    
    return results


def test_power():
    """Test power operation and gradients."""
    results = {}
    
    # x^2 at x=[2, 3]
    x = Tensor([2.0, 3.0])
    y = x ** 2  # [4, 9]
    
    if y is not None and y.data is not None:
        results['forward'] = np.allclose(y.data, [4, 9])
        y.backward()
        # dy/dx = 2x = [4, 6]
        results['grad'] = np.allclose(x.grad, [4, 6]) if x.grad is not None else False
    else:
        results['forward'] = False
        results['grad'] = False
    
    return results


def test_exp_log():
    """Test exp and log operations."""
    results = {}
    
    # exp
    x = Tensor([0.0, 1.0])
    y = x.exp()  # [1, e]
    
    if y is not None and y.data is not None:
        results['exp_forward'] = np.allclose(y.data, [1, np.e])
        y.backward()
        # d(e^x)/dx = e^x
        results['exp_grad'] = np.allclose(x.grad, [1, np.e]) if x.grad is not None else False
    else:
        results['exp_forward'] = False
        results['exp_grad'] = False
    
    # log
    x2 = Tensor([1.0, np.e])
    y2 = x2.log()  # [0, 1]
    
    if y2 is not None and y2.data is not None:
        results['log_forward'] = np.allclose(y2.data, [0, 1])
        y2.backward()
        # d(ln(x))/dx = 1/x
        results['log_grad'] = np.allclose(x2.grad, [1, 1/np.e]) if x2.grad is not None else False
    else:
        results['log_forward'] = False
        results['log_grad'] = False
    
    return results


def test_complex_expression():
    """Test a complex expression combining multiple operations."""
    results = {}
    
    # f(x, y) = (x + y) * (x - y) = x² - y²
    # at x=3, y=2: f = 9 - 4 = 5
    # df/dx = 2x = 6, df/dy = -2y = -4
    
    x = Tensor([3.0])
    y = Tensor([2.0])
    
    sum_xy = x + y      # 5
    diff_xy = x - y     # 1
    f = sum_xy * diff_xy  # 5
    
    if f is not None and f.data is not None:
        results['forward'] = np.allclose(f.data, [5.0])
        f.backward()
        results['grad_x'] = np.allclose(x.grad, [6.0]) if x.grad is not None else False
        results['grad_y'] = np.allclose(y.grad, [-4.0]) if y.grad is not None else False
    else:
        results['forward'] = False
        results['grad_x'] = False
        results['grad_y'] = False
    
    return results


def test_broadcast_division():
    """Test division with broadcasting."""
    results = {}
    
    # (2, 3) / (3,) - row broadcast
    a = Tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])  # (2, 3)
    b = Tensor([2.0, 2.0, 2.0])                        # (3,)
    c = a / b
    
    if c is not None and c.data is not None:
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        results['forward'] = np.allclose(c.data, expected)
        c.backward()
        
        # dc/da = 1/b = [0.5, 0.5, 0.5] broadcast
        results['grad_a'] = np.allclose(a.grad, 0.5 * np.ones((2, 3))) if a.grad is not None else False
        
        # dc/db = -a/b² summed over axis 0
        expected_b_grad = np.array([-2.5, -3.5, -4.5])  # Sum of -a/4
        results['grad_b'] = np.allclose(b.grad, expected_b_grad) if b.grad is not None else False
    else:
        results['forward'] = False
        results['grad_a'] = False
        results['grad_b'] = False
    
    return results


if __name__ == "__main__":
    print("Day 14: Tensor Addition/Multiplication with Gradients")
    print("=" * 60)
    
    print("\nSubtraction Tests:")
    sub_results = test_subtraction()
    for name, passed in sub_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nDivision Tests:")
    div_results = test_division()
    for name, passed in div_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nPower Tests:")
    pow_results = test_power()
    for name, passed in pow_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nExp/Log Tests:")
    exp_log_results = test_exp_log()
    for name, passed in exp_log_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nComplex Expression Tests:")
    complex_results = test_complex_expression()
    for name, passed in complex_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nBroadcast Division Tests:")
    broadcast_results = test_broadcast_division()
    for name, passed in broadcast_results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    
    print("\nRun test_day14.py for comprehensive tests!")
