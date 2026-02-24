"""
Day 11: Backward Pass - Automatic Gradient Computation
=====================================================
Estimated time: 1-2 hours
Prerequisites: Days 6-10 (Value class, operations)

Learning objectives:
- Implement the backward() method
- Understand topological sorting for gradient flow
- Accumulate gradients through the computation graph
- Test gradient computation against numerical gradients

This is the heart of automatic differentiation!
"""

import math


class Value:
    """Value class with backward pass."""
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    # ========================================================================
    # Exercise 1: Implement backward()
    # ========================================================================
    
    def backward(self):
        """
        Compute gradients for all Values in the computation graph.
        
        Steps:
        1. Build topological ordering of all nodes
        2. Set gradient of output to 1.0
        3. Call _backward() on each node in reverse topological order
        """
        # API hints:
        # - Topological sort: visit children before parents (DFS with post-order)
        # - Use a visited set to avoid cycles
        # - self.grad = 1.0 for the output node
        # - reversed(list) -> iterate in reverse order
        # - v._backward() -> call the backward function for each node
        
        # TODO: Build topological order using DFS
        topo = []
        visited = set()
        
        def build_topo(v):
            """Recursively build topological order."""
            pass  # Implement DFS traversal
        
        # TODO: Build topo starting from self
        pass  # Replace
        
        # TODO: Set gradient of output (self) to 1.0
        pass  # Replace
        
        # TODO: Call _backward on each node in reverse order
        pass  # Replace
    
    # ========================================================================
    # Operations with backward functions
    # ========================================================================
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return Value(other) / self
    
    # ========================================================================
    # Exercise 2: Implement ReLU
    # ========================================================================
    
    def relu(self):
        """
        ReLU activation: max(0, x)
        
        Backward: d/dx = 1 if x > 0 else 0
        """
        # API hints:
        # - max(0, value) -> forward pass
        # - Value(data, children, op) -> create output node
        # - Gradient flows only where input > 0
        # - self.grad += local_grad * out.grad
        
        # TODO: Implement forward
        out = None  # Replace
        
        # TODO: Implement backward
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 3: Implement exp
    # ========================================================================
    
    def exp(self):
        """
        Exponential: e^x
        
        Backward: d/dx(e^x) = e^x
        """
        # API hints:
        # - math.exp(x) -> compute e^x
        # - Value(data, children, op) -> create output node
        # - Derivative of exp(x) equals exp(x) itself
        # - self.grad += local_grad * out.grad
        
        # TODO: Implement forward
        out = None  # Replace
        
        # TODO: Implement backward
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 4: Implement tanh
    # ========================================================================
    
    def tanh(self):
        """
        Hyperbolic tangent: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        
        Backward: d/dx(tanh(x)) = 1 - tanh(x)^2
        """
        # API hints:
        # - Formula: tanh(x) = (e^2x - 1) / (e^2x + 1)
        # - math.exp(2 * x) -> compute e^(2x)
        # - Derivative: 1 - tanh(x)^2 = 1 - out.data^2
        # - self.grad += local_grad * out.grad
        
        t = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        out = Value(t, (self,), 'tanh')
        
        # TODO: Implement backward
        def _backward():
            pass  # Replace
        
        out._backward = _backward
        return out


# ============================================================================
# Exercise 5: Gradient Checking
# ============================================================================

def numerical_gradient(f, x, h=1e-5):
    """Compute numerical gradient using central difference."""
    return (f(x + h) - f(x - h)) / (2 * h)


def gradient_check(x_val, expression_fn, h=1e-5):
    """
    Check if analytical gradient matches numerical gradient.
    
    Args:
        x_val: The value at which to check
        expression_fn: Function that takes a Value and returns a Value
        h: Step size for numerical gradient
    
    Returns:
        (analytical_grad, numerical_grad, matches)
    """
    # Analytical gradient via backprop
    x = Value(x_val)
    y = expression_fn(x)
    y.backward()
    analytical = x.grad
    
    # Numerical gradient
    def f(val):
        return expression_fn(Value(val)).data
    
    numerical = numerical_gradient(f, x_val, h)
    
    matches = abs(analytical - numerical) < 1e-4
    
    return analytical, numerical, matches


if __name__ == "__main__":
    print("Day 11: Backward Pass")
    print("=" * 50)
    
    # Test backward
    print("\nTest: x^2 at x=3, gradient should be 6")
    x = Value(3.0)
    y = x ** 2
    y.backward()
    print(f"  x.grad = {x.grad} (expected: 6.0)")
    
    # Test chain rule
    print("\nTest: (x * 2 + 1)^2 at x=2")
    x = Value(2.0)
    y = (x * 2 + 1) ** 2  # (2*2+1)^2 = 25
    y.backward()
    # dy/dx = 2 * (2x + 1) * 2 = 4 * (2*2+1) = 20
    print(f"  y.data = {y.data} (expected: 25)")
    print(f"  x.grad = {x.grad} (expected: 20)")
    
    print("\nRun test_day11.py for comprehensive tests!")
