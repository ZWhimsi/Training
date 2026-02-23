"""
Day 6: The Value Class - Foundation of Autograd
===============================================
Estimated time: 1-2 hours
Prerequisites: Days 1-5 (derivatives, chain rule, graphs)

Learning objectives:
- Implement a Value class that tracks computation
- Store data, gradients, and parent relationships
- Build the foundation for automatic differentiation

This is the core of building your own PyTorch-like autograd!
"""

import math


# ============================================================================
# The Value Class
# ============================================================================

class Value:
    """
    A Value wraps a number and tracks its gradient.
    
    This is inspired by Andrej Karpathy's micrograd.
    """
    
    def __init__(self, data, _children=(), _op=''):
        """
        Initialize a Value.
        
        Args:
            data: The actual numerical value
            _children: Tuple of Values that produced this Value
            _op: String describing the operation that produced this
        """
        # TODO: Store the numerical data
        self.data = None  # Replace with: data
        
        # TODO: Initialize gradient to 0
        self.grad = None  # Replace with: 0.0
        
        # TODO: Store the backward function (will be set by operations)
        self._backward = lambda: None
        
        # TODO: Store children (parents in the computational graph)
        self._prev = None  # Replace with: set(_children)
        
        # TODO: Store the operation name (for debugging)
        self._op = None  # Replace with: _op
    
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    # ========================================================================
    # Exercise 1: Addition
    # ========================================================================
    
    def __add__(self, other):
        """
        Add two Values: self + other
        
        Forward: out = self + other
        Backward: d(out)/d(self) = 1, d(out)/d(other) = 1
        """
        # Handle adding with a plain number
        other = other if isinstance(other, Value) else Value(other)
        
        # TODO: Create output Value
        out = None  # Replace with: Value(self.data + other.data, (self, other), '+')
        
        # TODO: Define backward function
        def _backward():
            # Gradient flows equally to both inputs
            # HINT: self.grad += out.grad
            # HINT: other.grad += out.grad
            pass  # Replace with gradient updates
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        """Handle other + self (when other is not a Value)."""
        return self + other
    
    # ========================================================================
    # Exercise 2: Multiplication
    # ========================================================================
    
    def __mul__(self, other):
        """
        Multiply two Values: self * other
        
        Forward: out = self * other
        Backward: d(out)/d(self) = other, d(out)/d(other) = self
        """
        other = other if isinstance(other, Value) else Value(other)
        
        # TODO: Create output Value
        out = None  # Replace with: Value(self.data * other.data, (self, other), '*')
        
        # TODO: Define backward function
        def _backward():
            # Chain rule: multiply by other's value for self's grad
            # HINT: self.grad += other.data * out.grad
            # HINT: other.grad += self.data * out.grad
            pass  # Replace with gradient updates
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    # ========================================================================
    # Exercise 3: Negation and Subtraction
    # ========================================================================
    
    def __neg__(self):
        """Negate: -self"""
        # TODO: Implement using multiplication by -1
        return None  # Replace with: self * -1
    
    def __sub__(self, other):
        """Subtract: self - other"""
        # TODO: Implement using addition and negation
        return None  # Replace with: self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    # ========================================================================
    # Exercise 4: Power
    # ========================================================================
    
    def __pow__(self, n):
        """
        Power: self ** n (n is a constant, not a Value)
        
        Forward: out = self^n
        Backward: d(out)/d(self) = n * self^(n-1)
        """
        assert isinstance(n, (int, float)), "Only constant powers supported"
        
        # TODO: Create output Value
        out = None  # Replace with: Value(self.data ** n, (self,), f'**{n}')
        
        # TODO: Define backward function
        def _backward():
            # Power rule: d/dx(x^n) = n * x^(n-1)
            # HINT: self.grad += n * (self.data ** (n - 1)) * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 5: Division
    # ========================================================================
    
    def __truediv__(self, other):
        """Divide: self / other"""
        # TODO: Implement using multiplication and power
        # HINT: a / b = a * b^(-1)
        return None  # Replace with: self * (other ** -1)
    
    def __rtruediv__(self, other):
        return Value(other) / self


# ============================================================================
# Testing Your Implementation
# ============================================================================

def test_value_basic():
    """Test basic Value operations."""
    # Test creation
    a = Value(2.0)
    b = Value(3.0)
    
    # Test addition
    c = a + b
    assert c.data == 5.0, f"Addition failed: {c.data}"
    
    # Test multiplication
    d = a * b
    assert d.data == 6.0, f"Multiplication failed: {d.data}"
    
    # Test power
    e = a ** 2
    assert e.data == 4.0, f"Power failed: {e.data}"
    
    # Test with scalar
    f = a + 1
    assert f.data == 3.0, f"Scalar add failed: {f.data}"
    
    print("All basic tests passed!")


if __name__ == "__main__":
    print("Day 6: The Value Class")
    print("=" * 50)
    
    print("\nTesting basic operations...")
    try:
        test_value_basic()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nRun test_day06.py for comprehensive tests!")
