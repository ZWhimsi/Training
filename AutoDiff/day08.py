"""
Day 8: Division and Subtraction Backward Pass
=============================================
Estimated time: 1-2 hours
Prerequisites: Day 7 (Parent tracking, basic operations)

Learning objectives:
- Implement backward pass for subtraction operation
- Implement backward pass for division operation
- Understand quotient rule for derivatives
- Handle edge cases (division by zero)

Mathematical background:
- Subtraction: d/dx(a - b) = 1 for a, -1 for b
- Division: d/da(a / b) = 1/b, d/db(a / b) = -a/b²
"""

import math


class Value:
    """Value class with division and subtraction backward passes."""
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    def backward(self):
        """Compute gradients via backpropagation."""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        for v in reversed(topo):
            v._backward()
    
    # ========================================================================
    # Addition (provided for reference)
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
    
    # ========================================================================
    # Multiplication (provided for reference)
    # ========================================================================
    
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
    
    # ========================================================================
    # Exercise 1: Negation with Backward Pass
    # ========================================================================
    
    def __neg__(self):
        """
        Negate the value: -x
        
        Forward: out = -self.data
        Backward: d(-x)/dx = -1
        
        Returns:
            Value representing -self
        """
        # API hints:
        # - Forward: Value(-self.data, (self,), 'neg')
        # - Backward: d(-x)/dx = -1, so self.grad += -1 * out.grad
        out = None
        
        def _backward():
            pass
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 2: Subtraction with Backward Pass
    # ========================================================================
    
    def __sub__(self, other):
        """
        Subtract: a - b
        
        Forward: out = self.data - other.data
        Backward: d(a-b)/da = 1, d(a-b)/db = -1
        
        Returns:
            Value representing self - other
        """
        other = other if isinstance(other, Value) else Value(other)
        
        # API hints:
        # - Forward: Value(self.data - other.data, (self, other), '-')
        # - Backward: self.grad += 1 * out.grad, other.grad += -1 * out.grad
        out = None
        
        def _backward():
            pass
        
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        """Handle other - self when other is not a Value."""
        # API hints:
        # - Convert other to Value and subtract: Value(other) - self
        return None
    
    # ========================================================================
    # Exercise 3: Power with Backward Pass (for division)
    # ========================================================================
    
    def __pow__(self, n):
        """
        Power: x^n where n is a constant
        
        Forward: out = self.data ** n
        Backward: d(x^n)/dx = n * x^(n-1)
        
        This is needed for division since a/b = a * b^(-1)
        """
        out = Value(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            # Power rule: d/dx(x^n) = n * x^(n-1)
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 4: Division with Backward Pass
    # ========================================================================
    
    def __truediv__(self, other):
        """
        Divide: a / b
        
        Forward: out = self.data / other.data
        Backward using quotient rule:
            d(a/b)/da = 1/b
            d(a/b)/db = -a/b²
        
        Returns:
            Value representing self / other
        """
        other = other if isinstance(other, Value) else Value(other)
        
        # API hints:
        # - Forward: Value(self.data / other.data, (self, other), '/')
        # - Backward: self.grad += (1/other.data) * out.grad
        # - Backward: other.grad += (-self.data / other.data**2) * out.grad
        out = None
        
        def _backward():
            pass
        
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        """Handle other / self when other is not a Value."""
        # API hints:
        # - Convert other to Value and divide: Value(other) / self
        return None


# ============================================================================
# Exercise 5: Verify Subtraction Gradients
# ============================================================================

def verify_subtraction_gradients():
    """
    Test that subtraction gradients are correct.
    
    For f(a, b) = a - b:
        df/da = 1
        df/db = -1
    
    Returns:
        dict with 'a_grad', 'b_grad', 'a_correct', 'b_correct'
    """
    a = Value(5.0)
    b = Value(3.0)
    
    # API hints:
    # - Compute c = a - b
    # - Call c.backward() to compute gradients
    # - Check: a.grad should be 1.0, b.grad should be -1.0
    c = None
    if c is not None:
        c.backward()
    
    return {
        'a_grad': a.grad,
        'b_grad': b.grad,
        'a_correct': None,
        'b_correct': None
    }


# ============================================================================
# Exercise 6: Verify Division Gradients
# ============================================================================

def verify_division_gradients():
    """
    Test that division gradients are correct.
    
    For f(a, b) = a / b:
        df/da = 1/b
        df/db = -a/b²
    
    Returns:
        dict with results
    """
    a = Value(6.0)
    b = Value(2.0)
    
    # API hints:
    # - Compute c = a / b
    # - Call c.backward() to compute gradients
    # - Expected: a.grad = 1/b = 0.5, b.grad = -a/b² = -1.5
    c = None
    if c is not None:
        c.backward()
    
    expected_a_grad = 0.5
    expected_b_grad = -1.5
    
    return {
        'result': c.data if c else None,
        'a_grad': a.grad,
        'b_grad': b.grad,
        'a_correct': None,
        'b_correct': None
    }


# ============================================================================
# Exercise 7: Chain Rule with Division
# ============================================================================

def chain_rule_division():
    """
    Test chain rule with division.
    
    f(x) = (x + 1) / (x - 1) at x = 3
    
    Returns:
        dict with result and gradient
    """
    x = Value(3.0)
    
    # API hints:
    # - Build numerator = x + 1
    # - Build denominator = x - 1
    # - Compute f = numerator / denominator
    # - Call f.backward() to get gradient
    # - Expected: value = 2.0, gradient = -0.5
    numerator = None
    denominator = None
    f = None
    
    if f is not None:
        f.backward()
    
    expected_value = 2.0
    expected_grad = -0.5
    
    return {
        'value': f.data if f else None,
        'gradient': x.grad,
        'value_correct': None,
        'gradient_correct': None
    }


if __name__ == "__main__":
    print("Day 8: Division and Subtraction Backward")
    print("=" * 50)
    
    # Test subtraction
    print("\nTest: Subtraction gradients")
    sub_result = verify_subtraction_gradients()
    print(f"  a.grad = {sub_result['a_grad']} (expected: 1.0)")
    print(f"  b.grad = {sub_result['b_grad']} (expected: -1.0)")
    
    # Test division
    print("\nTest: Division gradients")
    div_result = verify_division_gradients()
    print(f"  6/2 = {div_result['result']}")
    print(f"  a.grad = {div_result['a_grad']} (expected: 0.5)")
    print(f"  b.grad = {div_result['b_grad']} (expected: -1.5)")
    
    # Test chain rule
    print("\nTest: Chain rule with division")
    chain_result = chain_rule_division()
    print(f"  (x+1)/(x-1) at x=3 = {chain_result['value']}")
    print(f"  gradient = {chain_result['gradient']} (expected: -0.5)")
    
    print("\nRun test_day08.py to verify your implementation!")
