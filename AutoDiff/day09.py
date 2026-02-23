"""
Day 9: Activation Functions Backward Pass
==========================================
Estimated time: 1.5-2 hours
Prerequisites: Day 8 (basic backward passes)

Learning objectives:
- Implement ReLU activation with backward pass
- Implement Sigmoid activation with backward pass
- Implement Tanh activation with backward pass
- Understand derivative of activation functions
- Handle non-differentiable points (ReLU at 0)

Mathematical background:
- ReLU: f(x) = max(0, x)
         f'(x) = 1 if x > 0, else 0
         
- Sigmoid: f(x) = 1 / (1 + e^(-x))
           f'(x) = f(x) * (1 - f(x))
           
- Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        f'(x) = 1 - tanh(x)²
"""

import math


class Value:
    """Value class with activation functions and backward passes."""
    
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
    
    # Basic operations (provided)
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
    
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)
    
    # ========================================================================
    # Exercise 1: ReLU Activation
    # ========================================================================
    
    def relu(self):
        """
        Rectified Linear Unit: max(0, x)
        
        Forward: out = max(0, self.data)
        Backward: d(relu)/dx = 1 if x > 0 else 0
        
        Note: At x=0, the derivative is technically undefined, but we
        commonly use 0 for simplicity.
        
        Returns:
            Value with ReLU applied
        """
        # TODO: Implement forward pass
        # HINT: Use max(0, self.data) or (self.data if self.data > 0 else 0)
        out = None  # Replace: Value(max(0, self.data), (self,), 'relu')
        
        # TODO: Implement backward pass
        def _backward():
            # Gradient passes through unchanged if input was positive
            # Gradient is zero if input was negative or zero
            # HINT: self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
            # Alternative: self.grad += (out.data > 0) * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 2: Leaky ReLU Activation
    # ========================================================================
    
    def leaky_relu(self, alpha=0.01):
        """
        Leaky ReLU: x if x > 0 else alpha * x
        
        Forward: out = x if x > 0 else alpha * x
        Backward: d(leaky_relu)/dx = 1 if x > 0 else alpha
        
        Args:
            alpha: Slope for negative values (default 0.01)
        
        Returns:
            Value with Leaky ReLU applied
        """
        # TODO: Implement forward pass
        forward_val = None  # Replace: self.data if self.data > 0 else alpha * self.data
        out = None  # Replace: Value(forward_val, (self,), 'leaky_relu')
        
        # TODO: Implement backward pass
        def _backward():
            # HINT: grad_multiplier = 1.0 if self.data > 0 else alpha
            # HINT: self.grad += grad_multiplier * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 3: Exponential (helper for sigmoid)
    # ========================================================================
    
    def exp(self):
        """
        Exponential: e^x
        
        Forward: out = e^(self.data)
        Backward: d(e^x)/dx = e^x
        
        Returns:
            Value with exp applied
        """
        # TODO: Implement forward pass
        out = None  # Replace: Value(math.exp(self.data), (self,), 'exp')
        
        # TODO: Implement backward pass
        def _backward():
            # The derivative of e^x is e^x = out.data
            # HINT: self.grad += out.data * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 4: Sigmoid Activation
    # ========================================================================
    
    def sigmoid(self):
        """
        Sigmoid: 1 / (1 + e^(-x))
        
        Forward: out = 1 / (1 + e^(-self.data))
        Backward: d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))
        
        Note: This elegant derivative makes sigmoid popular in neural networks!
        
        Returns:
            Value with sigmoid applied
        """
        # TODO: Implement forward pass
        # HINT: s = 1 / (1 + math.exp(-self.data))
        s = None  # Replace with sigmoid calculation
        out = None  # Replace: Value(s, (self,), 'sigmoid')
        
        # TODO: Implement backward pass
        def _backward():
            # The derivative is sigmoid * (1 - sigmoid) = out.data * (1 - out.data)
            # HINT: self.grad += out.data * (1 - out.data) * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 5: Tanh Activation
    # ========================================================================
    
    def tanh(self):
        """
        Hyperbolic tangent: (e^x - e^(-x)) / (e^x + e^(-x))
        
        Forward: out = tanh(self.data)
        Backward: d(tanh)/dx = 1 - tanh(x)²
        
        Returns:
            Value with tanh applied
        """
        # TODO: Implement forward pass
        # HINT: Use math.tanh or compute from exponentials
        t = None  # Replace: math.tanh(self.data)
        out = None  # Replace: Value(t, (self,), 'tanh')
        
        # TODO: Implement backward pass
        def _backward():
            # The derivative is 1 - tanh² = 1 - out.data²
            # HINT: self.grad += (1 - out.data ** 2) * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out
    
    # ========================================================================
    # Exercise 6: Softplus Activation (Bonus)
    # ========================================================================
    
    def softplus(self):
        """
        Softplus: log(1 + e^x) - a smooth approximation to ReLU
        
        Forward: out = log(1 + e^x)
        Backward: d(softplus)/dx = e^x / (1 + e^x) = sigmoid(x)
        
        Note: For numerical stability, use log(1 + e^x) directly for small x,
        and x + log(1 + e^(-x)) for large x. We'll keep it simple here.
        
        Returns:
            Value with softplus applied
        """
        # TODO: Implement forward pass
        # HINT: Use math.log(1 + math.exp(self.data))
        out = None  # Replace: Value(math.log(1 + math.exp(self.data)), (self,), 'softplus')
        
        # TODO: Implement backward pass
        def _backward():
            # derivative = sigmoid(x) = 1 / (1 + e^(-x))
            # HINT: sigmoid_val = 1 / (1 + math.exp(-self.data))
            # HINT: self.grad += sigmoid_val * out.grad
            pass  # Replace
        
        out._backward = _backward
        return out


# ============================================================================
# Exercise 7: Compare Activations
# ============================================================================

def compare_activations(x_val):
    """
    Compare different activation functions at a given input.
    
    Args:
        x_val: Input value to test
    
    Returns:
        dict with activation values and gradients
    """
    results = {}
    
    # TODO: Test ReLU
    x_relu = Value(x_val)
    y_relu = x_relu.relu() if hasattr(Value, 'relu') else None
    if y_relu:
        y_relu.backward()
        results['relu'] = {'value': y_relu.data, 'grad': x_relu.grad}
    
    # TODO: Test Sigmoid
    x_sig = Value(x_val)
    y_sig = x_sig.sigmoid() if hasattr(Value, 'sigmoid') else None
    if y_sig:
        y_sig.backward()
        results['sigmoid'] = {'value': y_sig.data, 'grad': x_sig.grad}
    
    # TODO: Test Tanh
    x_tanh = Value(x_val)
    y_tanh = x_tanh.tanh() if hasattr(Value, 'tanh') else None
    if y_tanh:
        y_tanh.backward()
        results['tanh'] = {'value': y_tanh.data, 'grad': x_tanh.grad}
    
    return results


# ============================================================================
# Numerical Gradient Checking
# ============================================================================

def numerical_gradient(f, x, h=1e-5):
    """Compute numerical gradient using central difference."""
    return (f(x + h) - f(x - h)) / (2 * h)


def check_activation_gradient(activation_name, x_val):
    """
    Verify analytical gradient matches numerical gradient.
    
    Args:
        activation_name: 'relu', 'sigmoid', or 'tanh'
        x_val: Input value
    
    Returns:
        (analytical, numerical, matches)
    """
    x = Value(x_val)
    
    if activation_name == 'relu':
        y = x.relu()
        def f(v): return max(0, v)
    elif activation_name == 'sigmoid':
        y = x.sigmoid()
        def f(v): return 1 / (1 + math.exp(-v))
    elif activation_name == 'tanh':
        y = x.tanh()
        def f(v): return math.tanh(v)
    else:
        return None, None, False
    
    y.backward()
    analytical = x.grad
    numerical = numerical_gradient(f, x_val)
    
    matches = abs(analytical - numerical) < 1e-4
    return analytical, numerical, matches


if __name__ == "__main__":
    print("Day 9: Activation Functions Backward")
    print("=" * 50)
    
    test_values = [-2.0, -0.5, 0.0, 0.5, 2.0]
    
    for x_val in test_values:
        print(f"\nInput x = {x_val}")
        print("-" * 30)
        
        results = compare_activations(x_val)
        
        for name, data in results.items():
            if data:
                print(f"  {name:10s}: value={data['value']:.4f}, grad={data['grad']:.4f}")
    
    print("\n" + "=" * 50)
    print("Numerical Gradient Check")
    print("=" * 50)
    
    for activation in ['relu', 'sigmoid', 'tanh']:
        x_val = 0.5
        try:
            analytical, numerical, matches = check_activation_gradient(activation, x_val)
            status = "✓" if matches else "✗"
            print(f"  {activation:10s}: analytical={analytical:.4f}, numerical={numerical:.4f} {status}")
        except:
            print(f"  {activation:10s}: Not implemented")
    
    print("\nRun test_day09.py to verify your implementation!")
