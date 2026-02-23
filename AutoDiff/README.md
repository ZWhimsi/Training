# Autodiff Track: Building an Automatic Differentiation Library

Build a PyTorch-like automatic differentiation library from scratch.

## Overview

Understanding automatic differentiation is fundamental to deep learning. This track guides you through building your own autograd engine, from basic scalar derivatives to a complete library with neural network support.

## Prerequisites

- Python 3.10+
- Calculus (derivatives, chain rule, partial derivatives)
- Linear algebra (matrices, vectors)
- Object-oriented programming

## Curriculum

### Phase 1: Mathematical Foundations (Days 1-5)

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Derivatives Review | Single variable derivatives |
| 2 | Chain Rule | Composite function derivatives |
| 3 | Partial Derivatives | Multivariable calculus |
| 4 | Computational Graphs | DAGs, nodes, edges |
| 5 | Forward vs Backward Mode | AD modes comparison |

### Phase 2: Scalar Autograd Engine (Days 6-12)

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 6 | Value Class | `data`, `grad` attributes |
| 7 | Parent Tracking | Graph construction |
| 8 | Addition & Subtraction | Basic operations |
| 9 | Multiplication & Division | More operations |
| 10 | Power & Negation | Complete arithmetic |
| 11 | Backward Pass | Chain rule implementation |
| 12 | Activation Functions | ReLU, sigmoid, tanh |

### Phase 3: Tensor Operations (Days 13-20)

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 13 | Tensor Class Design | Multi-dimensional arrays |
| 14 | Shape and Strides | Memory layout |
| 15 | Element-wise Ops | Tensor arithmetic |
| 16 | Matrix Multiplication | Matmul with gradients |
| 17 | Transpose | Gradient flow through transpose |
| 18 | Reduction Operations | Sum, mean with axis |
| 19 | Broadcasting | Shape expansion |
| 20 | Broadcasting Gradients | Gradient accumulation |

### Phase 4: Autograd Architecture (Days 21-25)

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 21 | Function Base Class | `forward()`, `backward()` |
| 22 | Context Saving | Storing values for backward |
| 23 | Graph Traversal | Topological sort |
| 24 | Multiple Paths | Gradient accumulation |
| 25 | Advanced Features | `detach()`, `requires_grad` |

### Phase 5: Neural Network Components (Days 26-30)

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 26 | Parameter Class | Learnable parameters |
| 27 | Linear Layer | Weights, bias, forward/backward |
| 28 | Activation Layers | ReLU, Sigmoid modules |
| 29 | Loss Functions | MSE, CrossEntropy |
| 30 | Module Base Class | `parameters()`, `forward()` |

### Phase 6: Complete Library (Days 31-35)

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 31 | SGD Optimizer | Parameter updates |
| 32 | Adam Optimizer | Momentum, adaptive learning |
| 33 | Sequential Container | Composing modules |
| 34 | Training Loop | Complete pipeline |
| 35 | MNIST Example | Real-world application |

## Core Concepts

### Computational Graph

```
        ┌───┐
   x ───┤ * ├───┐
        └───┘   │   ┌───┐
                ├───┤ + ├─── output
        ┌───┐   │   └───┘
   y ───┤ * ├───┘
        └───┘

Forward: Compute values from inputs to output
Backward: Compute gradients from output to inputs
```

### Chain Rule

For f(g(x)), the derivative is: df/dx = df/dg * dg/dx

In code:
```python
# Forward
g = x * 2
f = g + 3

# Backward
df_dg = 1       # derivative of f w.r.t. g
dg_dx = 2       # derivative of g w.r.t. x
df_dx = df_dg * dg_dx  # chain rule: 1 * 2 = 2
```

### Value Class Structure

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def backward(self):
        # Topological sort then reverse
        # Call _backward() on each node
        pass
```

## Building Blocks

By the end, you'll have implemented:

```python
# Tensor with autograd
x = Tensor([[1, 2], [3, 4]], requires_grad=True)
y = Tensor([[5, 6], [7, 8]], requires_grad=True)
z = x @ y  # Matrix multiply
loss = z.sum()
loss.backward()
print(x.grad)  # Gradients!

# Neural network
class MLP(Module):
    def __init__(self):
        self.fc1 = Linear(784, 128)
        self.fc2 = Linear(128, 10)
    
    def forward(self, x):
        x = relu(self.fc1(x))
        return self.fc2(x)

# Training
model = MLP()
optimizer = SGD(model.parameters(), lr=0.01)
for x, y in dataloader:
    pred = model(x)
    loss = cross_entropy(pred, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Resources

- [micrograd by Karpathy](https://github.com/karpathy/micrograd)
- [Automatic Differentiation Survey](https://arxiv.org/abs/1502.05767)
- [PyTorch Autograd Explained](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/)
- [Backpropagation Calculus (3Blue1Brown)](https://www.youtube.com/watch?v=tIeHLnjs5U8)

## Tips for Success

1. **Test incrementally**: Verify each operation before moving on
2. **Compare with PyTorch**: Use PyTorch as reference
3. **Draw graphs**: Visualize computation graphs
4. **Gradient checking**: Use numerical gradients to verify
5. **Start scalar**: Master scalars before tensors
