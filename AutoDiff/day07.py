"""
Day 7: Parent Tracking in the Value Class
=========================================
Estimated time: 1-2 hours
Prerequisites: Day 6 (Value class basics)

Learning objectives:
- Track parent nodes in computations
- Build computational graphs automatically
- Understand how operations create new nodes
- Prepare for backpropagation
"""

import math


class Value:
    """Value class with proper parent tracking."""
    
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    # ========================================================================
    # Exercise 1: Trace the Computation
    # ========================================================================
    
    def get_parents(self):
        """
        Return the immediate parent nodes.
        
        Returns:
            set of Value objects that are parents
        """
        # TODO: Return the _prev set
        return None  # Replace: self._prev
    
    def get_operation(self):
        """
        Return the operation that created this node.
        
        Returns:
            string describing the operation (e.g., '+', '*')
        """
        # TODO: Return the _op string
        return None  # Replace: self._op
    
    # ========================================================================
    # Exercise 2: Check if Leaf Node
    # ========================================================================
    
    def is_leaf(self):
        """
        Check if this is a leaf node (no parents).
        
        Leaf nodes are inputs, not results of operations.
        
        Returns:
            True if leaf, False otherwise
        """
        # TODO: Return True if no parents
        return None  # Replace: len(self._prev) == 0
    
    # ========================================================================
    # Exercise 3: Get All Ancestors
    # ========================================================================
    
    def get_all_ancestors(self):
        """
        Get all ancestor nodes (parents, grandparents, etc.).
        
        Returns:
            set of all Value objects in the computation graph
        """
        ancestors = set()
        
        def collect(node):
            for parent in node._prev:
                if parent not in ancestors:
                    ancestors.add(parent)
                    collect(parent)
        
        # TODO: Collect all ancestors
        collect(self)
        
        return ancestors
    
    # ========================================================================
    # Operations that create parent relationships
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
    
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)


# ============================================================================
# Exercise 4: Build and Inspect a Graph
# ============================================================================

def build_expression():
    """
    Build f(x, y) = x*y + x^2 and return all nodes.
    
    Returns:
        dict with 'x', 'y', 'xy', 'x_squared', 'result'
    """
    x = Value(2.0)
    y = Value(3.0)
    
    # TODO: Build the expression
    xy = None      # Replace: x * y
    x_squared = None  # Replace: x ** 2
    result = None  # Replace: xy + x_squared
    
    return {
        'x': x,
        'y': y,
        'xy': xy,
        'x_squared': x_squared,
        'result': result
    }


# ============================================================================
# Exercise 5: Verify Parent Relationships
# ============================================================================

def verify_parents(nodes):
    """
    Verify the parent relationships are correct.
    
    Args:
        nodes: dict from build_expression()
    
    Returns:
        dict with verification results
    """
    results = {}
    
    # TODO: Check that x and y are leaf nodes
    results['x_is_leaf'] = None  # Replace: nodes['x'].is_leaf()
    results['y_is_leaf'] = None  # Replace: nodes['y'].is_leaf()
    
    # TODO: Check that result has xy and x_squared as parents
    if nodes['result'] is not None:
        parents = nodes['result'].get_parents()
        results['result_parent_count'] = len(parents) if parents else 0
    
    # TODO: Check operations
    if nodes['xy'] is not None:
        results['xy_operation'] = nodes['xy'].get_operation()
    
    return results


# ============================================================================
# Exercise 6: Count Nodes in Graph
# ============================================================================

def count_graph_nodes(output):
    """
    Count total nodes in the computation graph.
    
    Args:
        output: The output Value node
    
    Returns:
        Total number of nodes (including output)
    """
    if output is None:
        return 0
    
    # TODO: Count output + all ancestors
    ancestors = output.get_all_ancestors()
    
    return None  # Replace: 1 + len(ancestors) if ancestors else 1


if __name__ == "__main__":
    print("Day 7: Parent Tracking")
    print("=" * 50)
    
    nodes = build_expression()
    if nodes['result'] is not None:
        print(f"\nResult: {nodes['result']}")
        print(f"Is leaf: {nodes['result'].is_leaf()}")
        print(f"Parents: {nodes['result'].get_parents()}")
        print(f"Operation: {nodes['result'].get_operation()}")
    
    print("\nRun test_day07.py to verify!")
