"""
Day 4: Computational Graphs
===========================
Estimated time: 1-2 hours
Prerequisites: Day 3 (gradients)

Learning objectives:
- Understand computational graphs (DAGs)
- Represent computations as nodes and edges
- Trace forward computations
- Understand why graphs are useful for autodiff
"""

# ============================================================================
# CONCEPT: Computational Graphs
# ============================================================================
# A computational graph represents a computation as a directed acyclic graph:
# - Nodes: operations or variables
# - Edges: data flow between operations
#
# Example: f(x, y) = (x + y) * (x + y)
#
#     x    y
#      \  /
#       (+) = a
#      /   \
#    (*)    (used twice)
#     |
#   result
# ============================================================================


class Node:
    """
    A node in the computational graph.
    """
    _id_counter = 0
    
    def __init__(self, value=None, op=None, inputs=None, name=None):
        """
        Args:
            value: The computed value at this node
            op: The operation that produced this node
            inputs: List of input nodes
            name: Optional name for debugging
        """
        self.value = value
        self.op = op
        self.inputs = inputs or []
        self.name = name or f"node_{Node._id_counter}"
        Node._id_counter += 1
    
    def __repr__(self):
        return f"Node({self.name}, value={self.value}, op={self.op})"


# ============================================================================
# Exercise 1: Create Input Nodes
# ============================================================================

def create_input_node(value, name=None):
    """
    Create an input node (leaf node with no operation).
    
    Args:
        value: The input value
        name: Optional name
    
    Returns:
        Node with value set and no operation
    """
    # TODO: Create a node with the given value
    # HINT: Node(value=value, op=None, inputs=[], name=name)
    return None  # Replace


# ============================================================================
# Exercise 2: Create Operation Nodes
# ============================================================================

def create_add_node(a, b, name=None):
    """
    Create a node representing a + b.
    
    Args:
        a, b: Input Node objects
        name: Optional name
    
    Returns:
        Node with computed sum
    """
    # TODO: Create node with value = a.value + b.value
    # HINT: Node(value=a.value + b.value, op='add', inputs=[a, b], name=name)
    return None  # Replace


def create_mul_node(a, b, name=None):
    """
    Create a node representing a * b.
    """
    # TODO: Create multiplication node
    return None  # Replace


def create_pow_node(a, n, name=None):
    """
    Create a node representing a ** n (n is a constant).
    """
    # TODO: Create power node
    # Note: n is a scalar, not a Node
    return None  # Replace


# ============================================================================
# Exercise 3: Build a Simple Graph
# ============================================================================

def build_polynomial_graph(x_val):
    """
    Build graph for f(x) = x² + 2x + 1 = (x + 1)²
    
    Returns:
        dict with all nodes: 'x', 'one', 'sum', 'result'
    """
    # TODO: Create input nodes
    x = create_input_node(x_val, 'x')
    one = create_input_node(1.0, 'one')
    
    # TODO: Create computation nodes
    # sum = x + 1
    sum_node = create_add_node(x, one, 'sum')
    
    # result = sum * sum = (x+1)²
    result = create_mul_node(sum_node, sum_node, 'result')
    
    return {
        'x': x,
        'one': one,
        'sum': sum_node,
        'result': result
    }


# ============================================================================
# Exercise 4: Traverse the Graph
# ============================================================================

def get_all_nodes(output_node):
    """
    Get all nodes in the graph by traversing from output to inputs.
    
    Returns:
        List of all nodes (in topological order, leaves first)
    """
    visited = set()
    result = []
    
    def visit(node):
        if node in visited:
            return
        visited.add(node)
        
        # TODO: Visit inputs first (recursive)
        for inp in node.inputs:
            visit(inp)
        
        # TODO: Add current node after inputs
        result.append(node)
    
    visit(output_node)
    return result


# ============================================================================
# Exercise 5: Count Operations
# ============================================================================

def count_operations(output_node):
    """
    Count number of each operation type in the graph.
    
    Returns:
        dict with operation counts
    """
    nodes = get_all_nodes(output_node)
    
    counts = {'add': 0, 'mul': 0, 'pow': 0, 'input': 0}
    
    # TODO: Count operations
    for node in nodes:
        if node.op is None:
            counts['input'] += 1
        elif node.op in counts:
            counts[node.op] += 1
    
    return counts


# ============================================================================
# Exercise 6: Visualize Graph
# ============================================================================

def print_graph(output_node):
    """
    Print a simple text representation of the graph.
    """
    nodes = get_all_nodes(output_node)
    
    print("\nComputational Graph:")
    print("-" * 40)
    
    for node in nodes:
        if node.op is None:
            print(f"  {node.name} = {node.value} (input)")
        else:
            input_names = [n.name for n in node.inputs]
            print(f"  {node.name} = {node.op}({', '.join(input_names)}) = {node.value}")


# ============================================================================
# Exercise 7: Evaluate Graph
# ============================================================================

def evaluate_graph(nodes_dict, new_x):
    """
    Re-evaluate a graph with a new x value.
    
    Args:
        nodes_dict: Dictionary of nodes from build_polynomial_graph
        new_x: New value for x
    
    Returns:
        The new result value
    """
    # TODO: Update values through the graph
    # Update x
    nodes_dict['x'].value = new_x
    
    # Re-compute sum
    nodes_dict['sum'].value = nodes_dict['x'].value + nodes_dict['one'].value
    
    # Re-compute result
    nodes_dict['result'].value = nodes_dict['sum'].value * nodes_dict['sum'].value
    
    return nodes_dict['result'].value


if __name__ == "__main__":
    print("Day 4: Computational Graphs")
    print("=" * 50)
    
    # Build graph for f(x) = (x + 1)²
    graph = build_polynomial_graph(2.0)
    
    if graph['result'] is not None:
        print(f"\nf(2) = (2 + 1)² = {graph['result'].value}")
        print_graph(graph['result'])
    
    print("\nRun test_day04.py to verify your implementations!")
