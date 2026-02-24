"""Test Suite for Day 7: Parent Tracking"""

import pytest

from day07 import Value, build_expression, verify_parents, count_graph_nodes


def test_get_parents():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    
    parents = c.get_parents()
    assert parents is not None, "Returned None"
    assert a in parents and b in parents, "Parents not tracked"


def test_get_operation():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    d = a * b
    
    assert c.get_operation() == '+', f"Expected '+', got {c.get_operation()}"
    assert d.get_operation() == '*', f"Expected '*', got {d.get_operation()}"


def test_is_leaf():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    
    assert a.is_leaf() is not None, "is_leaf returned None"
    assert a.is_leaf(), "Input should be leaf"
    assert not c.is_leaf(), "Result should not be leaf"


def test_get_all_ancestors():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    d = c * a  # Uses a again
    
    ancestors = d.get_all_ancestors()
    assert ancestors is not None, "Returned None"
    assert a in ancestors and b in ancestors and c in ancestors, "Missing ancestors"


def test_build_expression():
    nodes = build_expression()
    
    assert nodes['result'] is not None, "Result is None"
    # f(2, 3) = 2*3 + 2^2 = 6 + 4 = 10
    assert abs(nodes['result'].data - 10.0) <= 1e-6, f"Expected 10, got {nodes['result'].data}"


def test_verify_parents():
    nodes = build_expression()
    results = verify_parents(nodes)
    
    assert results.get('x_is_leaf') is True, "x should be leaf"
    assert results.get('y_is_leaf') is True, "y should be leaf"
    assert results.get('result_parent_count') == 2, "Result should have 2 parents"


def test_count_nodes():
    nodes = build_expression()
    count = count_graph_nodes(nodes['result'])
    
    assert count is not None, "Returned None"
    # Nodes: x, y, xy, x_squared, result = 5
    assert count == 5, f"Expected 5 nodes, got {count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
