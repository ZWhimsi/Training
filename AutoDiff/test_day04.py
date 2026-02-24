"""Test Suite for Day 4: Computational Graphs"""

import pytest

from day04 import (create_input_node, create_add_node, create_mul_node,
                   create_pow_node, build_polynomial_graph,
                   get_all_nodes, count_operations, evaluate_graph)


def test_input_node():
    node = create_input_node(5.0, 'x')
    
    assert node is not None, "Returned None"
    assert node.value == 5.0, f"Value should be 5.0, got {node.value}"
    assert node.op is None, "Input node should have no op"


def test_add_node():
    a = create_input_node(3.0, 'a')
    b = create_input_node(4.0, 'b')
    result = create_add_node(a, b, 'sum')
    
    assert result is not None, "Returned None"
    assert result.value == 7.0, f"3+4 should be 7, got {result.value}"
    assert result.op == 'add', "Op should be 'add'"


def test_mul_node():
    a = create_input_node(3.0, 'a')
    b = create_input_node(4.0, 'b')
    result = create_mul_node(a, b, 'prod')
    
    assert result is not None, "Returned None"
    assert result.value == 12.0, f"3*4 should be 12, got {result.value}"


def test_polynomial_graph():
    graph = build_polynomial_graph(2.0)
    
    assert graph['result'] is not None, "Result is None"
    # f(2) = (2+1)² = 9
    assert graph['result'].value == 9.0, f"f(2)=(2+1)² should be 9, got {graph['result'].value}"


def test_get_all_nodes():
    graph = build_polynomial_graph(2.0)
    nodes = get_all_nodes(graph['result'])
    
    assert nodes is not None and len(nodes) > 0, "No nodes returned"
    # Should have 4 nodes: x, one, sum, result
    assert len(nodes) == 4, f"Expected 4 nodes, got {len(nodes)}"


def test_count_operations():
    graph = build_polynomial_graph(2.0)
    counts = count_operations(graph['result'])
    
    assert counts is not None, "Returned None"
    # 2 inputs (x, one), 1 add, 1 mul
    assert counts.get('input') == 2, f"Expected 2 inputs, got {counts.get('input')}"
    assert counts.get('add') == 1, f"Expected 1 add, got {counts.get('add')}"
    assert counts.get('mul') == 1, f"Expected 1 mul, got {counts.get('mul')}"


def test_evaluate_graph():
    graph = build_polynomial_graph(2.0)
    # Re-evaluate with x=3
    result = evaluate_graph(graph, 3.0)
    # f(3) = (3+1)² = 16
    assert result == 16.0, f"f(3) should be 16, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
