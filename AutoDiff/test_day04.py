"""Test Suite for Day 4: Computational Graphs"""

import sys
from typing import Tuple

try:
    from day04 import (create_input_node, create_add_node, create_mul_node,
                       create_pow_node, build_polynomial_graph,
                       get_all_nodes, count_operations, evaluate_graph)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_input_node() -> Tuple[bool, str]:
    try:
        node = create_input_node(5.0, 'x')
        
        if node is None:
            return False, "Returned None"
        if node.value != 5.0:
            return False, f"Value should be 5.0, got {node.value}"
        if node.op is not None:
            return False, "Input node should have no op"
        return True, "Input node created"
    except Exception as e:
        return False, str(e)


def test_add_node() -> Tuple[bool, str]:
    try:
        a = create_input_node(3.0, 'a')
        b = create_input_node(4.0, 'b')
        result = create_add_node(a, b, 'sum')
        
        if result is None:
            return False, "Returned None"
        if result.value != 7.0:
            return False, f"3+4 should be 7, got {result.value}"
        if result.op != 'add':
            return False, "Op should be 'add'"
        return True, "3 + 4 = 7"
    except Exception as e:
        return False, str(e)


def test_mul_node() -> Tuple[bool, str]:
    try:
        a = create_input_node(3.0, 'a')
        b = create_input_node(4.0, 'b')
        result = create_mul_node(a, b, 'prod')
        
        if result is None:
            return False, "Returned None"
        if result.value != 12.0:
            return False, f"3*4 should be 12, got {result.value}"
        return True, "3 * 4 = 12"
    except Exception as e:
        return False, str(e)


def test_polynomial_graph() -> Tuple[bool, str]:
    try:
        graph = build_polynomial_graph(2.0)
        
        if graph['result'] is None:
            return False, "Result is None"
        
        # f(2) = (2+1)² = 9
        if graph['result'].value != 9.0:
            return False, f"f(2)=(2+1)² should be 9, got {graph['result'].value}"
        
        return True, "f(2) = (2+1)² = 9"
    except Exception as e:
        return False, str(e)


def test_get_all_nodes() -> Tuple[bool, str]:
    try:
        graph = build_polynomial_graph(2.0)
        nodes = get_all_nodes(graph['result'])
        
        if nodes is None or len(nodes) == 0:
            return False, "No nodes returned"
        
        # Should have 4 nodes: x, one, sum, result
        if len(nodes) != 4:
            return False, f"Expected 4 nodes, got {len(nodes)}"
        
        return True, f"Found {len(nodes)} nodes"
    except Exception as e:
        return False, str(e)


def test_count_operations() -> Tuple[bool, str]:
    try:
        graph = build_polynomial_graph(2.0)
        counts = count_operations(graph['result'])
        
        if counts is None:
            return False, "Returned None"
        
        # 2 inputs (x, one), 1 add, 1 mul
        if counts.get('input') != 2:
            return False, f"Expected 2 inputs, got {counts.get('input')}"
        if counts.get('add') != 1:
            return False, f"Expected 1 add, got {counts.get('add')}"
        if counts.get('mul') != 1:
            return False, f"Expected 1 mul, got {counts.get('mul')}"
        
        return True, "2 inputs, 1 add, 1 mul"
    except Exception as e:
        return False, str(e)


def test_evaluate_graph() -> Tuple[bool, str]:
    try:
        graph = build_polynomial_graph(2.0)
        
        # Re-evaluate with x=3
        result = evaluate_graph(graph, 3.0)
        
        # f(3) = (3+1)² = 16
        if result != 16.0:
            return False, f"f(3) should be 16, got {result}"
        
        return True, "f(3) = (3+1)² = 16"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("input_node", test_input_node),
        ("add_node", test_add_node),
        ("mul_node", test_mul_node),
        ("polynomial_graph", test_polynomial_graph),
        ("get_all_nodes", test_get_all_nodes),
        ("count_operations", test_count_operations),
        ("evaluate_graph", test_evaluate_graph),
    ]
    
    print(f"\n{'='*50}\nDay 4: Computational Graphs - Tests\n{'='*50}")
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    run_all_tests()
