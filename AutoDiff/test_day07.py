"""Test Suite for Day 7: Parent Tracking"""

import sys
from typing import Tuple

try:
    from day07 import Value, build_expression, verify_parents, count_graph_nodes
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_get_parents() -> Tuple[bool, str]:
    try:
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        
        parents = c.get_parents()
        if parents is None:
            return False, "Returned None"
        if a not in parents or b not in parents:
            return False, "Parents not tracked"
        return True, "Parents tracked correctly"
    except Exception as e:
        return False, str(e)


def test_get_operation() -> Tuple[bool, str]:
    try:
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        d = a * b
        
        if c.get_operation() != '+':
            return False, f"Expected '+', got {c.get_operation()}"
        if d.get_operation() != '*':
            return False, f"Expected '*', got {d.get_operation()}"
        return True, "Operations tracked"
    except Exception as e:
        return False, str(e)


def test_is_leaf() -> Tuple[bool, str]:
    try:
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        
        if a.is_leaf() is None:
            return False, "is_leaf returned None"
        if not a.is_leaf():
            return False, "Input should be leaf"
        if c.is_leaf():
            return False, "Result should not be leaf"
        return True, "Leaf detection correct"
    except Exception as e:
        return False, str(e)


def test_get_all_ancestors() -> Tuple[bool, str]:
    try:
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        d = c * a  # Uses a again
        
        ancestors = d.get_all_ancestors()
        if ancestors is None:
            return False, "Returned None"
        if a not in ancestors or b not in ancestors or c not in ancestors:
            return False, "Missing ancestors"
        return True, f"Found {len(ancestors)} ancestors"
    except Exception as e:
        return False, str(e)


def test_build_expression() -> Tuple[bool, str]:
    try:
        nodes = build_expression()
        
        if nodes['result'] is None:
            return False, "Result is None"
        
        # f(2, 3) = 2*3 + 2^2 = 6 + 4 = 10
        if abs(nodes['result'].data - 10.0) > 1e-6:
            return False, f"Expected 10, got {nodes['result'].data}"
        
        return True, "f(2,3) = 2*3 + 2² = 10"
    except Exception as e:
        return False, str(e)


def test_verify_parents() -> Tuple[bool, str]:
    try:
        nodes = build_expression()
        results = verify_parents(nodes)
        
        if results.get('x_is_leaf') is not True:
            return False, "x should be leaf"
        if results.get('y_is_leaf') is not True:
            return False, "y should be leaf"
        if results.get('result_parent_count') != 2:
            return False, "Result should have 2 parents"
        
        return True, "Parent verification passed"
    except Exception as e:
        return False, str(e)


def test_count_nodes() -> Tuple[bool, str]:
    try:
        nodes = build_expression()
        count = count_graph_nodes(nodes['result'])
        
        if count is None:
            return False, "Returned None"
        
        # Nodes: x, y, xy, x_squared, result = 5
        if count != 5:
            return False, f"Expected 5 nodes, got {count}"
        
        return True, "5 nodes in graph"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("get_parents", test_get_parents),
        ("get_operation", test_get_operation),
        ("is_leaf", test_is_leaf),
        ("get_all_ancestors", test_get_all_ancestors),
        ("build_expression", test_build_expression),
        ("verify_parents", test_verify_parents),
        ("count_nodes", test_count_nodes),
    ]
    
    print(f"\n{'='*50}\nDay 7: Parent Tracking - Tests\n{'='*50}")
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
