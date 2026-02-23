"""Test Suite for Day 10: Model Save/Load"""

import torch
import torch.nn as nn
import os
import tempfile
from typing import Tuple

try:
    from day10 import (save_model, load_model, save_checkpoint, load_checkpoint,
                       inspect_state_dict, count_parameters, prepare_for_inference,
                       trace_model, CheckpointManager)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def create_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )


def test_save_load_model() -> Tuple[bool, str]:
    try:
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        
        model1 = create_model()
        save_model(model1, path)
        
        if not os.path.exists(path):
            os.unlink(path)
            return False, "File not created"
        
        model2 = create_model()
        load_model(model2, path)
        
        # Check parameters match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            if not torch.allclose(p1, p2):
                os.unlink(path)
                return False, "Parameters don't match"
        
        os.unlink(path)
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_checkpoint() -> Tuple[bool, str]:
    try:
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters())
        
        save_checkpoint(model, optimizer, epoch=5, loss=0.123, path=path)
        
        if not os.path.exists(path):
            os.unlink(path)
            return False, "File not created"
        
        new_model = create_model()
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        info = load_checkpoint(new_model, new_optimizer, path)
        
        if info['epoch'] == 0 and info['loss'] == 0.0:
            os.unlink(path)
            return False, "Not implemented"
        
        if info['epoch'] != 5:
            os.unlink(path)
            return False, f"Epoch: {info['epoch']}"
        
        os.unlink(path)
        return True, f"OK (epoch={info['epoch']}, loss={info['loss']:.3f})"
    except Exception as e:
        return False, str(e)


def test_inspect_state_dict() -> Tuple[bool, str]:
    try:
        model = create_model()
        info = inspect_state_dict(model)
        
        if not info:
            return False, "Not implemented"
        
        # Should have 4 entries (2 weights, 2 biases)
        if len(info) != 4:
            return False, f"Expected 4 entries, got {len(info)}"
        
        return True, f"OK ({len(info)} entries)"
    except Exception as e:
        return False, str(e)


def test_count_parameters() -> Tuple[bool, str]:
    try:
        model = create_model()
        counts = count_parameters(model)
        
        if counts['total'] == 0:
            return False, "Not implemented"
        
        # Linear(10, 32): 10*32 + 32 = 352
        # Linear(32, 3): 32*3 + 3 = 99
        # Total: 451
        expected = 451
        if counts['total'] != expected:
            return False, f"Expected {expected}, got {counts['total']}"
        
        return True, f"OK ({counts['total']} params)"
    except Exception as e:
        return False, str(e)


def test_prepare_inference() -> Tuple[bool, str]:
    try:
        model = create_model()
        
        # Should be in training mode initially
        model.train()
        
        model = prepare_for_inference(model)
        
        if model.training:
            return False, "Still in training mode"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_trace_model() -> Tuple[bool, str]:
    try:
        model = create_model()
        example_input = torch.randn(1, 10)
        
        traced = trace_model(model, example_input)
        
        if traced is None:
            return False, "Not implemented"
        
        # Test traced model
        output = traced(example_input)
        if output.shape != torch.Size([1, 3]):
            return False, f"Output shape: {output.shape}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_checkpoint_manager() -> Tuple[bool, str]:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(tmpdir, max_to_keep=3)
            
            model = create_model()
            optimizer = torch.optim.Adam(model.parameters())
            
            # Save 5 checkpoints
            for epoch in range(5):
                path = manager.save(model, optimizer, epoch, 0.1 * epoch)
            
            if not path:
                return False, "Not implemented"
            
            # Should only keep 3
            checkpoint_files = [f for f in os.listdir(tmpdir) if f.endswith('.pt')]
            if len(checkpoint_files) > 3:
                return False, f"Should keep only 3, found {len(checkpoint_files)}"
            
            return True, f"OK ({len(checkpoint_files)} checkpoints)"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("save_load_model", test_save_load_model),
        ("checkpoint", test_checkpoint),
        ("inspect_state_dict", test_inspect_state_dict),
        ("count_parameters", test_count_parameters),
        ("prepare_inference", test_prepare_inference),
        ("trace_model", test_trace_model),
        ("checkpoint_manager", test_checkpoint_manager),
    ]
    
    print(f"\n{'='*50}\nDay 10: Model Save/Load - Tests\n{'='*50}")
    
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        return
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    run_all_tests()
