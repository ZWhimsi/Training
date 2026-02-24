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
        
        torch.manual_seed(42)
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Do a training step to update optimizer state
        x = torch.randn(4, 10)
        y = torch.randint(0, 3, (4,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch=5, loss=0.123, path=path)
        
        if not os.path.exists(path):
            os.unlink(path)
            return False, "File not created"
        
        # Create new model and optimizer
        torch.manual_seed(42)
        new_model = create_model()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        info = load_checkpoint(new_model, new_optimizer, path)
        
        if info['epoch'] == 0 and info['loss'] == 0.0:
            os.unlink(path)
            return False, "Not implemented"
        
        # Verify epoch matches
        if info['epoch'] != 5:
            os.unlink(path)
            return False, f"Epoch: got {info['epoch']}, expected 5"
        
        # Verify loss matches
        if abs(info['loss'] - 0.123) > 1e-5:
            os.unlink(path)
            return False, f"Loss: got {info['loss']}, expected 0.123"
        
        # Verify model parameters match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            if not torch.allclose(p1, p2):
                os.unlink(path)
                return False, "Model parameters don't match after loading"
        
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
        
        # Verify shapes are correct
        expected_shapes = {
            '0.weight': (32, 10),
            '0.bias': (32,),
            '2.weight': (3, 32),
            '2.bias': (3,)
        }
        
        for name, (shape, dtype) in info.items():
            if name in expected_shapes:
                if shape != expected_shapes[name]:
                    return False, f"{name} shape: got {shape}, expected {expected_shapes[name]}"
            if dtype != torch.float32:
                return False, f"{name} dtype: got {dtype}, expected torch.float32"
        
        return True, f"OK ({len(info)} entries with correct shapes)"
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
        
        # Check requires_grad is disabled for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                return False, f"{name} still has requires_grad=True"
        
        # Verify model can still produce output
        x = torch.randn(2, 10)
        output = model(x)
        if output.shape != (2, 3):
            return False, f"Output shape wrong: {output.shape}"
        if torch.isnan(output).any():
            return False, "Output contains NaN"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_trace_model() -> Tuple[bool, str]:
    try:
        torch.manual_seed(42)
        model = create_model()
        model.eval()
        example_input = torch.randn(1, 10)
        
        traced = trace_model(model, example_input)
        
        if traced is None:
            return False, "Not implemented"
        
        # Test traced model produces same output as original
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = traced(example_input)
        
        if traced_output.shape != torch.Size([1, 3]):
            return False, f"Output shape: {traced_output.shape}, expected (1, 3)"
        
        if not torch.allclose(traced_output, original_output, atol=1e-5):
            return False, f"Traced output differs: {traced_output} vs {original_output}"
        
        # Test with different input
        test_input = torch.randn(4, 10)
        with torch.no_grad():
            original_test = model(test_input)
            traced_test = traced(test_input)
        
        if not torch.allclose(traced_test, original_test, atol=1e-5):
            return False, "Traced model gives different output for new input"
        
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
