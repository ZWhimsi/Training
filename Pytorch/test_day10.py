"""Test Suite for Day 10: Model Save/Load"""

import torch
import pytest
import torch.nn as nn
import os
import tempfile
try:
    from day10 import (save_model, load_model, save_checkpoint, load_checkpoint,
                       inspect_state_dict, count_parameters, prepare_for_inference,
                       trace_model, CheckpointManager)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def create_model():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 3)
    )

def test_save_load_model():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    
    try:
        model1 = create_model()
        save_model(model1, path)
        
        assert os.path.exists(path), "File not created"
        
        model2 = create_model()
        load_model(model2, path)
        
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Parameters don't match"
    finally:
        if os.path.exists(path):
            os.unlink(path)

def test_checkpoint():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    
    try:
        torch.manual_seed(42)
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        x = torch.randn(4, 10)
        y = torch.randint(0, 3, (4,))
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        optimizer.step()
        
        save_checkpoint(model, optimizer, epoch=5, loss=0.123, path=path)
        
        assert os.path.exists(path), "File not created"
        
        torch.manual_seed(42)
        new_model = create_model()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        
        info = load_checkpoint(new_model, new_optimizer, path)
        
        assert not (info['epoch'] == 0 and info['loss'] == 0.0), "Not implemented"
        assert info['epoch'] == 5, f"Epoch: got {info['epoch']}, expected 5"
        assert abs(info['loss'] - 0.123) <= 1e-5, f"Loss: got {info['loss']}, expected 0.123"
        
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2), "Model parameters don't match after loading"
    finally:
        if os.path.exists(path):
            os.unlink(path)

def test_inspect_state_dict():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = create_model()
    info = inspect_state_dict(model)
    
    assert info, "Not implemented"
    assert len(info) == 4, f"Expected 4 entries, got {len(info)}"
    
    expected_shapes = {
        '0.weight': (32, 10),
        '0.bias': (32,),
        '2.weight': (3, 32),
        '2.bias': (3,)
    }
    
    for name, (shape, dtype) in info.items():
        if name in expected_shapes:
            assert shape == expected_shapes[name], f"{name} shape: got {shape}, expected {expected_shapes[name]}"
        assert dtype == torch.float32, f"{name} dtype: got {dtype}, expected torch.float32"

def test_count_parameters():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = create_model()
    counts = count_parameters(model)
    
    assert counts['total'] != 0, "Not implemented"
    
    expected = 451
    assert counts['total'] == expected, f"Expected {expected}, got {counts['total']}"

def test_prepare_inference():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = create_model()
    model.train()
    
    model = prepare_for_inference(model)
    
    assert not model.training, "Still in training mode"
    
    for name, param in model.named_parameters():
        assert not param.requires_grad, f"{name} still has requires_grad=True"
    
    x = torch.randn(2, 10)
    output = model(x)
    assert output.shape == (2, 3), f"Output shape wrong: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"

def test_trace_model():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = create_model()
    model.eval()
    example_input = torch.randn(1, 10)
    
    traced = trace_model(model, example_input)
    
    assert traced is not None, "Not implemented"
    
    with torch.no_grad():
        original_output = model(example_input)
        traced_output = traced(example_input)
    
    assert traced_output.shape == torch.Size([1, 3]), f"Output shape: {traced_output.shape}, expected (1, 3)"
    assert torch.allclose(traced_output, original_output, atol=1e-5), f"Traced output differs: {traced_output} vs {original_output}"
    
    test_input = torch.randn(4, 10)
    with torch.no_grad():
        original_test = model(test_input)
        traced_test = traced(test_input)
    
    assert torch.allclose(traced_test, original_test, atol=1e-5), "Traced model gives different output for new input"

def test_checkpoint_manager():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, max_to_keep=3)
        
        model = create_model()
        optimizer = torch.optim.Adam(model.parameters())
        
        for epoch in range(5):
            path = manager.save(model, optimizer, epoch, 0.1 * epoch)
        
        assert path, "Not implemented"
        
        checkpoint_files = [f for f in os.listdir(tmpdir) if f.endswith('.pt')]
        assert len(checkpoint_files) <= 3, f"Should keep only 3, found {len(checkpoint_files)}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
