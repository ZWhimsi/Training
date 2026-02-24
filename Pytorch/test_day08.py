"""Test Suite for Day 8: Training Loop"""

import torch
import pytest
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
try:
    from day08 import (train_step, val_step, train_epoch, validate_epoch,
                       train_model, EarlyStopping, create_toy_dataset)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def create_simple_setup():
    model = nn.Linear(4, 2)
    X = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    return model, (X, y), optimizer, loss_fn

def test_train_step():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model, batch, optimizer, loss_fn = create_simple_setup()
    
    initial_params = [p.clone() for p in model.parameters()]
    
    model.eval()
    with torch.no_grad():
        expected_loss = loss_fn(model(batch[0]), batch[1]).item()
    model.train()
    
    loss = train_step(model, batch, optimizer, loss_fn)
    
    assert loss != 0.0, "Not implemented (returned 0)"
    
    assert abs(loss - expected_loss) <= 1e-5, f"Loss {loss:.4f} doesn't match expected {expected_loss:.4f}"
    
    params_changed = False
    for p_init, p_new in zip(initial_params, model.parameters()):
        if not torch.allclose(p_init, p_new):
            params_changed = True
            break
    
    assert params_changed, "Parameters didn't change after step"
    assert model.training, "Model should be in training mode after train_step"

def test_val_step():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model, batch, _, loss_fn = create_simple_setup()
    
    metrics = val_step(model, batch, loss_fn)
    
    assert not (metrics['loss'] == 0.0 and metrics['accuracy'] == 0.0), "Not implemented (both zero)"
    
    model.eval()
    with torch.no_grad():
        outputs = model(batch[0])
        expected_loss = loss_fn(outputs, batch[1]).item()
        preds = outputs.argmax(dim=-1)
        expected_acc = (preds == batch[1]).float().mean().item()
    
    assert abs(metrics['loss'] - expected_loss) <= 1e-5, f"Loss: got {metrics['loss']:.4f}, expected {expected_loss:.4f}"
    assert abs(metrics['accuracy'] - expected_acc) <= 1e-5, f"Accuracy: got {metrics['accuracy']:.4f}, expected {expected_acc:.4f}"
    assert 0 <= metrics['accuracy'] <= 1, f"Accuracy out of range: {metrics['accuracy']}"

def test_train_epoch():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    X = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    data = TensorDataset(X, y)
    loader = DataLoader(data, batch_size=16, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    model.eval()
    batch_losses = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_loss = loss_fn(model(batch_x), batch_y).item()
            batch_losses.append(batch_loss)
    expected_avg_loss = sum(batch_losses) / len(batch_losses)
    
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    loss = train_epoch(model, loader, optimizer, loss_fn)
    
    assert loss != 0.0, "Not implemented"
    assert abs(loss - expected_avg_loss) <= 0.5, f"Avg loss {loss:.4f} far from expected {expected_avg_loss:.4f}"
    assert 0 <= loss <= 10, f"Loss out of reasonable range: {loss:.4f}"

def test_validate_epoch():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = nn.Linear(4, 2)
    X = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    data = TensorDataset(X, y)
    loader = DataLoader(data, batch_size=16, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            total_loss += loss_fn(outputs, batch_y).item()
            preds = outputs.argmax(dim=-1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_y.size(0)
    expected_avg_loss = total_loss / len(loader)
    expected_accuracy = total_correct / total_samples
    
    metrics = validate_epoch(model, loader, loss_fn)
    
    assert metrics['loss'] != 0.0, "Not implemented"
    assert abs(metrics['loss'] - expected_avg_loss) <= 1e-5, f"Loss: got {metrics['loss']:.4f}, expected {expected_avg_loss:.4f}"
    assert abs(metrics['accuracy'] - expected_accuracy) <= 1e-5, f"Accuracy: got {metrics['accuracy']:.4f}, expected {expected_accuracy:.4f}"

def test_train_model():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
    train_data = create_toy_dataset(200)
    val_data = create_toy_dataset(50)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    history = train_model(model, train_loader, val_loader, optimizer, loss_fn, n_epochs=3)
    
    assert history['train_loss'], "Not implemented"
    assert len(history['train_loss']) == 3, f"Expected 3 epochs, got {len(history['train_loss'])}"
    assert len(history['val_loss']) == 3, f"val_loss should have 3 entries, got {len(history['val_loss'])}"
    assert len(history['val_accuracy']) == 3, f"val_accuracy should have 3 entries, got {len(history['val_accuracy'])}"
    
    for i, loss in enumerate(history['train_loss']):
        assert 0 <= loss <= 10, f"train_loss[{i}]={loss:.4f} out of range"
    
    for i, acc in enumerate(history['val_accuracy']):
        assert 0 <= acc <= 1, f"val_accuracy[{i}]={acc:.4f} out of range [0,1]"
    
    assert history['train_loss'][-1] <= history['train_loss'][0] * 2, f"Training loss increased significantly: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}"

def test_early_stopping():
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    es = EarlyStopping(patience=3, min_delta=0.01)
    
    result1 = es(1.0)
    assert result1 == False, f"Should return False when improving, got {result1}"
    result2 = es(0.9)
    assert result2 == False, f"Should return False when improving, got {result2}"
    result3 = es(0.8)
    assert result3 == False, f"Should return False when improving, got {result3}"
    
    assert not es.should_stop, "Shouldn't stop while improving"
    assert abs(es.best_loss - 0.8) <= 1e-5, f"best_loss should be 0.8, got {es.best_loss}"
    
    es(0.85)
    assert es.counter == 1, f"Counter should be 1 after first non-improvement, got {es.counter}"
    
    es(0.82)
    assert es.counter == 2, f"Counter should be 2 after second non-improvement, got {es.counter}"
    
    es(0.83)
    assert es.counter == 3, f"Counter should be 3 after third non-improvement, got {es.counter}"
    
    assert es.should_stop, "Should stop after patience exceeded"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
