"""Test Suite for Day 8: Training Loop"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

try:
    from day08 import (train_step, val_step, train_epoch, validate_epoch,
                       train_model, EarlyStopping, create_toy_dataset)
    IMPORT_SUCCESS = True
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


def test_train_step() -> Tuple[bool, str]:
    try:
        torch.manual_seed(42)
        model, batch, optimizer, loss_fn = create_simple_setup()
        
        initial_params = [p.clone() for p in model.parameters()]
        
        # Compute expected loss BEFORE the step (this is what train_step should return)
        model.eval()
        with torch.no_grad():
            expected_loss = loss_fn(model(batch[0]), batch[1]).item()
        model.train()
        
        loss = train_step(model, batch, optimizer, loss_fn)
        
        if loss == 0.0:
            return False, "Not implemented (returned 0)"
        
        # Loss should match expected (before gradient update)
        if abs(loss - expected_loss) > 1e-5:
            return False, f"Loss {loss:.4f} doesn't match expected {expected_loss:.4f}"
        
        # Check parameters changed after gradient step
        params_changed = False
        for p_init, p_new in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_new):
                params_changed = True
                break
        
        if not params_changed:
            return False, "Parameters didn't change after step"
        
        # Verify model is in training mode
        if not model.training:
            return False, "Model should be in training mode after train_step"
        
        return True, f"OK (loss={loss:.4f})"
    except Exception as e:
        return False, str(e)


def test_val_step() -> Tuple[bool, str]:
    try:
        torch.manual_seed(42)
        model, batch, _, loss_fn = create_simple_setup()
        
        metrics = val_step(model, batch, loss_fn)
        
        if metrics['loss'] == 0.0 and metrics['accuracy'] == 0.0:
            return False, "Not implemented (both zero)"
        
        # Compute expected loss and accuracy manually
        model.eval()
        with torch.no_grad():
            outputs = model(batch[0])
            expected_loss = loss_fn(outputs, batch[1]).item()
            preds = outputs.argmax(dim=-1)
            expected_acc = (preds == batch[1]).float().mean().item()
        
        # Validate loss matches exactly
        if abs(metrics['loss'] - expected_loss) > 1e-5:
            return False, f"Loss: got {metrics['loss']:.4f}, expected {expected_loss:.4f}"
        
        # Validate accuracy matches exactly
        if abs(metrics['accuracy'] - expected_acc) > 1e-5:
            return False, f"Accuracy: got {metrics['accuracy']:.4f}, expected {expected_acc:.4f}"
        
        if metrics['accuracy'] < 0 or metrics['accuracy'] > 1:
            return False, f"Accuracy out of range: {metrics['accuracy']}"
        
        return True, f"OK (loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.2f})"
    except Exception as e:
        return False, str(e)


def test_train_epoch() -> Tuple[bool, str]:
    try:
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        X = torch.randn(100, 4)
        y = torch.randint(0, 2, (100,))
        data = TensorDataset(X, y)
        loader = DataLoader(data, batch_size=16, shuffle=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        # Compute expected average loss manually (first pass through data)
        model.eval()
        batch_losses = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_loss = loss_fn(model(batch_x), batch_y).item()
                batch_losses.append(batch_loss)
        expected_avg_loss = sum(batch_losses) / len(batch_losses)
        
        # Reset model and optimizer for actual test
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        loss = train_epoch(model, loader, optimizer, loss_fn)
        
        if loss == 0.0:
            return False, "Not implemented"
        
        # Loss should be close to expected (accounting for parameter updates during epoch)
        if abs(loss - expected_avg_loss) > 0.5:
            return False, f"Avg loss {loss:.4f} far from expected {expected_avg_loss:.4f}"
        
        # Verify loss is positive and reasonable
        if loss < 0 or loss > 10:
            return False, f"Loss out of reasonable range: {loss:.4f}"
        
        return True, f"OK (avg_loss={loss:.4f})"
    except Exception as e:
        return False, str(e)


def test_validate_epoch() -> Tuple[bool, str]:
    try:
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        X = torch.randn(100, 4)
        y = torch.randint(0, 2, (100,))
        data = TensorDataset(X, y)
        loader = DataLoader(data, batch_size=16, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()
        
        # Compute expected metrics manually
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
        
        if metrics['loss'] == 0.0:
            return False, "Not implemented"
        
        # Validate loss matches
        if abs(metrics['loss'] - expected_avg_loss) > 1e-5:
            return False, f"Loss: got {metrics['loss']:.4f}, expected {expected_avg_loss:.4f}"
        
        # Validate accuracy matches
        if abs(metrics['accuracy'] - expected_accuracy) > 1e-5:
            return False, f"Accuracy: got {metrics['accuracy']:.4f}, expected {expected_accuracy:.4f}"
        
        return True, f"OK (loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.2f})"
    except Exception as e:
        return False, str(e)


def test_train_model() -> Tuple[bool, str]:
    try:
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
        train_data = create_toy_dataset(200)
        val_data = create_toy_dataset(50)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        history = train_model(model, train_loader, val_loader, optimizer, loss_fn, n_epochs=3)
        
        if not history['train_loss']:
            return False, "Not implemented"
        
        if len(history['train_loss']) != 3:
            return False, f"Expected 3 epochs, got {len(history['train_loss'])}"
        
        # Verify all metrics are recorded
        if len(history['val_loss']) != 3:
            return False, f"val_loss should have 3 entries, got {len(history['val_loss'])}"
        if len(history['val_accuracy']) != 3:
            return False, f"val_accuracy should have 3 entries, got {len(history['val_accuracy'])}"
        
        # Verify losses are positive and reasonable
        for i, loss in enumerate(history['train_loss']):
            if loss < 0 or loss > 10:
                return False, f"train_loss[{i}]={loss:.4f} out of range"
        
        # Verify accuracies are in valid range [0, 1]
        for i, acc in enumerate(history['val_accuracy']):
            if acc < 0 or acc > 1:
                return False, f"val_accuracy[{i}]={acc:.4f} out of range [0,1]"
        
        # Training loss should generally decrease (or at least not explode)
        if history['train_loss'][-1] > history['train_loss'][0] * 2:
            return False, f"Training loss increased significantly: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}"
        
        return True, f"OK ({len(history['train_loss'])} epochs, final_loss={history['train_loss'][-1]:.4f})"
    except Exception as e:
        return False, str(e)


def test_early_stopping() -> Tuple[bool, str]:
    try:
        es = EarlyStopping(patience=3, min_delta=0.01)
        
        # Improving - should not stop
        result1 = es(1.0)
        if result1 != False:
            return False, f"Should return False when improving, got {result1}"
        result2 = es(0.9)
        if result2 != False:
            return False, f"Should return False when improving, got {result2}"
        result3 = es(0.8)
        if result3 != False:
            return False, f"Should return False when improving, got {result3}"
        
        if es.should_stop:
            return False, "Shouldn't stop while improving"
        
        # Verify best_loss is tracked
        if abs(es.best_loss - 0.8) > 1e-5:
            return False, f"best_loss should be 0.8, got {es.best_loss}"
        
        # Not improving - counter should increment
        es(0.85)  # worse by 0.05 > min_delta
        if es.counter != 1:
            return False, f"Counter should be 1 after first non-improvement, got {es.counter}"
        
        es(0.82)  # worse by 0.02 > min_delta
        if es.counter != 2:
            return False, f"Counter should be 2 after second non-improvement, got {es.counter}"
        
        es(0.83)  # worse by 0.03 > min_delta
        if es.counter != 3:
            return False, f"Counter should be 3 after third non-improvement, got {es.counter}"
        
        if not es.should_stop:
            return False, "Should stop after patience exceeded"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("train_step", test_train_step),
        ("val_step", test_val_step),
        ("train_epoch", test_train_epoch),
        ("validate_epoch", test_validate_epoch),
        ("train_model", test_train_model),
        ("early_stopping", test_early_stopping),
    ]
    
    print(f"\n{'='*50}\nDay 8: Training Loop - Tests\n{'='*50}")
    
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
