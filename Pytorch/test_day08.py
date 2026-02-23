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
        model, batch, optimizer, loss_fn = create_simple_setup()
        
        initial_params = [p.clone() for p in model.parameters()]
        loss = train_step(model, batch, optimizer, loss_fn)
        
        if loss == 0.0:
            return False, "Not implemented"
        
        # Check parameters changed
        params_changed = False
        for p_init, p_new in zip(initial_params, model.parameters()):
            if not torch.allclose(p_init, p_new):
                params_changed = True
                break
        
        if not params_changed:
            return False, "Parameters didn't change"
        
        return True, f"OK (loss={loss:.4f})"
    except Exception as e:
        return False, str(e)


def test_val_step() -> Tuple[bool, str]:
    try:
        model, batch, _, loss_fn = create_simple_setup()
        
        metrics = val_step(model, batch, loss_fn)
        
        if metrics['loss'] == 0.0 and metrics['accuracy'] == 0.0:
            return False, "Not implemented"
        
        if metrics['accuracy'] < 0 or metrics['accuracy'] > 1:
            return False, f"Invalid accuracy: {metrics['accuracy']}"
        
        return True, f"OK (acc={metrics['accuracy']:.2f})"
    except Exception as e:
        return False, str(e)


def test_train_epoch() -> Tuple[bool, str]:
    try:
        model = nn.Linear(4, 2)
        data = TensorDataset(torch.randn(100, 4), torch.randint(0, 2, (100,)))
        loader = DataLoader(data, batch_size=16)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        loss = train_epoch(model, loader, optimizer, loss_fn)
        
        if loss == 0.0:
            return False, "Not implemented"
        
        return True, f"OK (avg_loss={loss:.4f})"
    except Exception as e:
        return False, str(e)


def test_validate_epoch() -> Tuple[bool, str]:
    try:
        model = nn.Linear(4, 2)
        data = TensorDataset(torch.randn(100, 4), torch.randint(0, 2, (100,)))
        loader = DataLoader(data, batch_size=16)
        loss_fn = nn.CrossEntropyLoss()
        
        metrics = validate_epoch(model, loader, loss_fn)
        
        if metrics['loss'] == 0.0:
            return False, "Not implemented"
        
        return True, f"OK (acc={metrics['accuracy']:.2f})"
    except Exception as e:
        return False, str(e)


def test_train_model() -> Tuple[bool, str]:
    try:
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
        
        return True, f"OK ({len(history['train_loss'])} epochs)"
    except Exception as e:
        return False, str(e)


def test_early_stopping() -> Tuple[bool, str]:
    try:
        es = EarlyStopping(patience=3, min_delta=0.01)
        
        # Improving
        es(1.0)
        es(0.9)
        es(0.8)
        
        if es.should_stop:
            return False, "Shouldn't stop while improving"
        
        # Not improving
        es(0.85)  # worse
        es(0.82)  # worse
        es(0.83)  # worse
        
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
