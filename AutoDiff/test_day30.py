"""Test Suite for Day 30: Training Loop Infrastructure"""

import numpy as np
import sys
import os
import tempfile
from typing import Tuple

try:
    from day30 import (
        Tensor,
        Parameter,
        Module,
        Linear,
        ReLU,
        Sequential,
        Optimizer,
        Adam,
        Dataset,
        DataLoader,
        MSELoss,
        CrossEntropyLoss,
        train_epoch,
        validate_epoch,
        save_checkpoint,
        load_checkpoint,
        EarlyStopping,
        Trainer,
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_dataset_creation() -> Tuple[bool, str]:
    """Test Dataset creation."""
    try:
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        
        if dataset is None:
            return False, "Dataset is None"
        
        return True, "Dataset created"
    except Exception as e:
        return False, str(e)


def test_dataset_length() -> Tuple[bool, str]:
    """Test Dataset length."""
    try:
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        
        if len(dataset) != 100:
            return False, f"Expected 100, got {len(dataset)}"
        
        return True, "Dataset length correct"
    except Exception as e:
        return False, str(e)


def test_dataset_getitem() -> Tuple[bool, str]:
    """Test Dataset indexing."""
    try:
        X = np.arange(20).reshape(5, 4).astype(float)
        Y = np.arange(10).reshape(5, 2).astype(float)
        dataset = Dataset(X, Y)
        
        x, y = dataset[2]
        
        if not np.allclose(x, X[2]):
            return False, "X mismatch"
        if not np.allclose(y, Y[2]):
            return False, "Y mismatch"
        
        return True, "Dataset indexing works"
    except Exception as e:
        return False, str(e)


def test_dataloader_creation() -> Tuple[bool, str]:
    """Test DataLoader creation."""
    try:
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        loader = DataLoader(dataset, batch_size=32)
        
        if loader is None:
            return False, "DataLoader is None"
        
        return True, "DataLoader created"
    except Exception as e:
        return False, str(e)


def test_dataloader_length() -> Tuple[bool, str]:
    """Test DataLoader batch count."""
    try:
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        
        loader = DataLoader(dataset, batch_size=32, drop_last=False)
        if len(loader) != 4:
            return False, f"Expected 4 batches, got {len(loader)}"
        
        loader_drop = DataLoader(dataset, batch_size=32, drop_last=True)
        if len(loader_drop) != 3:
            return False, f"Expected 3 batches with drop_last, got {len(loader_drop)}"
        
        return True, "DataLoader length correct"
    except Exception as e:
        return False, str(e)


def test_dataloader_iteration() -> Tuple[bool, str]:
    """Test DataLoader iteration."""
    try:
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        loader = DataLoader(dataset, batch_size=32)
        
        batches = list(loader)
        
        if len(batches) != 4:
            return False, f"Expected 4 batches, got {len(batches)}"
        
        if batches[0][0].shape != (32, 4):
            return False, f"Wrong batch shape: {batches[0][0].shape}"
        
        if batches[-1][0].shape[0] != 4:
            return False, f"Last batch should have 4 samples"
        
        return True, "DataLoader iteration works"
    except Exception as e:
        return False, str(e)


def test_dataloader_shuffle() -> Tuple[bool, str]:
    """Test DataLoader shuffling."""
    try:
        np.random.seed(42)
        X = np.arange(100).reshape(-1, 1).astype(float)
        Y = np.arange(100).reshape(-1, 1).astype(float)
        dataset = Dataset(X, Y)
        
        loader = DataLoader(dataset, batch_size=100, shuffle=True)
        batch_x, _ = next(iter(loader))
        
        if np.allclose(batch_x.flatten(), np.arange(100)):
            return False, "Data not shuffled"
        
        return True, "DataLoader shuffles correctly"
    except Exception as e:
        return False, str(e)


def test_mse_loss() -> Tuple[bool, str]:
    """Test MSE loss function."""
    try:
        criterion = MSELoss()
        
        pred = Tensor(np.array([1.0, 2.0, 3.0]))
        target = Tensor(np.array([1.0, 2.0, 3.0]))
        
        loss = criterion(pred, target)
        if not np.isclose(loss.data, 0.0):
            return False, f"Expected 0, got {loss.data}"
        
        pred2 = Tensor(np.array([2.0, 3.0, 4.0]))
        loss2 = criterion(pred2, target)
        if not np.isclose(loss2.data, 1.0):
            return False, f"Expected 1.0, got {loss2.data}"
        
        return True, "MSE loss correct"
    except Exception as e:
        return False, str(e)


def test_mse_loss_backward() -> Tuple[bool, str]:
    """Test MSE loss backward."""
    try:
        criterion = MSELoss()
        
        pred = Tensor(np.array([1.0, 2.0, 3.0]))
        target = Tensor(np.array([0.0, 0.0, 0.0]))
        
        loss = criterion(pred, target)
        loss.backward()
        
        if np.allclose(pred.grad, 0):
            return False, "Gradient is zero"
        
        return True, "MSE backward works"
    except Exception as e:
        return False, str(e)


def test_train_epoch() -> Tuple[bool, str]:
    """Test train_epoch function."""
    try:
        np.random.seed(42)
        
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        loader = DataLoader(dataset, batch_size=32)
        
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        optimizer = Adam(model.parameters())
        criterion = MSELoss()
        
        metrics = train_epoch(model, loader, optimizer, criterion)
        
        if 'loss' not in metrics:
            return False, "No loss in metrics"
        if metrics['loss'] <= 0:
            return False, f"Invalid loss: {metrics['loss']}"
        
        return True, f"train_epoch loss: {metrics['loss']:.4f}"
    except Exception as e:
        return False, str(e)


def test_validate_epoch() -> Tuple[bool, str]:
    """Test validate_epoch function."""
    try:
        np.random.seed(42)
        
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        loader = DataLoader(dataset, batch_size=32)
        
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        criterion = MSELoss()
        
        metrics = validate_epoch(model, loader, criterion)
        
        if 'loss' not in metrics:
            return False, "No loss in metrics"
        
        return True, f"validate_epoch loss: {metrics['loss']:.4f}"
    except Exception as e:
        return False, str(e)


def test_checkpoint_save_load() -> Tuple[bool, str]:
    """Test checkpoint saving and loading."""
    try:
        np.random.seed(42)
        
        model1 = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        optimizer1 = Adam(model1.parameters())
        
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            filepath = f.name
        
        try:
            save_checkpoint(filepath, model1, optimizer1, epoch=5, loss=0.123)
            
            if not os.path.exists(filepath):
                return False, "Checkpoint file not created"
            
            model2 = Sequential(
                Linear(4, 8),
                ReLU(),
                Linear(8, 2)
            )
            optimizer2 = Adam(model2.parameters())
            
            metadata = load_checkpoint(filepath, model2, optimizer2)
            
            if metadata.get('epoch') != 5:
                return False, f"Epoch mismatch: {metadata.get('epoch')}"
            
            state1 = model1.state_dict()
            state2 = model2.state_dict()
            
            for key in state1:
                if key in state2:
                    if not np.allclose(state1[key], state2[key]):
                        return False, f"State mismatch in {key}"
            
            return True, "Checkpoint save/load works"
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    except Exception as e:
        return False, str(e)


def test_early_stopping_creation() -> Tuple[bool, str]:
    """Test EarlyStopping creation."""
    try:
        es = EarlyStopping(patience=5, min_delta=0.001)
        
        if es.patience != 5:
            return False, f"Wrong patience: {es.patience}"
        
        return True, "EarlyStopping created"
    except Exception as e:
        return False, str(e)


def test_early_stopping_behavior() -> Tuple[bool, str]:
    """Test EarlyStopping behavior."""
    try:
        es = EarlyStopping(patience=3, min_delta=0)
        
        stop = es(1.0)
        if stop:
            return False, "Should not stop on first call"
        
        stop = es(0.8)
        if stop:
            return False, "Should not stop when improving"
        
        for i in range(3):
            stop = es(0.85)
        
        if not stop:
            return False, "Should stop after patience exhausted"
        
        return True, "EarlyStopping behavior correct"
    except Exception as e:
        return False, str(e)


def test_early_stopping_best() -> Tuple[bool, str]:
    """Test EarlyStopping tracks best."""
    try:
        es = EarlyStopping(patience=5)
        
        es(1.0)
        if not es.is_best:
            return False, "First value should be best"
        
        es(0.5)
        if not es.is_best:
            return False, "Improvement should be best"
        
        es(0.6)
        if es.is_best:
            return False, "Worse value should not be best"
        
        return True, "EarlyStopping best tracking works"
    except Exception as e:
        return False, str(e)


def test_trainer_creation() -> Tuple[bool, str]:
    """Test Trainer creation."""
    try:
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        optimizer = Adam(model.parameters())
        criterion = MSELoss()
        
        trainer = Trainer(model, optimizer, criterion)
        
        if trainer is None:
            return False, "Trainer is None"
        
        return True, "Trainer created"
    except Exception as e:
        return False, str(e)


def test_trainer_fit() -> Tuple[bool, str]:
    """Test Trainer fit method."""
    try:
        np.random.seed(42)
        
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        train_loader = DataLoader(dataset, batch_size=32)
        
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        optimizer = Adam(model.parameters())
        criterion = MSELoss()
        
        trainer = Trainer(model, optimizer, criterion)
        history = trainer.fit(train_loader, epochs=20, verbose=False)
        
        if 'train_loss' not in history:
            return False, "No train_loss in history"
        
        if len(history['train_loss']) != 20:
            return False, f"Wrong history length: {len(history['train_loss'])}"
        
        if history['train_loss'][-1] >= history['train_loss'][0]:
            return False, "Loss should decrease"
        
        return True, f"Loss: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}"
    except Exception as e:
        return False, str(e)


def test_trainer_with_validation() -> Tuple[bool, str]:
    """Test Trainer with validation."""
    try:
        np.random.seed(42)
        
        X_train = np.random.randn(80, 4)
        Y_train = np.random.randn(80, 2)
        X_val = np.random.randn(20, 4)
        Y_val = np.random.randn(20, 2)
        
        train_dataset = Dataset(X_train, Y_train)
        val_dataset = Dataset(X_val, Y_val)
        train_loader = DataLoader(train_dataset, batch_size=16)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        optimizer = Adam(model.parameters())
        criterion = MSELoss()
        
        trainer = Trainer(model, optimizer, criterion)
        history = trainer.fit(train_loader, val_loader, epochs=20, verbose=False)
        
        if 'val_loss' not in history:
            return False, "No val_loss in history"
        
        return True, "Training with validation works"
    except Exception as e:
        return False, str(e)


def test_trainer_early_stopping() -> Tuple[bool, str]:
    """Test Trainer with early stopping."""
    try:
        np.random.seed(42)
        
        X = np.random.randn(100, 4)
        Y = np.random.randn(100, 2)
        dataset = Dataset(X, Y)
        loader = DataLoader(dataset, batch_size=32)
        
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2)
        )
        optimizer = Adam(model.parameters())
        criterion = MSELoss()
        
        trainer = Trainer(model, optimizer, criterion)
        history = trainer.fit(
            loader, 
            epochs=1000,  # High number
            early_stopping_patience=5,
            verbose=False
        )
        
        if len(history['train_loss']) >= 1000:
            return False, "Early stopping didn't trigger"
        
        return True, f"Stopped at epoch {len(history['train_loss'])}"
    except Exception as e:
        return False, str(e)


def test_complete_training_pipeline() -> Tuple[bool, str]:
    """Test complete training pipeline."""
    try:
        np.random.seed(42)
        
        X = np.random.randn(200, 10)
        Y = (X @ np.random.randn(10, 3)) + np.random.randn(200, 3) * 0.1
        
        train_dataset = Dataset(X[:160], Y[:160])
        val_dataset = Dataset(X[160:], Y[160:])
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        model = Sequential(
            Linear(10, 32),
            ReLU(),
            Linear(32, 16),
            ReLU(),
            Linear(16, 3)
        )
        
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = MSELoss()
        
        trainer = Trainer(model, optimizer, criterion)
        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=50,
            early_stopping_patience=10,
            verbose=False
        )
        
        final_loss = history['train_loss'][-1]
        initial_loss = history['train_loss'][0]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        if improvement < 50:
            return False, f"Only {improvement:.1f}% improvement"
        
        return True, f"Pipeline works: {improvement:.1f}% improvement"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("dataset_creation", test_dataset_creation),
        ("dataset_length", test_dataset_length),
        ("dataset_getitem", test_dataset_getitem),
        ("dataloader_creation", test_dataloader_creation),
        ("dataloader_length", test_dataloader_length),
        ("dataloader_iteration", test_dataloader_iteration),
        ("dataloader_shuffle", test_dataloader_shuffle),
        ("mse_loss", test_mse_loss),
        ("mse_loss_backward", test_mse_loss_backward),
        ("train_epoch", test_train_epoch),
        ("validate_epoch", test_validate_epoch),
        ("checkpoint_save_load", test_checkpoint_save_load),
        ("early_stopping_creation", test_early_stopping_creation),
        ("early_stopping_behavior", test_early_stopping_behavior),
        ("early_stopping_best", test_early_stopping_best),
        ("trainer_creation", test_trainer_creation),
        ("trainer_fit", test_trainer_fit),
        ("trainer_with_validation", test_trainer_with_validation),
        ("trainer_early_stopping", test_trainer_early_stopping),
        ("complete_training_pipeline", test_complete_training_pipeline),
    ]
    
    print(f"\n{'='*60}")
    print("Day 30: Training Loop Infrastructure - Tests")
    print(f"{'='*60}")
    
    passed = 0
    for name, fn in tests:
        try:
            p, m = fn()
        except Exception as e:
            p, m = False, str(e)
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    
    print(f"\nSummary: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n" + "="*60)
        print("CONGRATULATIONS!")
        print("You've completed the AutoDiff training framework!")
        print("="*60)
        print("\nYou now have a complete mini deep learning library with:")
        print("  - Automatic differentiation (Tensor)")
        print("  - Neural network modules (Module, Linear, ReLU, etc.)")
        print("  - Loss functions (MSE, CrossEntropy)")
        print("  - Optimizers (SGD, Adam, AdamW)")
        print("  - Learning rate schedulers")
        print("  - Gradient clipping and regularization")
        print("  - DataLoader and training infrastructure")
        print("\nNext steps:")
        print("  - Add more layers (Conv2D, BatchNorm, LSTM)")
        print("  - Implement GPU support")
        print("  - Build real projects!")
        print("="*60)
    
    return passed == len(tests)


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    success = run_all_tests()
    sys.exit(0 if success else 1)
