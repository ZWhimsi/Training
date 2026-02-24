"""Test Suite for Day 14: Cross-Entropy Loss
Run: pytest test_day14.py -v
"""

import pytest
import torch
import torch.nn.functional as F

CUDA_AVAILABLE = torch.cuda.is_available()

if CUDA_AVAILABLE:
    try:
        from day14 import (log_softmax_rows, cross_entropy_loss, 
                           cross_entropy_mean, cross_entropy_smooth)
        IMPORT_SUCCESS = True
    except ImportError as e:
        IMPORT_SUCCESS = False
        IMPORT_ERROR = str(e)
else:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = "CUDA not available"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day14")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_log_softmax():
    """Test log softmax on rows."""
    x = torch.randn(32, 64, device='cuda')
    result = log_softmax_rows(x)
    expected = F.log_softmax(x, dim=-1)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day14")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cross_entropy_basic():
    """Test basic cross-entropy loss."""
    batch_size, n_classes = 16, 10
    logits = torch.randn(batch_size, n_classes, device='cuda')
    targets = torch.randint(0, n_classes, (batch_size,), device='cuda')
    
    result = cross_entropy_loss(logits, targets)
    expected = F.cross_entropy(logits, targets, reduction='none')
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-4, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day14")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_cross_entropy_mean():
    """Test mean cross-entropy loss."""
    batch_size, n_classes = 16, 10
    logits = torch.randn(batch_size, n_classes, device='cuda')
    targets = torch.randint(0, n_classes, (batch_size,), device='cuda')
    
    result = cross_entropy_mean(logits, targets)
    assert result is not None, "Returned None"
    
    expected = F.cross_entropy(logits, targets)
    err = abs(result.item() - expected.item())
    assert err <= 1e-4, f"Error: {err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day14")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_label_smoothing():
    """Test label smoothing cross-entropy."""
    batch_size, n_classes = 16, 10
    logits = torch.randn(batch_size, n_classes, device='cuda')
    targets = torch.randint(0, n_classes, (batch_size,), device='cuda')
    smoothing = 0.1
    
    result = cross_entropy_smooth(logits, targets, smoothing)
    expected = F.cross_entropy(logits, targets, reduction='none', label_smoothing=smoothing)
    
    max_err = (result - expected).abs().max().item()
    assert max_err <= 1e-3, f"Error: {max_err:.6f}"


@pytest.mark.skipif(not IMPORT_SUCCESS, reason="Could not import from day14")
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_different_sizes():
    """Test different batch and class sizes."""
    for batch, classes in [(8, 5), (32, 100), (64, 1000)]:
        logits = torch.randn(batch, classes, device='cuda')
        targets = torch.randint(0, classes, (batch,), device='cuda')
        
        result = cross_entropy_loss(logits, targets)
        expected = F.cross_entropy(logits, targets, reduction='none')
        
        max_err = (result - expected).abs().max().item()
        assert max_err <= 1e-3, f"Failed at {batch}x{classes}: err={max_err:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
