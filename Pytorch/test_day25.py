"""Test Suite for Day 25: Advanced Attention Patterns"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
try:
    from day25 import (
        create_sliding_window_mask, SlidingWindowAttention,
        create_dilated_mask, create_block_sparse_mask, BlockSparseAttention,
        AttentionWithSinks, LocalGlobalAttention,
        compute_attention_memory, analyze_attention_patterns
    )
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_sliding_window_mask_shape():
    """Test sliding window mask shape."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len, window_size = 16, 4
    mask = create_sliding_window_mask(seq_len, window_size)
    
    assert mask.shape == (seq_len, seq_len), f"Shape {mask.shape} != {(seq_len, seq_len)}"

def test_sliding_window_mask_causal():
    """Test that causal sliding window mask is lower triangular-ish."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len, window_size = 8, 3
    mask = create_sliding_window_mask(seq_len, window_size, causal=True)
    
    upper = torch.triu(mask, diagonal=1)
    assert upper.sum() <= 0, "Causal mask has non-zero upper triangle"
    
    diag = torch.diag(mask)
    assert torch.all(diag == 1), "Diagonal should be 1"

def test_sliding_window_mask_window():
    """Test that window size is enforced."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len, window_size = 10, 3
    mask = create_sliding_window_mask(seq_len, window_size, causal=True)
    
    assert mask[7, 0] == 0, "Position outside window should be masked"
    assert mask[7, 7] == 1, "Current position should not be masked"
    assert mask[7, 4] == 1, "Position within window should not be masked"

def test_sliding_window_mask_bidirectional():
    """Test bidirectional sliding window mask."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len, window_size = 8, 2
    mask = create_sliding_window_mask(seq_len, window_size, causal=False)
    
    for i in range(2, 7):
        assert mask[4, i] == 1, f"Position {i} should be attended from position 4"
    
    assert mask[4, 0] == 0, "Position 0 should be masked from position 4"

def test_sliding_window_attention_init():
    """Test SlidingWindowAttention initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, window = 128, 4, 8
    attn = SlidingWindowAttention(d_model, num_heads, window)
    
    assert attn.W_q is not None, "W_q not initialized"
    assert attn.W_k is not None, "W_k not initialized"
    assert attn.window_size == window, "window_size not set"

def test_sliding_window_attention_forward():
    """Test SlidingWindowAttention forward pass with window pattern verification."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, window = 128, 4, 8
    attn = SlidingWindowAttention(d_model, num_heads, window)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    batch, seq_len = 2, 16
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    
    output, weights = attn(x)
    
    assert output.shape == x.shape, f"Output shape {output.shape} != {x.shape}"
    
    expected_weights = (batch, num_heads, seq_len, seq_len)
    assert weights.shape == expected_weights, "Weights shape wrong"
    
    for i in range(seq_len):
        assert weights[0, 0, i, i] >= 1e-6, f"Position {i} should attend to self"
        if i < seq_len - 1:
            future_attn = weights[0, 0, i, i+1:].sum()
            assert future_attn <= 1e-5, f"Position {i} should not attend to future positions"
    
    assert output.abs().sum() != 0, "Output is all zeros"
    
    attn_sums = weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4), "Attention weights don't sum to 1"

def test_sliding_window_attention_sparse():
    """Test that sliding window attention has sparse pattern."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, window = 128, 4, 4
    attn = SlidingWindowAttention(d_model, num_heads, window)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    batch, seq_len = 1, 32
    x = torch.randn(batch, seq_len, d_model)
    
    _, weights = attn(x)
    
    zero_ratio = (weights < 1e-5).float().mean()
    
    assert zero_ratio >= 0.5, f"Not sparse enough: {zero_ratio:.2%} zeros"

def test_dilated_mask_shape():
    """Test dilated attention mask shape."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len, window, dilation = 16, 4, 2
    mask = create_dilated_mask(seq_len, window, dilation)
    
    assert mask.shape == (seq_len, seq_len), f"Shape {mask.shape} != {(seq_len, seq_len)}"

def test_dilated_mask_pattern():
    """Test dilated mask attends to strided positions."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len, window, dilation = 16, 3, 2
    mask = create_dilated_mask(seq_len, window, dilation, causal=True)
    
    assert mask[10, 10] == 1, "Should attend to self"
    assert mask[10, 8] == 1, "Should attend to position 8 (dilation step)"
    assert mask[10, 6] == 1, "Should attend to position 6 (2 dilation steps)"
    
    assert mask[10, 9] == 0, "Should NOT attend to position 9 (not on stride)"

def test_block_sparse_mask():
    """Test block sparse attention mask."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len, block_size = 16, 4
    mask = create_block_sparse_mask(seq_len, block_size, num_global_blocks=1)
    
    assert mask.shape == (seq_len, seq_len), "Shape wrong"
    
    assert mask[0, 0] == 1 and mask[5, 5] == 1, "Diagonal blocks should be attended"
    
    assert mask[0, 15] == 1, "Global block should attend everywhere"

def test_block_sparse_attention():
    """Test BlockSparseAttention produces valid output."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, block_size = 128, 4, 8
    attn = BlockSparseAttention(d_model, num_heads, block_size)
    
    assert attn.W_q is not None, "BlockSparseAttention not initialized"
    
    batch, seq_len = 2, 32
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    
    output, weights = attn(x)
    
    assert output.shape == x.shape, "Output shape wrong"
    
    assert output.abs().sum() != 0, "Output is all zeros"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Output contains NaN or Inf"
    
    attn_sums = weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4), "Attention weights don't sum to 1"
    
    zero_ratio = (weights < 1e-5).float().mean()
    assert zero_ratio >= 0.3, f"Block sparse should have some sparsity, got {zero_ratio:.2%} zeros"

def test_attention_sinks_init():
    """Test AttentionWithSinks initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 128, 4
    attn = AttentionWithSinks(d_model, num_heads, num_sink_tokens=4, window_size=16)
    
    assert attn.W_q is not None, "W_q not initialized"
    assert attn.num_sink_tokens == 4, "num_sink_tokens not set"

def test_attention_sinks_mask():
    """Test sink attention mask pattern."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 128, 4
    attn = AttentionWithSinks(d_model, num_heads, num_sink_tokens=4, window_size=8)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    seq_len = 20
    mask = attn.create_sink_window_mask(seq_len)
    
    assert mask[15, 0] == 1, "Should attend to sink token 0"
    assert mask[15, 3] == 1, "Should attend to sink token 3"
    
    assert mask[15, 15] == 1, "Should attend to self"
    assert mask[15, 8] == 1, "Should attend to position in window"

def test_attention_sinks_forward():
    """Test AttentionWithSinks forward pass with sink attention verification."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads = 128, 4
    num_sinks = 4
    attn = AttentionWithSinks(d_model, num_heads, num_sink_tokens=num_sinks, window_size=16)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    batch, seq_len = 2, 32
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    
    output, weights = attn(x)
    
    assert output.shape == x.shape, "Output shape wrong"
    
    assert output.abs().sum() != 0, "Output is all zeros"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Output contains NaN or Inf"
    
    attn_sums = weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4), "Attention weights don't sum to 1"
    
    sink_attn = weights[0, 0, -1, :num_sinks].sum()
    assert sink_attn >= 0.01, f"Sink tokens should receive some attention, got {sink_attn:.4f}"

def test_local_global_attention():
    """Test LocalGlobalAttention with global token verification."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, local_window = 128, 4, 8
    attn = LocalGlobalAttention(d_model, num_heads, local_window)
    
    assert attn.W_q is not None, "LocalGlobalAttention not initialized"
    
    batch, seq_len = 2, 32
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)
    global_indices = torch.tensor([0])
    
    output, weights = attn(x, global_indices)
    
    assert output.shape == x.shape, "Output shape wrong"
    
    assert output.abs().sum() != 0, "Output is all zeros"
    
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "Output contains NaN or Inf"
    
    attn_sums = weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4), "Attention weights don't sum to 1"
    
    global_attn_received = weights[0, 0, :, 0].mean()
    assert global_attn_received >= 0.01, f"Global token should receive attention, got mean={global_attn_received:.4f}"

def test_local_global_mask():
    """Test local + global mask pattern."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, local_window = 128, 4, 4
    attn = LocalGlobalAttention(d_model, num_heads, local_window)
    
    seq_len = 16
    global_indices = torch.tensor([0])
    mask = attn.create_local_global_mask(seq_len, global_indices)
    
    assert torch.all(mask[0, :] == 1), "Global token should attend to all"
    
    assert torch.all(mask[:, 0] == 1), "All should attend to global token"

def test_memory_comparison():
    """Test memory comparison function."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    mem = compute_attention_memory(8192, num_heads=32, head_dim=128)
    
    assert mem['standard_bytes'] != 0, "Standard memory not computed"
    assert mem['sliding_window_bytes'] != 0, "Sliding window memory not computed"
    
    assert mem['sliding_window_bytes'] <= mem['standard_bytes'], "Sliding window should use less memory"

def test_attention_pattern_analysis():
    """Test attention pattern analysis."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, heads, seq = 1, 4, 16
    
    logits = torch.randn(batch, heads, seq, seq)
    weights = F.softmax(logits, dim=-1)
    
    stats = analyze_attention_patterns(weights)
    
    assert 'entropy' in stats, "Missing entropy"
    assert 'sink_attention' in stats, "Missing sink_attention"

def test_gradient_flow_sliding():
    """Test gradient flow through sliding window attention."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, num_heads, window = 128, 4, 8
    attn = SlidingWindowAttention(d_model, num_heads, window)
    
    assert attn.W_q is not None, "Attention not initialized"
    
    x = torch.randn(2, 16, d_model, requires_grad=True)
    output, _ = attn(x)
    
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient to input"
    assert x.grad.abs().sum() != 0, "Gradients are zero"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
