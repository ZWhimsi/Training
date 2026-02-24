"""Test Suite for Day 14: Positional Encoding"""

import torch
import pytest
import torch.nn as nn
import math
try:
    from day14 import (create_sinusoidal_encoding, SinusoidalPositionalEncoding,
                       LearnedPositionalEncoding, create_relative_position_bias,
                       compute_relative_positions, compute_rope_frequencies,
                       apply_rope, RoPEPositionalEncoding, compare_encoding_properties,
                       TransformerEmbedding)
    IMPORT_SUCCESS = True
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)

def test_sinusoidal_encoding_shape():
    """Test sinusoidal encoding has correct shape and structure."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    max_seq_len, d_model = 100, 64
    
    pe = create_sinusoidal_encoding(max_seq_len, d_model)
    
    assert pe.shape == torch.Size([100, 64]), f"Expected shape [100, 64], got {list(pe.shape)}"
    
    assert pe.abs().sum() >= 1e-6, "Not implemented (all zeros)"
    
    expected_pe_1_0 = math.sin(1.0)
    assert torch.allclose(pe[1, 0], torch.tensor(expected_pe_1_0), atol=1e-5), f"PE[1,0] = {pe[1,0].item():.6f}, expected sin(1) = {expected_pe_1_0:.6f}"
    
    expected_pe_1_1 = math.cos(1.0)
    assert torch.allclose(pe[1, 1], torch.tensor(expected_pe_1_1), atol=1e-5), f"PE[1,1] = {pe[1,1].item():.6f}, expected cos(1) = {expected_pe_1_1:.6f}"

def test_sinusoidal_encoding_values():
    """Test sinusoidal encoding values are correct."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    max_seq_len, d_model = 50, 32
    
    pe = create_sinusoidal_encoding(max_seq_len, d_model)
    
    assert torch.allclose(pe[0, 0::2], torch.zeros(d_model // 2), atol=1e-5), "Position 0 even dims should be 0 (sin(0))"
    
    assert torch.allclose(pe[0, 1::2], torch.ones(d_model // 2), atol=1e-5), "Position 0 odd dims should be 1 (cos(0))"
    
    assert pe.max() <= 1.0 and pe.min() >= -1.0, "Values should be in [-1, 1]"

def test_sinusoidal_module():
    """Test SinusoidalPositionalEncoding module."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, max_seq_len = 64, 100
    
    pe_module = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout=0.0)
    
    assert pe_module.pe is not None, "Not implemented"
    
    x = torch.randn(4, 20, d_model)
    out = pe_module(x)
    
    assert not torch.allclose(out, x), "Output same as input"
    
    assert out.shape == x.shape, f"Shape changed: {x.shape} -> {out.shape}"
    
    pe_expected = create_sinusoidal_encoding(max_seq_len, d_model)
    expected_out = x + pe_expected[:20, :].unsqueeze(0)
    
    assert torch.allclose(out, expected_out, atol=1e-5), f"Output doesn't match x + PE: max diff {(out - expected_out).abs().max():.6f}"

def test_learned_positional_encoding():
    """Test LearnedPositionalEncoding module."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, max_seq_len = 64, 100
    
    pe_module = LearnedPositionalEncoding(d_model, max_seq_len, dropout=0.0)
    
    assert pe_module.position_embedding is not None, "Not implemented"
    
    x = torch.randn(4, 20, d_model)
    out = pe_module(x)
    
    assert not torch.allclose(out, x), "Output same as input"
    
    num_params = sum(p.numel() for p in pe_module.parameters())
    expected_params = max_seq_len * d_model
    assert num_params == expected_params, f"Expected {expected_params} params, got {num_params}"

def test_relative_positions():
    """Test relative position computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len = 4
    
    rel_pos = compute_relative_positions(seq_len)
    
    expected = torch.tensor([
        [ 0,  1,  2,  3],
        [-1,  0,  1,  2],
        [-2, -1,  0,  1],
        [-3, -2, -1,  0]
    ])
    
    assert torch.equal(rel_pos, expected), f"Expected:\n{expected}\nGot:\n{rel_pos}"

def test_relative_position_bias():
    """Test relative position bias creation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    seq_len, num_heads = 16, 8
    
    bias = create_relative_position_bias(seq_len, num_heads)
    
    assert isinstance(bias, nn.Parameter), "Should be nn.Parameter"
    
    assert bias.shape == torch.Size([num_heads, seq_len, seq_len]), f"Expected shape [{num_heads}, {seq_len}, {seq_len}], got {list(bias.shape)}"
    
    assert bias.requires_grad, "Bias should require gradients (be learnable)"
    
    attn_scores = torch.randn(2, num_heads, seq_len, seq_len)
    biased_scores = attn_scores + bias
    assert biased_scores.shape == attn_scores.shape, "Adding bias changes shape"

def test_rope_frequencies():
    """Test RoPE frequency computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim, max_seq_len = 64, 100
    
    freqs = compute_rope_frequencies(dim, max_seq_len)
    
    assert freqs.shape == torch.Size([max_seq_len, dim // 2]), f"Expected shape [{max_seq_len}, {dim // 2}], got {list(freqs.shape)}"
    
    assert torch.allclose(freqs[0], torch.zeros(dim // 2)), "Position 0 should have all zeros"
    
    assert freqs[1, 0] > freqs[0, 0], "Frequencies should increase with position"

def test_apply_rope():
    """Test RoPE application."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq_len, dim = 4, 20, 64
    
    x = torch.randn(batch, seq_len, dim)
    freqs = compute_rope_frequencies(dim, seq_len)
    
    assert freqs.abs().sum() >= 1e-6, "Frequencies not implemented"
    
    rotated = apply_rope(x, freqs)
    
    assert rotated.shape == x.shape, f"Shape changed: {x.shape} -> {rotated.shape}"
    
    assert not torch.allclose(rotated, x), "Output same as input"
    
    assert torch.allclose(rotated[:, 0, :], x[:, 0, :], atol=1e-5), "Position 0 should be unchanged"

def test_rope_module():
    """Test RoPEPositionalEncoding module."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    dim, max_seq_len = 64, 100
    
    rope = RoPEPositionalEncoding(dim, max_seq_len)
    
    assert rope.freqs is not None, "Not implemented"
    
    x = torch.randn(4, 20, dim)
    out = rope(x)
    
    assert out.shape == x.shape, f"Shape changed: {x.shape} -> {out.shape}"
    
    expected_out = apply_rope(x, rope.freqs)
    assert torch.allclose(out, expected_out, atol=1e-5), f"Output doesn't match apply_rope: max diff {(out - expected_out).abs().max():.6f}"
    
    assert torch.allclose(out[:, 0, :], x[:, 0, :], atol=1e-5), "Position 0 should be unchanged"

def test_transformer_embedding():
    """Test TransformerEmbedding module."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    vocab_size, d_model, max_seq_len = 10000, 128, 512
    
    emb = TransformerEmbedding(vocab_size, d_model, max_seq_len, 
                               dropout=0.0, use_learned_pos=False)
    
    assert emb.token_embedding is not None, "Not implemented"
    
    tokens = torch.randint(0, vocab_size, (4, 32))
    out = emb(tokens)
    
    assert out.shape == torch.Size([4, 32, d_model]), f"Expected shape [4, 32, {d_model}], got {list(out.shape)}"
    
    token_embed = emb.token_embedding(tokens) * math.sqrt(d_model)
    pe = create_sinusoidal_encoding(max_seq_len, d_model)
    expected = token_embed + pe[:32, :].unsqueeze(0)
    
    assert torch.allclose(out, expected, atol=1e-5), f"Output doesn't match expected: max diff {(out - expected).abs().max():.6f}"

def test_sinusoidal_extrapolation():
    """Test that sinusoidal encoding can extrapolate to longer sequences."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, train_len, test_len = 64, 100, 200
    
    pe_module = SinusoidalPositionalEncoding(d_model, train_len, dropout=0.0)
    
    assert pe_module.pe is not None, "Not implemented"
    
    pe_long = create_sinusoidal_encoding(test_len, d_model)
    
    assert not torch.isnan(pe_long).any() and not torch.isinf(pe_long).any(), "NaN or inf values in extrapolated encoding"

def test_encoding_comparison():
    """Test comparison of encoding methods."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    result = compare_encoding_properties()
    
    assert not (result['sinusoidal_params'] == 0 and result['learned_params'] == 0), "Not implemented"
    
    assert result['learned_params'] >= 0, "Learned encoding should have parameters"

def test_positional_encoding_gradient():
    """Test gradient flow through positional encodings."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    d_model, seq_len = 32, 16
    
    learned = LearnedPositionalEncoding(d_model, seq_len, dropout=0.0)
    
    assert learned.position_embedding is not None, "Learned encoding not implemented"
    
    x = torch.randn(4, seq_len, d_model, requires_grad=True)
    out = learned(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient for input"
    
    assert learned.position_embedding.weight.grad is not None, "No gradient for position embeddings"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
