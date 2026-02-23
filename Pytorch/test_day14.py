"""Test Suite for Day 14: Positional Encoding"""

import torch
import torch.nn as nn
import math
from typing import Tuple

try:
    from day14 import (create_sinusoidal_encoding, SinusoidalPositionalEncoding,
                       LearnedPositionalEncoding, create_relative_position_bias,
                       compute_relative_positions, compute_rope_frequencies,
                       apply_rope, RoPEPositionalEncoding, compare_encoding_properties,
                       TransformerEmbedding)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_sinusoidal_encoding_shape() -> Tuple[bool, str]:
    """Test sinusoidal encoding has correct shape."""
    try:
        max_seq_len, d_model = 100, 64
        
        pe = create_sinusoidal_encoding(max_seq_len, d_model)
        
        if pe.shape != torch.Size([100, 64]):
            return False, f"Expected shape [100, 64], got {list(pe.shape)}"
        
        # Check it's not all zeros (implemented)
        if pe.abs().sum() < 1e-6:
            return False, "Not implemented (all zeros)"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_sinusoidal_encoding_values() -> Tuple[bool, str]:
    """Test sinusoidal encoding values are correct."""
    try:
        max_seq_len, d_model = 50, 32
        
        pe = create_sinusoidal_encoding(max_seq_len, d_model)
        
        # Position 0 should have sin(0)=0 for even dims, cos(0)=1 for odd dims
        if not torch.allclose(pe[0, 0::2], torch.zeros(d_model // 2), atol=1e-5):
            return False, "Position 0 even dims should be 0 (sin(0))"
        
        if not torch.allclose(pe[0, 1::2], torch.ones(d_model // 2), atol=1e-5):
            return False, "Position 0 odd dims should be 1 (cos(0))"
        
        # Check values are in [-1, 1]
        if pe.max() > 1.0 or pe.min() < -1.0:
            return False, "Values should be in [-1, 1]"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_sinusoidal_module() -> Tuple[bool, str]:
    """Test SinusoidalPositionalEncoding module."""
    try:
        d_model, max_seq_len = 64, 100
        
        pe_module = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout=0.0)
        
        if pe_module.pe is None:
            return False, "Not implemented"
        
        x = torch.randn(4, 20, d_model)
        out = pe_module(x)
        
        # Output should be different from input (positional encoding added)
        if torch.allclose(out, x):
            return False, "Output same as input"
        
        # Shape should be preserved
        if out.shape != x.shape:
            return False, f"Shape changed: {x.shape} -> {out.shape}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_learned_positional_encoding() -> Tuple[bool, str]:
    """Test LearnedPositionalEncoding module."""
    try:
        d_model, max_seq_len = 64, 100
        
        pe_module = LearnedPositionalEncoding(d_model, max_seq_len, dropout=0.0)
        
        if pe_module.position_embedding is None:
            return False, "Not implemented"
        
        x = torch.randn(4, 20, d_model)
        out = pe_module(x)
        
        # Output should be different from input
        if torch.allclose(out, x):
            return False, "Output same as input"
        
        # Should have learnable parameters
        num_params = sum(p.numel() for p in pe_module.parameters())
        expected_params = max_seq_len * d_model
        if num_params != expected_params:
            return False, f"Expected {expected_params} params, got {num_params}"
        
        return True, f"OK ({num_params} params)"
    except Exception as e:
        return False, str(e)


def test_relative_positions() -> Tuple[bool, str]:
    """Test relative position computation."""
    try:
        seq_len = 4
        
        rel_pos = compute_relative_positions(seq_len)
        
        expected = torch.tensor([
            [ 0,  1,  2,  3],
            [-1,  0,  1,  2],
            [-2, -1,  0,  1],
            [-3, -2, -1,  0]
        ])
        
        if not torch.equal(rel_pos, expected):
            return False, f"Expected:\n{expected}\nGot:\n{rel_pos}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_relative_position_bias() -> Tuple[bool, str]:
    """Test relative position bias creation."""
    try:
        seq_len, num_heads = 16, 8
        
        bias = create_relative_position_bias(seq_len, num_heads)
        
        # Should be a Parameter
        if not isinstance(bias, nn.Parameter):
            return False, "Should be nn.Parameter"
        
        # Check shape
        if bias.shape != torch.Size([num_heads, seq_len, seq_len]):
            return False, f"Expected shape [{num_heads}, {seq_len}, {seq_len}], got {list(bias.shape)}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_rope_frequencies() -> Tuple[bool, str]:
    """Test RoPE frequency computation."""
    try:
        dim, max_seq_len = 64, 100
        
        freqs = compute_rope_frequencies(dim, max_seq_len)
        
        if freqs.shape != torch.Size([max_seq_len, dim // 2]):
            return False, f"Expected shape [{max_seq_len}, {dim // 2}], got {list(freqs.shape)}"
        
        # Position 0 should have all zeros
        if not torch.allclose(freqs[0], torch.zeros(dim // 2)):
            return False, "Position 0 should have all zeros"
        
        # Frequencies should increase with position
        if not (freqs[1, 0] > freqs[0, 0]):
            return False, "Frequencies should increase with position"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_apply_rope() -> Tuple[bool, str]:
    """Test RoPE application."""
    try:
        batch, seq_len, dim = 4, 20, 64
        
        x = torch.randn(batch, seq_len, dim)
        freqs = compute_rope_frequencies(dim, seq_len)
        
        # Check freqs is not all zeros (implemented)
        if freqs.abs().sum() < 1e-6:
            return False, "Frequencies not implemented"
        
        rotated = apply_rope(x, freqs)
        
        if rotated.shape != x.shape:
            return False, f"Shape changed: {x.shape} -> {rotated.shape}"
        
        # Output should be different from input
        if torch.allclose(rotated, x):
            return False, "Output same as input"
        
        # Position 0 should be unchanged (rotation by 0)
        # Actually position 0 has freq*0=0, so sin=0, cos=1
        # x_rotated = x * cos(0) - x * sin(0) = x
        if not torch.allclose(rotated[:, 0, :], x[:, 0, :], atol=1e-5):
            return False, "Position 0 should be unchanged"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_rope_module() -> Tuple[bool, str]:
    """Test RoPEPositionalEncoding module."""
    try:
        dim, max_seq_len = 64, 100
        
        rope = RoPEPositionalEncoding(dim, max_seq_len)
        
        if rope.freqs is None:
            return False, "Not implemented"
        
        x = torch.randn(4, 20, dim)
        out = rope(x)
        
        if out.shape != x.shape:
            return False, f"Shape changed: {x.shape} -> {out.shape}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_transformer_embedding() -> Tuple[bool, str]:
    """Test TransformerEmbedding module."""
    try:
        vocab_size, d_model, max_seq_len = 10000, 128, 512
        
        emb = TransformerEmbedding(vocab_size, d_model, max_seq_len, 
                                   dropout=0.0, use_learned_pos=False)
        
        if emb.token_embedding is None:
            return False, "Not implemented"
        
        tokens = torch.randint(0, vocab_size, (4, 32))
        out = emb(tokens)
        
        if out.shape != torch.Size([4, 32, d_model]):
            return False, f"Expected shape [4, 32, {d_model}], got {list(out.shape)}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_sinusoidal_extrapolation() -> Tuple[bool, str]:
    """Test that sinusoidal encoding can extrapolate to longer sequences."""
    try:
        d_model, train_len, test_len = 64, 100, 200
        
        pe_module = SinusoidalPositionalEncoding(d_model, train_len, dropout=0.0)
        
        if pe_module.pe is None:
            return False, "Not implemented"
        
        # Create encoding for longer sequence
        pe_long = create_sinusoidal_encoding(test_len, d_model)
        
        # Should produce valid values (not NaN/inf)
        if torch.isnan(pe_long).any() or torch.isinf(pe_long).any():
            return False, "NaN or inf values in extrapolated encoding"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def test_encoding_comparison() -> Tuple[bool, str]:
    """Test comparison of encoding methods."""
    try:
        result = compare_encoding_properties()
        
        if result['sinusoidal_params'] == 0 and result['learned_params'] == 0:
            return False, "Not implemented"
        
        # Learned should have parameters, sinusoidal should not
        # (excluding dropout which both might have)
        if result['learned_params'] <= 0:
            return False, "Learned encoding should have parameters"
        
        return True, f"OK (sin={result['sinusoidal_params']}, learn={result['learned_params']})"
    except Exception as e:
        return False, str(e)


def test_positional_encoding_gradient() -> Tuple[bool, str]:
    """Test gradient flow through positional encodings."""
    try:
        d_model, seq_len = 32, 16
        
        # Test learned encoding gradient
        learned = LearnedPositionalEncoding(d_model, seq_len, dropout=0.0)
        
        if learned.position_embedding is None:
            return False, "Learned encoding not implemented"
        
        x = torch.randn(4, seq_len, d_model, requires_grad=True)
        out = learned(x)
        loss = out.sum()
        loss.backward()
        
        if x.grad is None:
            return False, "No gradient for input"
        
        if learned.position_embedding.weight.grad is None:
            return False, "No gradient for position embeddings"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("sinusoidal_shape", test_sinusoidal_encoding_shape),
        ("sinusoidal_values", test_sinusoidal_encoding_values),
        ("sinusoidal_module", test_sinusoidal_module),
        ("learned_encoding", test_learned_positional_encoding),
        ("relative_positions", test_relative_positions),
        ("relative_position_bias", test_relative_position_bias),
        ("rope_frequencies", test_rope_frequencies),
        ("apply_rope", test_apply_rope),
        ("rope_module", test_rope_module),
        ("transformer_embedding", test_transformer_embedding),
        ("sinusoidal_extrapolation", test_sinusoidal_extrapolation),
        ("encoding_comparison", test_encoding_comparison),
        ("gradient_flow", test_positional_encoding_gradient),
    ]
    
    print(f"\n{'='*50}\nDay 14: Positional Encoding - Tests\n{'='*50}")
    
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
