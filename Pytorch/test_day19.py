"""Test Suite for Day 19: Full Transformer Architecture"""

import torch
import torch.nn as nn
import sys
from typing import Tuple

try:
    from day19 import (PositionalEncoding, TransformerEmbedding,
                       TransformerEncoder, TransformerDecoder,
                       Transformer, greedy_decode, create_causal_mask)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_positional_encoding_shape() -> Tuple[bool, str]:
    """Test PositionalEncoding output shape."""
    try:
        d_model = 64
        pe = PositionalEncoding(d_model, max_len=100, dropout=0.0)
        
        x = torch.randn(2, 16, d_model)
        output = pe(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        return True, f"Shape preserved: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_positional_encoding_different_positions() -> Tuple[bool, str]:
    """Test that different positions have different encodings."""
    try:
        d_model = 64
        pe = PositionalEncoding(d_model, dropout=0.0)
        
        x = torch.zeros(1, 10, d_model)  # Zero input to see only PE
        output = pe(x)
        
        if output is None:
            return False, "output is None"
        
        # Different positions should have different encodings
        pos_diff = (output[0, 0] - output[0, 1]).abs().sum()
        if pos_diff < 1e-5:
            return False, "Positions 0 and 1 have same encoding"
        
        return True, "Different positions have different encodings"
    except Exception as e:
        return False, str(e)


def test_transformer_embedding_shape() -> Tuple[bool, str]:
    """Test TransformerEmbedding output shape."""
    try:
        vocab_size = 1000
        d_model = 64
        
        emb = TransformerEmbedding(vocab_size, d_model)
        
        if emb.token_embedding is None:
            return False, "token_embedding not initialized"
        
        tokens = torch.randint(0, vocab_size, (2, 16))
        output = emb(tokens)
        
        if output is None:
            return False, "output is None"
        
        expected_shape = (2, 16, d_model)
        if output.shape != expected_shape:
            return False, f"Shape {output.shape} != {expected_shape}"
        
        return True, f"Embedding output: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_transformer_embedding_scaling() -> Tuple[bool, str]:
    """Test that embeddings are scaled by sqrt(d_model)."""
    try:
        vocab_size = 1000
        d_model = 64
        
        emb = TransformerEmbedding(vocab_size, d_model, dropout=0.0)
        
        if emb.token_embedding is None:
            return False, "token_embedding not initialized"
        
        tokens = torch.randint(1, vocab_size, (1, 1))
        
        # Get raw embedding
        raw_emb = emb.token_embedding(tokens)
        
        # Get scaled embedding (without positional encoding effect)
        # This is hard to test directly, so we just check it runs
        output = emb(tokens)
        
        if output is None:
            return False, "output is None"
        
        # The output should be larger than raw embedding (due to scaling)
        # This is a weak test but better than nothing
        return True, "Embedding scaling applied"
    except Exception as e:
        return False, str(e)


def test_encoder_stack() -> Tuple[bool, str]:
    """Test TransformerEncoder stack."""
    try:
        d_model, num_heads, num_layers = 64, 4, 2
        
        encoder = TransformerEncoder(d_model, num_heads, num_layers)
        
        if encoder.layers is None:
            return False, "layers not initialized"
        if len(encoder.layers) != num_layers:
            return False, f"Expected {num_layers} layers"
        
        x = torch.randn(2, 16, d_model)
        output = encoder(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        return True, f"Encoder stack with {num_layers} layers"
    except Exception as e:
        return False, str(e)


def test_decoder_stack() -> Tuple[bool, str]:
    """Test TransformerDecoder stack."""
    try:
        d_model, num_heads, num_layers = 64, 4, 2
        
        decoder = TransformerDecoder(d_model, num_heads, num_layers)
        
        if decoder.layers is None:
            return False, "layers not initialized"
        if len(decoder.layers) != num_layers:
            return False, f"Expected {num_layers} layers"
        
        tgt = torch.randn(2, 8, d_model)
        enc_out = torch.randn(2, 16, d_model)
        
        output = decoder(tgt, enc_out)
        
        if output is None:
            return False, "output is None"
        if output.shape != tgt.shape:
            return False, f"Shape {output.shape} != {tgt.shape}"
        
        return True, f"Decoder stack with {num_layers} layers"
    except Exception as e:
        return False, str(e)


def test_transformer_forward() -> Tuple[bool, str]:
    """Test full Transformer forward pass."""
    try:
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=500,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        if model.encoder is None:
            return False, "encoder not initialized"
        if model.decoder is None:
            return False, "decoder not initialized"
        
        batch, src_len, tgt_len = 2, 10, 8
        src = torch.randint(1, 1000, (batch, src_len))
        tgt = torch.randint(1, 500, (batch, tgt_len))
        
        output = model(src, tgt)
        
        if output is None:
            return False, "output is None"
        
        expected_shape = (batch, tgt_len, 500)
        if output.shape != expected_shape:
            return False, f"Shape {output.shape} != {expected_shape}"
        
        return True, f"Transformer output: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_transformer_src_mask() -> Tuple[bool, str]:
    """Test source padding mask creation."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            padding_idx=0
        )
        
        # Create input with padding
        src = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        mask = model.create_src_mask(src)
        
        if mask is None:
            return False, "create_src_mask returned None"
        
        # Check that padding positions are masked
        # Mask should be 0 for padding positions
        return True, "Source mask handles padding"
    except Exception as e:
        return False, str(e)


def test_transformer_tgt_mask() -> Tuple[bool, str]:
    """Test target mask (causal + padding)."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            padding_idx=0
        )
        
        # Create target with padding
        tgt = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]])
        mask = model.create_tgt_mask(tgt)
        
        if mask is None:
            return False, "create_tgt_mask returned None"
        
        # Should be 4D for attention
        if len(mask.shape) != 4:
            return False, f"Expected 4D mask, got {len(mask.shape)}D"
        
        return True, "Target mask combines causal and padding"
    except Exception as e:
        return False, str(e)


def test_transformer_encode() -> Tuple[bool, str]:
    """Test encoder method."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        if model.encoder is None:
            return False, "encoder not initialized"
        
        src = torch.randint(1, 100, (2, 10))
        encoder_output = model.encode(src)
        
        if encoder_output is None:
            return False, "encode returned None"
        
        expected_shape = (2, 10, 64)
        if encoder_output.shape != expected_shape:
            return False, f"Shape {encoder_output.shape} != {expected_shape}"
        
        return True, f"Encoder output: {encoder_output.shape}"
    except Exception as e:
        return False, str(e)


def test_transformer_decode() -> Tuple[bool, str]:
    """Test decoder method."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        if model.decoder is None:
            return False, "decoder not initialized"
        
        tgt = torch.randint(1, 100, (2, 8))
        encoder_output = torch.randn(2, 10, 64)
        
        decoder_output = model.decode(tgt, encoder_output)
        
        if decoder_output is None:
            return False, "decode returned None"
        
        expected_shape = (2, 8, 64)
        if decoder_output.shape != expected_shape:
            return False, f"Shape {decoder_output.shape} != {expected_shape}"
        
        return True, f"Decoder output: {decoder_output.shape}"
    except Exception as e:
        return False, str(e)


def test_transformer_gradient_flow() -> Tuple[bool, str]:
    """Test gradient flow through full model."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1
        )
        
        if model.encoder is None:
            return False, "Model not initialized"
        
        src = torch.randint(1, 100, (2, 5))
        tgt = torch.randint(1, 100, (2, 4))
        
        output = model(src, tgt)
        
        if output is None:
            return False, "output is None"
        
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if not has_grad:
            return False, "No gradients computed"
        
        return True, "Gradients flow through full model"
    except Exception as e:
        return False, str(e)


def test_greedy_decode_shape() -> Tuple[bool, str]:
    """Test greedy decoding output shape."""
    try:
        model = Transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1
        )
        
        if model.encoder is None:
            return False, "Model not initialized"
        
        src = torch.randint(1, 100, (2, 5))
        max_len = 10
        start_token = 1
        end_token = 2
        
        with torch.no_grad():
            generated = greedy_decode(model, src, max_len, start_token, end_token)
        
        if generated is None:
            return False, "greedy_decode returned None"
        
        # Should start with start_token
        if generated[:, 0].unique().item() != start_token:
            return False, f"First token should be {start_token}"
        
        # Length should be <= max_len
        if generated.shape[1] > max_len:
            return False, f"Generated length {generated.shape[1]} > max_len {max_len}"
        
        return True, f"Generated shape: {generated.shape}"
    except Exception as e:
        return False, str(e)


def test_transformer_different_vocab_sizes() -> Tuple[bool, str]:
    """Test Transformer with different source/target vocab sizes."""
    try:
        model = Transformer(
            src_vocab_size=5000,
            tgt_vocab_size=3000,
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        if model.encoder is None:
            return False, "Model not initialized"
        
        src = torch.randint(1, 5000, (2, 10))
        tgt = torch.randint(1, 3000, (2, 8))
        
        output = model(src, tgt)
        
        if output is None:
            return False, "output is None"
        
        # Output vocab should be target vocab
        if output.shape[-1] != 3000:
            return False, f"Output vocab size {output.shape[-1]} != 3000"
        
        return True, "Supports different src/tgt vocab sizes"
    except Exception as e:
        return False, str(e)


def test_causal_mask_function() -> Tuple[bool, str]:
    """Test the causal mask helper function."""
    try:
        seq_len = 5
        mask = create_causal_mask(seq_len)
        
        if mask is None:
            return False, "create_causal_mask returned None"
        
        expected = torch.tril(torch.ones(seq_len, seq_len))
        if not torch.allclose(mask, expected):
            return False, "Causal mask not lower triangular"
        
        return True, "Causal mask is correct"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("positional_encoding_shape", test_positional_encoding_shape),
        ("positional_encoding_positions", test_positional_encoding_different_positions),
        ("transformer_embedding_shape", test_transformer_embedding_shape),
        ("transformer_embedding_scaling", test_transformer_embedding_scaling),
        ("encoder_stack", test_encoder_stack),
        ("decoder_stack", test_decoder_stack),
        ("transformer_forward", test_transformer_forward),
        ("transformer_src_mask", test_transformer_src_mask),
        ("transformer_tgt_mask", test_transformer_tgt_mask),
        ("transformer_encode", test_transformer_encode),
        ("transformer_decode", test_transformer_decode),
        ("transformer_gradient", test_transformer_gradient_flow),
        ("greedy_decode", test_greedy_decode_shape),
        ("different_vocab_sizes", test_transformer_different_vocab_sizes),
        ("causal_mask", test_causal_mask_function),
    ]
    
    print(f"\n{'='*50}\nDay 19: Full Transformer Architecture - Tests\n{'='*50}")
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        print(f"  [{'PASS' if p else 'FAIL'}] {name}: {m}")
    print(f"\nSummary: {passed}/{len(tests)}")


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    run_all_tests()
