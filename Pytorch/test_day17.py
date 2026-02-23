"""Test Suite for Day 17: Transformer Encoder Block"""

import torch
import torch.nn as nn
import sys
from typing import Tuple

try:
    from day17 import (LayerNorm, FeedForward, PostNormEncoderBlock, 
                       PreNormEncoderBlock, TransformerEncoder,
                       TransformerEncoderWithEmbedding, PositionalEncoding)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_layer_norm_shape() -> Tuple[bool, str]:
    """Test LayerNorm output shape."""
    try:
        d_model = 64
        norm = LayerNorm(d_model)
        
        if norm.gamma is None:
            return False, "gamma not initialized"
        if norm.beta is None:
            return False, "beta not initialized"
        
        x = torch.randn(2, 8, d_model)
        output = norm(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        return True, f"Shape preserved: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_layer_norm_stats() -> Tuple[bool, str]:
    """Test that LayerNorm produces correct mean/variance."""
    try:
        d_model = 64
        norm = LayerNorm(d_model)
        
        x = torch.randn(2, 8, d_model)
        output = norm(x)
        
        if output is None:
            return False, "output is None"
        
        # After normalization, mean should be ~0 and std ~1
        # (before scale/shift are applied, but with default gamma=1, beta=0)
        mean = output.mean(dim=-1)
        std = output.std(dim=-1, unbiased=False)
        
        # Should be close to 0 mean and 1 std
        if not torch.allclose(mean, torch.zeros_like(mean), atol=1e-5):
            return False, f"Mean not close to 0: {mean.mean().item():.4f}"
        if not torch.allclose(std, torch.ones_like(std), atol=1e-1):
            return False, f"Std not close to 1: {std.mean().item():.4f}"
        
        return True, "Mean~0, Std~1 after normalization"
    except Exception as e:
        return False, str(e)


def test_layer_norm_vs_pytorch() -> Tuple[bool, str]:
    """Compare custom LayerNorm with PyTorch's."""
    try:
        d_model = 64
        our_norm = LayerNorm(d_model)
        pytorch_norm = nn.LayerNorm(d_model)
        
        # Copy parameters
        with torch.no_grad():
            pytorch_norm.weight.copy_(our_norm.gamma)
            pytorch_norm.bias.copy_(our_norm.beta)
        
        x = torch.randn(2, 8, d_model)
        our_output = our_norm(x)
        pytorch_output = pytorch_norm(x)
        
        if our_output is None:
            return False, "output is None"
        
        if not torch.allclose(our_output, pytorch_output, atol=1e-5):
            return False, "Doesn't match PyTorch LayerNorm"
        
        return True, "Matches PyTorch nn.LayerNorm"
    except Exception as e:
        return False, str(e)


def test_feedforward_shape() -> Tuple[bool, str]:
    """Test FeedForward network shape."""
    try:
        d_model = 64
        ffn = FeedForward(d_model)
        
        if ffn.linear1 is None:
            return False, "linear1 not initialized"
        if ffn.linear2 is None:
            return False, "linear2 not initialized"
        
        x = torch.randn(2, 8, d_model)
        output = ffn(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        return True, f"Shape preserved: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_feedforward_expansion() -> Tuple[bool, str]:
    """Test that FFN has correct expansion factor."""
    try:
        d_model = 64
        ffn = FeedForward(d_model)
        
        if ffn.linear1 is None:
            return False, "linear1 not initialized"
        
        # Default d_ff should be 4 * d_model
        expected_d_ff = d_model * 4
        actual_d_ff = ffn.linear1.out_features
        
        if actual_d_ff != expected_d_ff:
            return False, f"d_ff={actual_d_ff}, expected {expected_d_ff}"
        
        return True, f"Expansion factor: {actual_d_ff // d_model}x"
    except Exception as e:
        return False, str(e)


def test_post_norm_encoder_block() -> Tuple[bool, str]:
    """Test PostNormEncoderBlock."""
    try:
        d_model, num_heads = 64, 4
        block = PostNormEncoderBlock(d_model, num_heads)
        
        if block.self_attn is None:
            return False, "self_attn not initialized"
        if block.norm1 is None:
            return False, "norm1 not initialized"
        
        x = torch.randn(2, 8, d_model)
        output = block(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        return True, "PostNormEncoderBlock works"
    except Exception as e:
        return False, str(e)


def test_pre_norm_encoder_block() -> Tuple[bool, str]:
    """Test PreNormEncoderBlock."""
    try:
        d_model, num_heads = 64, 4
        block = PreNormEncoderBlock(d_model, num_heads)
        
        if block.self_attn is None:
            return False, "self_attn not initialized"
        if block.norm1 is None:
            return False, "norm1 not initialized"
        
        x = torch.randn(2, 8, d_model)
        output = block(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        return True, "PreNormEncoderBlock works"
    except Exception as e:
        return False, str(e)


def test_residual_connection() -> Tuple[bool, str]:
    """Test that residual connections are working (gradient flows through)."""
    try:
        d_model, num_heads = 64, 4
        block = PreNormEncoderBlock(d_model, num_heads)
        
        if block.self_attn is None:
            return False, "Block not initialized"
        
        x = torch.randn(2, 8, d_model, requires_grad=True)
        output = block(x)
        
        if output is None:
            return False, "output is None"
        
        # Compute gradient
        loss = output.sum()
        loss.backward()
        
        # With residual connections, gradient should flow to input
        if x.grad is None:
            return False, "No gradient flow to input"
        if x.grad.abs().sum() == 0:
            return False, "Gradients are all zero"
        
        return True, "Residual connections enable gradient flow"
    except Exception as e:
        return False, str(e)


def test_encoder_stack() -> Tuple[bool, str]:
    """Test stacked encoder."""
    try:
        d_model, num_heads, num_layers = 64, 4, 4
        encoder = TransformerEncoder(d_model, num_heads, num_layers)
        
        if encoder.layers is None:
            return False, "layers not initialized"
        if len(encoder.layers) != num_layers:
            return False, f"Expected {num_layers} layers, got {len(encoder.layers)}"
        
        x = torch.randn(2, 8, d_model)
        output = encoder(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        return True, f"TransformerEncoder with {num_layers} layers works"
    except Exception as e:
        return False, str(e)


def test_encoder_with_embedding() -> Tuple[bool, str]:
    """Test complete encoder with embeddings."""
    try:
        vocab_size = 1000
        d_model, num_heads, num_layers = 64, 4, 2
        
        encoder = TransformerEncoderWithEmbedding(
            vocab_size, d_model, num_heads, num_layers
        )
        
        if encoder.token_emb is None:
            return False, "token_emb not initialized"
        if encoder.encoder is None:
            return False, "encoder not initialized"
        
        # Token indices
        x = torch.randint(0, vocab_size, (2, 16))
        output = encoder(x)
        
        if output is None:
            return False, "output is None"
        
        expected_shape = (2, 16, d_model)
        if output.shape != expected_shape:
            return False, f"Shape {output.shape} != {expected_shape}"
        
        return True, f"Full encoder: tokens->embeddings->encoder"
    except Exception as e:
        return False, str(e)


def test_positional_encoding() -> Tuple[bool, str]:
    """Test positional encoding."""
    try:
        d_model = 64
        pos_enc = PositionalEncoding(d_model)
        
        x = torch.randn(2, 16, d_model)
        output = pos_enc(x)
        
        if output.shape != x.shape:
            return False, f"Shape changed: {output.shape}"
        
        # Check that positions are different
        pos_diff = (output[0, 0] - output[0, 1]).abs().sum()
        if pos_diff < 1e-5:
            return False, "Different positions have same encoding"
        
        return True, "Positional encoding adds position info"
    except Exception as e:
        return False, str(e)


def test_pre_vs_post_norm_different() -> Tuple[bool, str]:
    """Test that pre-norm and post-norm produce different outputs."""
    try:
        d_model, num_heads = 64, 4
        
        pre = PreNormEncoderBlock(d_model, num_heads)
        post = PostNormEncoderBlock(d_model, num_heads)
        
        if pre.self_attn is None or post.self_attn is None:
            return False, "Blocks not initialized"
        
        # Copy weights to ensure only architecture differs
        with torch.no_grad():
            post.self_attn.W_q.weight.copy_(pre.self_attn.W_q.weight)
            post.self_attn.W_q.bias.copy_(pre.self_attn.W_q.bias)
            post.self_attn.W_k.weight.copy_(pre.self_attn.W_k.weight)
            post.self_attn.W_k.bias.copy_(pre.self_attn.W_k.bias)
            post.self_attn.W_v.weight.copy_(pre.self_attn.W_v.weight)
            post.self_attn.W_v.bias.copy_(pre.self_attn.W_v.bias)
            post.self_attn.W_o.weight.copy_(pre.self_attn.W_o.weight)
            post.self_attn.W_o.bias.copy_(pre.self_attn.W_o.bias)
        
        x = torch.randn(2, 8, d_model)
        
        pre_out = pre(x)
        post_out = post(x)
        
        if pre_out is None or post_out is None:
            return False, "Output is None"
        
        # They should NOT be identical (different architectures)
        if torch.allclose(pre_out, post_out, atol=1e-5):
            return False, "Pre-norm and post-norm outputs are identical"
        
        return True, "Pre-norm and post-norm produce different results"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("layer_norm_shape", test_layer_norm_shape),
        ("layer_norm_stats", test_layer_norm_stats),
        ("layer_norm_vs_pytorch", test_layer_norm_vs_pytorch),
        ("feedforward_shape", test_feedforward_shape),
        ("feedforward_expansion", test_feedforward_expansion),
        ("post_norm_block", test_post_norm_encoder_block),
        ("pre_norm_block", test_pre_norm_encoder_block),
        ("residual_connection", test_residual_connection),
        ("encoder_stack", test_encoder_stack),
        ("encoder_with_embedding", test_encoder_with_embedding),
        ("positional_encoding", test_positional_encoding),
        ("pre_vs_post_norm", test_pre_vs_post_norm_different),
    ]
    
    print(f"\n{'='*50}\nDay 17: Transformer Encoder Block - Tests\n{'='*50}")
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
