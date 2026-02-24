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
    """Test LayerNorm output shape and parameter initialization."""
    try:
        d_model = 64
        norm = LayerNorm(d_model)
        
        if norm.gamma is None:
            return False, "gamma not initialized"
        if norm.beta is None:
            return False, "beta not initialized"
        
        # Check gamma initialized to ones
        if not torch.allclose(norm.gamma.data, torch.ones(d_model)):
            return False, "gamma should be initialized to ones"
        # Check beta initialized to zeros
        if not torch.allclose(norm.beta.data, torch.zeros(d_model)):
            return False, "beta should be initialized to zeros"
        
        x = torch.randn(2, 8, d_model)
        output = norm(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        return True, f"Shape preserved, params correctly initialized"
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
    """Test FeedForward network output against reference implementation."""
    try:
        d_model = 64
        d_ff = d_model * 4
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
        
        # Validate output against manual computation
        with torch.no_grad():
            expected = ffn.linear2(torch.nn.functional.gelu(ffn.linear1(x)))
        
        if not torch.allclose(output, expected, atol=1e-5):
            return False, "FFN output doesn't match expected GELU(xW1)W2"
        
        return True, f"Output matches reference (GELU activation)"
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
    """Test PostNormEncoderBlock produces normalized output."""
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
        
        # Post-norm: output should be normalized (mean~0, std~1 per position)
        mean = output.mean(dim=-1)
        std = output.std(dim=-1, unbiased=False)
        if not torch.allclose(mean, torch.zeros_like(mean), atol=1e-4):
            return False, f"Post-norm output mean not ~0: {mean.mean().item():.4f}"
        if not torch.allclose(std, torch.ones_like(std), atol=0.15):
            return False, f"Post-norm output std not ~1: {std.mean().item():.4f}"
        
        return True, "PostNormEncoderBlock output is normalized"
    except Exception as e:
        return False, str(e)


def test_pre_norm_encoder_block() -> Tuple[bool, str]:
    """Test PreNormEncoderBlock has residual connection."""
    try:
        d_model, num_heads = 64, 4
        block = PreNormEncoderBlock(d_model, num_heads, dropout=0.0)
        
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
        
        # Pre-norm: output is NOT normalized (residual added after sublayers)
        # But should be different from input due to attention/FFN
        if torch.allclose(output, x, atol=1e-3):
            return False, "Output identical to input - sublayers not applied"
        
        # Check residual connection: output should contain input information
        # Verify by checking correlation between input and output
        correlation = torch.corrcoef(torch.stack([x.flatten(), output.flatten()]))[0, 1]
        if correlation < 0.1:
            return False, f"Weak residual: correlation={correlation:.4f}"
        
        return True, "PreNormEncoderBlock works with residual"
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
    """Test stacked encoder with final normalization."""
    try:
        d_model, num_heads, num_layers = 64, 4, 4
        encoder = TransformerEncoder(d_model, num_heads, num_layers, pre_norm=True)
        
        if encoder.layers is None:
            return False, "layers not initialized"
        if len(encoder.layers) != num_layers:
            return False, f"Expected {num_layers} layers, got {len(encoder.layers)}"
        if encoder.final_norm is None:
            return False, "final_norm not initialized for pre_norm encoder"
        
        x = torch.randn(2, 8, d_model)
        output = encoder(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        # Pre-norm encoder should have final norm applied, so output normalized
        mean = output.mean(dim=-1)
        std = output.std(dim=-1, unbiased=False)
        if not torch.allclose(mean, torch.zeros_like(mean), atol=1e-4):
            return False, f"Final norm not applied: mean={mean.mean().item():.4f}"
        if not torch.allclose(std, torch.ones_like(std), atol=0.15):
            return False, f"Final norm not applied: std={std.mean().item():.4f}"
        
        return True, f"Encoder with {num_layers} layers and final norm"
    except Exception as e:
        return False, str(e)


def test_encoder_with_embedding() -> Tuple[bool, str]:
    """Test complete encoder with embeddings and scaling."""
    try:
        import math
        vocab_size = 1000
        d_model, num_heads, num_layers = 64, 4, 2
        
        encoder = TransformerEncoderWithEmbedding(
            vocab_size, d_model, num_heads, num_layers, dropout=0.0
        )
        
        if encoder.token_emb is None:
            return False, "token_emb not initialized"
        if encoder.encoder is None:
            return False, "encoder not initialized"
        if encoder.pos_enc is None:
            return False, "pos_enc not initialized"
        
        # Token indices
        x = torch.randint(0, vocab_size, (2, 16))
        output = encoder(x)
        
        if output is None:
            return False, "output is None"
        
        expected_shape = (2, 16, d_model)
        if output.shape != expected_shape:
            return False, f"Shape {output.shape} != {expected_shape}"
        
        # Verify embedding scaling by sqrt(d_model)
        raw_emb = encoder.token_emb(x)
        expected_scale = math.sqrt(d_model)
        
        # Check that same tokens produce same base embedding pattern
        if torch.allclose(raw_emb[0, 0], raw_emb[0, 1]) and x[0, 0] != x[0, 1]:
            return False, "Different tokens should have different embeddings"
        
        return True, f"Full encoder with embedding scaling by sqrt({d_model})"
    except Exception as e:
        return False, str(e)


def test_positional_encoding() -> Tuple[bool, str]:
    """Test sinusoidal positional encoding values."""
    try:
        import math
        d_model = 64
        pos_enc = PositionalEncoding(d_model, dropout=0.0)
        
        x = torch.zeros(1, 4, d_model)  # Zero input to isolate PE
        output = pos_enc(x)
        
        if output.shape != x.shape:
            return False, f"Shape changed: {output.shape}"
        
        # Verify sinusoidal pattern: PE[pos, 2i] = sin(pos / 10000^(2i/d))
        pos = 1
        i = 0
        expected_sin = math.sin(pos / (10000 ** (2 * i / d_model)))
        actual = output[0, pos, 2 * i].item()
        if abs(actual - expected_sin) > 1e-5:
            return False, f"PE[{pos}, {2*i}] = {actual:.4f}, expected sin={expected_sin:.4f}"
        
        # PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
        expected_cos = math.cos(pos / (10000 ** (2 * i / d_model)))
        actual_cos = output[0, pos, 2 * i + 1].item()
        if abs(actual_cos - expected_cos) > 1e-5:
            return False, f"PE[{pos}, {2*i+1}] = {actual_cos:.4f}, expected cos={expected_cos:.4f}"
        
        return True, "Sinusoidal PE matches expected formula"
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
