"""Test Suite for Day 20: GPT-style Decoder-Only Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple

try:
    from day20 import (RMSNorm, GPTBlock, GPT, compute_lm_loss,
                       prepare_lm_batch, get_gpt2_config)
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def test_rms_norm_shape() -> Tuple[bool, str]:
    """Test RMSNorm output shape."""
    try:
        d_model = 64
        norm = RMSNorm(d_model)
        
        if norm.weight is None:
            return False, "weight not initialized"
        
        x = torch.randn(2, 8, d_model)
        output = norm(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        return True, f"RMSNorm shape: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_rms_norm_normalization() -> Tuple[bool, str]:
    """Test that RMSNorm actually normalizes."""
    try:
        d_model = 64
        norm = RMSNorm(d_model)
        
        if norm.weight is None:
            return False, "weight not initialized"
        
        x = torch.randn(2, 8, d_model) * 10  # Large values
        output = norm(x)
        
        if output is None:
            return False, "output is None"
        
        # RMS of output should be close to 1 (with default weight=1)
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        
        # Should be approximately 1
        if not torch.allclose(rms, torch.ones_like(rms), atol=0.2):
            return False, f"RMS not ~1: {rms.mean().item():.4f}"
        
        return True, f"RMS after norm: {rms.mean().item():.4f}"
    except Exception as e:
        return False, str(e)


def test_gpt_block_shape() -> Tuple[bool, str]:
    """Test GPTBlock output shape."""
    try:
        d_model, num_heads = 64, 4
        batch, seq = 2, 16
        
        block = GPTBlock(d_model, num_heads)
        
        if block.W_q is None:
            return False, "W_q not initialized"
        
        x = torch.randn(batch, seq, d_model)
        output, _ = block(x)
        
        if output is None:
            return False, "output is None"
        if output.shape != x.shape:
            return False, f"Shape {output.shape} != {x.shape}"
        
        return True, f"GPTBlock output: {output.shape}"
    except Exception as e:
        return False, str(e)


def test_gpt_block_causal() -> Tuple[bool, str]:
    """Test that GPTBlock respects causal masking."""
    try:
        d_model, num_heads = 64, 4
        batch, seq = 1, 8
        
        block = GPTBlock(d_model, num_heads, dropout=0.0)
        
        if block.W_q is None:
            return False, "Block not initialized"
        
        x = torch.randn(batch, seq, d_model)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq, seq))
        
        output1, _ = block(x, causal_mask)
        
        if output1 is None:
            return False, "output is None"
        
        # Modify future tokens - shouldn't affect past outputs
        x_modified = x.clone()
        x_modified[:, -1, :] = torch.randn(d_model)  # Change last token
        
        output2, _ = block(x_modified, causal_mask)
        
        # First position output should be identical
        if not torch.allclose(output1[:, 0], output2[:, 0], atol=1e-5):
            return False, "Causal masking not working - future affects past"
        
        return True, "Causal masking verified"
    except Exception as e:
        return False, str(e)


def test_gpt_block_kv_cache() -> Tuple[bool, str]:
    """Test GPTBlock KV caching for generation."""
    try:
        d_model, num_heads = 64, 4
        batch = 2
        
        block = GPTBlock(d_model, num_heads, dropout=0.0)
        
        if block.W_q is None:
            return False, "Block not initialized"
        
        # Full sequence forward
        x_full = torch.randn(batch, 5, d_model)
        mask_full = torch.tril(torch.ones(5, 5))
        output_full, cache = block(x_full, mask_full)
        
        if output_full is None:
            return False, "output is None"
        if cache is None:
            return False, "KV cache is None"
        
        # Check cache shapes
        k, v = cache
        if k.shape[1] != 5:
            return False, f"Cache K seq length {k.shape[1]} != 5"
        
        return True, f"KV cache works, K shape: {k.shape}"
    except Exception as e:
        return False, str(e)


def test_gpt_model_shape() -> Tuple[bool, str]:
    """Test GPT model output shape."""
    try:
        vocab_size = 1000
        d_model, num_heads, num_layers = 64, 4, 2
        
        model = GPT(vocab_size, d_model, num_heads, num_layers, max_len=128)
        
        if model.token_embedding is None:
            return False, "token_embedding not initialized"
        if model.blocks is None:
            return False, "blocks not initialized"
        
        batch, seq = 2, 16
        input_ids = torch.randint(0, vocab_size, (batch, seq))
        
        logits, _ = model(input_ids)
        
        if logits is None:
            return False, "logits is None"
        
        expected_shape = (batch, seq, vocab_size)
        if logits.shape != expected_shape:
            return False, f"Shape {logits.shape} != {expected_shape}"
        
        return True, f"GPT output: {logits.shape}"
    except Exception as e:
        return False, str(e)


def test_gpt_generation() -> Tuple[bool, str]:
    """Test GPT generation."""
    try:
        vocab_size = 100
        model = GPT(vocab_size, d_model=32, num_heads=2, num_layers=2, max_len=50)
        
        if model.token_embedding is None:
            return False, "Model not initialized"
        
        prompt = torch.randint(0, vocab_size, (1, 5))
        max_new_tokens = 10
        
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0)
        
        if generated is None:
            return False, "generate returned None"
        
        # Check length increased
        if generated.shape[1] <= prompt.shape[1]:
            return False, "No new tokens generated"
        
        # Check length is at most prompt + max_new_tokens
        if generated.shape[1] > prompt.shape[1] + max_new_tokens:
            return False, f"Generated too many tokens: {generated.shape[1]}"
        
        return True, f"Generated {generated.shape[1] - prompt.shape[1]} new tokens"
    except Exception as e:
        return False, str(e)


def test_gpt_with_cache() -> Tuple[bool, str]:
    """Test GPT with KV cache for efficient generation."""
    try:
        vocab_size = 100
        model = GPT(vocab_size, d_model=32, num_heads=2, num_layers=2, max_len=50)
        
        if model.token_embedding is None:
            return False, "Model not initialized"
        
        # First forward with cache
        input_ids = torch.randint(0, vocab_size, (1, 5))
        logits1, cache = model(input_ids, use_cache=True)
        
        if logits1 is None:
            return False, "logits is None"
        if cache is None or len(cache) == 0:
            return False, "Cache not returned"
        
        # Continue with just the new token
        new_token = torch.randint(0, vocab_size, (1, 1))
        logits2, _ = model(new_token, use_cache=True, past_kv_cache=cache)
        
        if logits2 is None:
            return False, "Second forward failed"
        if logits2.shape != (1, 1, vocab_size):
            return False, f"Wrong shape with cache: {logits2.shape}"
        
        return True, "KV cache speeds up generation"
    except Exception as e:
        return False, str(e)


def test_compute_lm_loss() -> Tuple[bool, str]:
    """Test language modeling loss computation."""
    try:
        batch, seq, vocab_size = 2, 10, 100
        
        logits = torch.randn(batch, seq, vocab_size)
        targets = torch.randint(0, vocab_size, (batch, seq))
        
        loss = compute_lm_loss(logits, targets)
        
        if loss is None:
            return False, "loss is None"
        
        # Loss should be a scalar
        if loss.dim() != 0:
            return False, f"Loss should be scalar, got dim={loss.dim()}"
        
        # Loss should be positive
        if loss.item() <= 0:
            return False, "Loss should be positive"
        
        # Check roughly reasonable magnitude
        if loss.item() > 100:
            return False, f"Loss suspiciously high: {loss.item()}"
        
        return True, f"LM loss: {loss.item():.4f}"
    except Exception as e:
        return False, str(e)


def test_prepare_lm_batch() -> Tuple[bool, str]:
    """Test LM batch preparation (shift by 1)."""
    try:
        batch, seq = 2, 10
        token_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
        
        input_ids, target_ids = prepare_lm_batch(token_ids)
        
        if input_ids is None:
            return False, "input_ids is None"
        if target_ids is None:
            return False, "target_ids is None"
        
        # Should be one shorter
        if input_ids.shape != (batch, seq - 1):
            return False, f"Input shape {input_ids.shape} != ({batch}, {seq-1})"
        if target_ids.shape != (batch, seq - 1):
            return False, f"Target shape {target_ids.shape} != ({batch}, {seq-1})"
        
        # Input should be [0, 1, ..., seq-2]
        # Target should be [1, 2, ..., seq-1]
        if input_ids[0, 0].item() != 0:
            return False, "Input should start at 0"
        if target_ids[0, 0].item() != 1:
            return False, "Target should start at 1"
        
        return True, f"Input: [0..{seq-2}], Target: [1..{seq-1}]"
    except Exception as e:
        return False, str(e)


def test_gpt_gradient_flow() -> Tuple[bool, str]:
    """Test gradient flow through GPT model."""
    try:
        model = GPT(vocab_size=100, d_model=32, num_heads=2, num_layers=2)
        
        if model.token_embedding is None:
            return False, "Model not initialized"
        
        input_ids = torch.randint(0, 100, (2, 8))
        logits, _ = model(input_ids)
        
        if logits is None:
            return False, "logits is None"
        
        loss = logits.sum()
        loss.backward()
        
        # Check embeddings have gradients
        if model.token_embedding.weight.grad is None:
            return False, "No gradient to token embedding"
        
        return True, "Gradients flow through entire model"
    except Exception as e:
        return False, str(e)


def test_gpt2_configs() -> Tuple[bool, str]:
    """Test GPT-2 configuration function."""
    try:
        for size in ['small', 'medium', 'large', 'xl']:
            config = get_gpt2_config(size)
            
            if 'd_model' not in config:
                return False, f"Missing d_model in {size}"
            if 'num_heads' not in config:
                return False, f"Missing num_heads in {size}"
            if 'num_layers' not in config:
                return False, f"Missing num_layers in {size}"
            
            # Verify d_model divisible by num_heads
            if config['d_model'] % config['num_heads'] != 0:
                return False, f"d_model not divisible by heads in {size}"
        
        return True, "All GPT-2 configs valid"
    except Exception as e:
        return False, str(e)


def test_gpt_with_rms_norm() -> Tuple[bool, str]:
    """Test GPT model with RMSNorm."""
    try:
        model = GPT(
            vocab_size=100, 
            d_model=32, 
            num_heads=2, 
            num_layers=2,
            use_rms_norm=True
        )
        
        if model.token_embedding is None:
            return False, "Model not initialized"
        
        input_ids = torch.randint(0, 100, (1, 8))
        logits, _ = model(input_ids)
        
        if logits is None:
            return False, "logits is None"
        
        return True, "GPT with RMSNorm works"
    except Exception as e:
        return False, str(e)


def test_gpt_top_k_sampling() -> Tuple[bool, str]:
    """Test top-k sampling in generation."""
    try:
        model = GPT(vocab_size=100, d_model=32, num_heads=2, num_layers=2)
        
        if model.token_embedding is None:
            return False, "Model not initialized"
        
        prompt = torch.randint(0, 100, (1, 3))
        
        with torch.no_grad():
            # Generate with different top_k values
            gen_k5 = model.generate(prompt.clone(), max_new_tokens=5, top_k=5)
            gen_k50 = model.generate(prompt.clone(), max_new_tokens=5, top_k=50)
        
        if gen_k5 is None or gen_k50 is None:
            return False, "Generation failed"
        
        return True, "Top-k sampling works"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("rms_norm_shape", test_rms_norm_shape),
        ("rms_norm_normalization", test_rms_norm_normalization),
        ("gpt_block_shape", test_gpt_block_shape),
        ("gpt_block_causal", test_gpt_block_causal),
        ("gpt_block_kv_cache", test_gpt_block_kv_cache),
        ("gpt_model_shape", test_gpt_model_shape),
        ("gpt_generation", test_gpt_generation),
        ("gpt_with_cache", test_gpt_with_cache),
        ("compute_lm_loss", test_compute_lm_loss),
        ("prepare_lm_batch", test_prepare_lm_batch),
        ("gpt_gradient_flow", test_gpt_gradient_flow),
        ("gpt2_configs", test_gpt2_configs),
        ("gpt_with_rms_norm", test_gpt_with_rms_norm),
        ("gpt_top_k_sampling", test_gpt_top_k_sampling),
    ]
    
    print(f"\n{'='*50}\nDay 20: GPT-style Decoder-Only Model - Tests\n{'='*50}")
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
