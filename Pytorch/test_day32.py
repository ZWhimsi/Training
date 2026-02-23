"""Test Suite for Day 32: Training and Inference Pipeline"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from typing import Tuple

try:
    from day32 import (
        TrainingConfig, GenerationConfig,
        SimplifiedDeepSeekModel,
        compute_lm_loss, compute_loss_with_mask,
        get_lr_scheduler, visualize_lr_schedule,
        training_step, training_step_with_accumulation,
        sample_greedy, sample_temperature, sample_top_k, sample_top_p,
        generate, Trainer,
        compute_perplexity, count_tokens_per_second, estimate_memory_usage,
        create_dummy_batches
    )
    from torch.optim import AdamW
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


def get_test_model():
    """Create small test model."""
    return SimplifiedDeepSeekModel(
        vocab_size=1000,
        d_model=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=128
    )


def test_lm_loss_basic() -> Tuple[bool, str]:
    """Test basic LM loss computation."""
    try:
        batch, seq_len, vocab_size = 2, 16, 100
        
        logits = torch.randn(batch, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch, seq_len))
        
        loss = compute_lm_loss(logits, labels)
        
        if loss.item() == 0.0:
            return False, "Loss not computed (returned 0)"
        
        if loss.item() < 0:
            return False, "Loss should be positive"
        
        # Random logits should give loss around log(vocab_size)
        expected_random_loss = math.log(vocab_size)
        if abs(loss.item() - expected_random_loss) > 2:
            return False, f"Loss {loss.item():.2f} unexpected for random logits"
        
        return True, f"LM loss: {loss.item():.4f}"
    except Exception as e:
        return False, str(e)


def test_lm_loss_shape() -> Tuple[bool, str]:
    """Test that LM loss shifts correctly."""
    try:
        batch, seq_len, vocab_size = 2, 10, 50
        
        # Create logits where position i strongly predicts token i+1
        logits = torch.zeros(batch, seq_len, vocab_size)
        labels = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        
        # Set correct token probability high
        for i in range(seq_len - 1):
            logits[:, i, (i + 1) % vocab_size] = 10.0  # High logit for next token
        
        loss = compute_lm_loss(logits, labels)
        
        # Loss should be low since we predict correctly
        if loss.item() > 1.0:
            return False, f"Loss too high ({loss.item():.4f}) for correct predictions"
        
        return True, f"Loss for correct predictions: {loss.item():.4f}"
    except Exception as e:
        return False, str(e)


def test_masked_loss() -> Tuple[bool, str]:
    """Test loss with attention mask."""
    try:
        batch, seq_len, vocab_size = 2, 16, 100
        
        logits = torch.randn(batch, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch, seq_len))
        
        # Mask half the sequence
        mask = torch.ones(batch, seq_len)
        mask[:, seq_len//2:] = 0
        
        loss = compute_loss_with_mask(logits, labels, mask)
        
        if loss.item() == 0.0:
            return False, "Masked loss not computed"
        
        return True, f"Masked loss: {loss.item():.4f}"
    except Exception as e:
        return False, str(e)


def test_lr_scheduler_warmup() -> Tuple[bool, str]:
    """Test LR scheduler warmup phase."""
    try:
        model = nn.Linear(10, 10)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=100, max_steps=1000)
        
        # At step 0, LR should be ~0
        lr_start = optimizer.param_groups[0]['lr']
        
        # Simulate 50 steps
        for _ in range(50):
            scheduler.step()
        
        lr_mid = optimizer.param_groups[0]['lr']
        
        # LR should increase during warmup
        if lr_mid <= lr_start + 1e-6:
            return False, "LR not increasing during warmup"
        
        return True, f"Warmup LR: {lr_start:.6f} -> {lr_mid:.6f}"
    except Exception as e:
        return False, str(e)


def test_lr_scheduler_decay() -> Tuple[bool, str]:
    """Test LR scheduler cosine decay."""
    try:
        model = nn.Linear(10, 10)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=10, max_steps=100)
        
        # Go through warmup
        for _ in range(20):
            scheduler.step()
        
        lr_after_warmup = optimizer.param_groups[0]['lr']
        
        # Go to end
        for _ in range(80):
            scheduler.step()
        
        lr_end = optimizer.param_groups[0]['lr']
        
        # LR should decrease after warmup
        if lr_end >= lr_after_warmup:
            return False, "LR not decaying after warmup"
        
        return True, f"Decay: {lr_after_warmup:.6f} -> {lr_end:.6f}"
    except Exception as e:
        return False, str(e)


def test_visualize_lr() -> Tuple[bool, str]:
    """Test LR schedule visualization."""
    try:
        lrs = visualize_lr_schedule(warmup_steps=100, max_steps=1000, base_lr=1e-3)
        
        if len(lrs) != 1000:
            return False, f"Expected 1000 values, got {len(lrs)}"
        
        # Check warmup
        if lrs[50] >= lrs[100]:
            return False, "LR should increase during warmup"
        
        # Check decay
        if lrs[500] <= lrs[999]:
            return False, "LR should decrease after warmup"
        
        return True, f"LR range: {min(lrs):.6f} - {max(lrs):.6f}"
    except Exception as e:
        return False, str(e)


def test_training_step() -> Tuple[bool, str]:
    """Test single training step."""
    try:
        model = get_test_model()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        batch = {'input_ids': torch.randint(0, 1000, (2, 32))}
        
        # Get initial params
        initial_params = [p.clone() for p in model.parameters()]
        
        loss = training_step(model, optimizer, batch)
        
        if loss == 0.0:
            return False, "Training step not implemented"
        
        # Check params changed
        params_changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, model.parameters())
        )
        
        if not params_changed:
            return False, "Parameters didn't change after step"
        
        return True, f"Training loss: {loss:.4f}"
    except Exception as e:
        return False, str(e)


def test_gradient_accumulation() -> Tuple[bool, str]:
    """Test gradient accumulation."""
    try:
        model = get_test_model()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        # Create micro-batches
        batches = [
            {'input_ids': torch.randint(0, 1000, (2, 32))}
            for _ in range(4)
        ]
        
        initial_params = [p.clone() for p in model.parameters()]
        
        loss = training_step_with_accumulation(model, optimizer, batches)
        
        if loss == 0.0:
            return False, "Gradient accumulation not implemented"
        
        params_changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, model.parameters())
        )
        
        if not params_changed:
            return False, "Parameters didn't change"
        
        return True, f"Accumulated loss: {loss:.4f}"
    except Exception as e:
        return False, str(e)


def test_sample_greedy() -> Tuple[bool, str]:
    """Test greedy sampling."""
    try:
        batch, vocab = 3, 100
        logits = torch.randn(batch, vocab)
        
        # Set specific max indices
        logits[0, 42] = 100.0
        logits[1, 7] = 100.0
        logits[2, 99] = 100.0
        
        samples = sample_greedy(logits)
        
        expected = torch.tensor([42, 7, 99])
        if not torch.equal(samples, expected):
            return False, f"Expected {expected.tolist()}, got {samples.tolist()}"
        
        return True, "Greedy sampling correct"
    except Exception as e:
        return False, str(e)


def test_sample_temperature() -> Tuple[bool, str]:
    """Test temperature sampling."""
    try:
        batch, vocab = 2, 100
        logits = torch.zeros(batch, vocab)
        logits[:, 0] = 10.0  # Strong preference for token 0
        
        # Low temperature should be nearly deterministic
        samples_low = [sample_temperature(logits, 0.1) for _ in range(10)]
        if not all(s[0] == 0 for s in samples_low):
            return False, "Low temp should sample token 0"
        
        # High temperature should be more random
        samples_high = [sample_temperature(logits, 10.0) for _ in range(20)]
        unique_samples = len(set(s[0].item() for s in samples_high))
        
        if unique_samples < 2:
            return False, "High temp should produce variety"
        
        return True, f"Temperature sampling works ({unique_samples} unique tokens)"
    except Exception as e:
        return False, str(e)


def test_sample_top_k() -> Tuple[bool, str]:
    """Test top-k sampling."""
    try:
        batch, vocab = 2, 100
        logits = torch.randn(batch, vocab)
        
        # Make top 5 tokens have high logits
        top_indices = [10, 20, 30, 40, 50]
        for idx in top_indices:
            logits[:, idx] = 100.0
        
        # Sample with k=5
        samples = [sample_top_k(logits, k=5) for _ in range(20)]
        
        # All samples should be from top 5
        for s in samples:
            if s[0].item() not in top_indices:
                return False, f"Sampled {s[0].item()} not in top-5"
        
        return True, "Top-K sampling restricts to top tokens"
    except Exception as e:
        return False, str(e)


def test_sample_top_p() -> Tuple[bool, str]:
    """Test top-p (nucleus) sampling."""
    try:
        batch, vocab = 2, 100
        
        # Create logits where first 3 tokens have 90% probability
        logits = torch.ones(batch, vocab) * -10
        logits[:, 0] = 5.0  # ~50%
        logits[:, 1] = 4.0  # ~30%
        logits[:, 2] = 3.0  # ~10%
        
        # With p=0.9, should mostly sample from tokens 0, 1, 2
        samples = [sample_top_p(logits, p=0.9) for _ in range(30)]
        
        in_nucleus = sum(1 for s in samples if s[0].item() < 3)
        
        if in_nucleus < 25:  # Should be mostly from nucleus
            return False, f"Only {in_nucleus}/30 samples from nucleus"
        
        return True, f"Top-P: {in_nucleus}/30 from nucleus"
    except Exception as e:
        return False, str(e)


def test_generate_basic() -> Tuple[bool, str]:
    """Test basic generation."""
    try:
        model = get_test_model()
        config = GenerationConfig(max_new_tokens=5, do_sample=False)
        
        prompt = torch.randint(0, 1000, (1, 3))
        
        output = generate(model, prompt, config)
        
        expected_len = prompt.size(1) + config.max_new_tokens
        if output.size(1) != expected_len:
            return False, f"Output length {output.size(1)} != {expected_len}"
        
        # Prompt should be preserved
        if not torch.equal(output[:, :3], prompt):
            return False, "Prompt not preserved in output"
        
        return True, f"Generated {output.size(1)} tokens"
    except Exception as e:
        return False, str(e)


def test_generate_with_sampling() -> Tuple[bool, str]:
    """Test generation with sampling."""
    try:
        model = get_test_model()
        config = GenerationConfig(
            max_new_tokens=10,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        prompt = torch.randint(0, 1000, (2, 5))  # Batch of 2
        
        output = generate(model, prompt, config)
        
        if output.size(0) != 2:
            return False, "Batch size not preserved"
        
        expected_len = prompt.size(1) + config.max_new_tokens
        if output.size(1) != expected_len:
            return False, f"Length {output.size(1)} != {expected_len}"
        
        return True, f"Generated batch of {output.shape}"
    except Exception as e:
        return False, str(e)


def test_trainer_init() -> Tuple[bool, str]:
    """Test Trainer initialization."""
    try:
        model = get_test_model()
        config = TrainingConfig(learning_rate=1e-4, warmup_steps=10, max_steps=100)
        
        trainer = Trainer(model, config)
        
        if trainer.optimizer is None:
            return False, "Optimizer not initialized"
        if trainer.scheduler is None:
            return False, "Scheduler not initialized"
        
        return True, "Trainer initialized"
    except Exception as e:
        return False, str(e)


def test_trainer_step() -> Tuple[bool, str]:
    """Test Trainer train_step."""
    try:
        model = get_test_model()
        config = TrainingConfig(learning_rate=1e-4, warmup_steps=10, max_steps=100)
        trainer = Trainer(model, config)
        
        if trainer.optimizer is None:
            return False, "Trainer not initialized"
        
        batch = {'input_ids': torch.randint(0, 1000, (2, 32))}
        
        loss = trainer.train_step(batch)
        
        if loss == 0.0:
            return False, "Train step returned 0"
        
        if trainer.global_step != 1:
            return False, f"Step counter wrong: {trainer.global_step}"
        
        return True, f"Train step loss: {loss:.4f}"
    except Exception as e:
        return False, str(e)


def test_trainer_evaluate() -> Tuple[bool, str]:
    """Test Trainer evaluation."""
    try:
        model = get_test_model()
        config = TrainingConfig()
        trainer = Trainer(model, config)
        
        if trainer.optimizer is None:
            return False, "Trainer not initialized"
        
        eval_batches = create_dummy_batches(5, batch_size=2, seq_len=32, vocab_size=1000)
        
        eval_loss = trainer.evaluate(eval_batches)
        
        if eval_loss == 0.0:
            return False, "Evaluate returned 0"
        
        return True, f"Eval loss: {eval_loss:.4f}"
    except Exception as e:
        return False, str(e)


def test_perplexity() -> Tuple[bool, str]:
    """Test perplexity computation."""
    try:
        # Perplexity of loss=1 should be e^1 ≈ 2.718
        ppl = compute_perplexity(1.0)
        
        if ppl == 0.0:
            return False, "Perplexity not computed"
        
        expected = math.exp(1.0)
        if abs(ppl - expected) > 0.01:
            return False, f"Perplexity {ppl:.3f} != {expected:.3f}"
        
        return True, f"Perplexity(loss=1): {ppl:.3f}"
    except Exception as e:
        return False, str(e)


def test_memory_estimation() -> Tuple[bool, str]:
    """Test memory estimation."""
    try:
        model = get_test_model()
        
        mem = estimate_memory_usage(model, batch_size=8, seq_len=512)
        
        if mem['total_mb'] == 0.0:
            return False, "Memory not estimated"
        
        if mem['parameters_mb'] > mem['total_mb']:
            return False, "Parameters larger than total?"
        
        return True, f"Estimated {mem['total_mb']:.1f} MB total"
    except Exception as e:
        return False, str(e)


def test_throughput() -> Tuple[bool, str]:
    """Test throughput measurement."""
    try:
        model = get_test_model()
        
        tps = count_tokens_per_second(model, batch_size=4, seq_len=64, num_runs=3)
        
        if tps == 0.0:
            return False, "Throughput not measured"
        
        return True, f"Throughput: {tps:.0f} tokens/sec"
    except Exception as e:
        return False, str(e)


def test_full_training_cycle() -> Tuple[bool, str]:
    """Test complete training cycle."""
    try:
        model = get_test_model()
        config = TrainingConfig(
            learning_rate=1e-4,
            warmup_steps=5,
            max_steps=20,
            log_interval=5
        )
        trainer = Trainer(model, config)
        
        if trainer.optimizer is None:
            return False, "Trainer not initialized"
        
        train_batches = create_dummy_batches(10, batch_size=2, seq_len=32, vocab_size=1000)
        
        history = trainer.train(train_batches, num_epochs=1)
        
        if len(history['train_loss']) == 0:
            return False, "No training history recorded"
        
        return True, f"Trained {trainer.global_step} steps"
    except Exception as e:
        return False, str(e)


def run_all_tests():
    tests = [
        ("lm_loss_basic", test_lm_loss_basic),
        ("lm_loss_shape", test_lm_loss_shape),
        ("masked_loss", test_masked_loss),
        ("lr_scheduler_warmup", test_lr_scheduler_warmup),
        ("lr_scheduler_decay", test_lr_scheduler_decay),
        ("visualize_lr", test_visualize_lr),
        ("training_step", test_training_step),
        ("gradient_accumulation", test_gradient_accumulation),
        ("sample_greedy", test_sample_greedy),
        ("sample_temperature", test_sample_temperature),
        ("sample_top_k", test_sample_top_k),
        ("sample_top_p", test_sample_top_p),
        ("generate_basic", test_generate_basic),
        ("generate_with_sampling", test_generate_with_sampling),
        ("trainer_init", test_trainer_init),
        ("trainer_step", test_trainer_step),
        ("trainer_evaluate", test_trainer_evaluate),
        ("perplexity", test_perplexity),
        ("memory_estimation", test_memory_estimation),
        ("throughput", test_throughput),
        ("full_training_cycle", test_full_training_cycle),
    ]
    
    print(f"\n{'='*65}")
    print("Day 32: Training and Inference Pipeline - Tests")
    print(f"{'='*65}")
    
    passed = 0
    for name, fn in tests:
        p, m = fn()
        passed += p
        status = "PASS" if p else "FAIL"
        print(f"  [{status}] {name}: {m}")
    
    print(f"\n{'='*65}")
    print(f"Summary: {passed}/{len(tests)} tests passed")
    print(f"{'='*65}")
    
    if passed == len(tests):
        print("\n🎉 Congratulations! You've completed the PyTorch track!")
        print("   You've built DeepSeek Math with MLA from scratch!")
    
    return passed == len(tests)


if __name__ == "__main__":
    if not IMPORT_SUCCESS:
        print(f"Import error: {IMPORT_ERROR}")
        sys.exit(1)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
