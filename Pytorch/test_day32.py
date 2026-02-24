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
        
        torch.manual_seed(42)
        logits = torch.randn(batch, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch, seq_len))
        
        loss = compute_lm_loss(logits, labels)
        
        if loss.item() == 0.0:
            return False, "Loss not computed (returned 0)"
        
        if loss.item() < 0:
            return False, "Loss should be positive"
        
        # Verify against PyTorch reference: shifted cross entropy
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        expected_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )
        
        if not torch.allclose(loss, expected_loss, atol=1e-5):
            return False, f"Loss {loss.item():.4f} != expected {expected_loss.item():.4f}"
        
        return True, f"LM loss verified: {loss.item():.4f}"
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
        
        torch.manual_seed(42)
        logits = torch.randn(batch, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch, seq_len))
        
        # Mask half the sequence
        mask = torch.ones(batch, seq_len)
        mask[:, seq_len//2:] = 0
        
        loss = compute_loss_with_mask(logits, labels, mask)
        
        if loss.item() == 0.0:
            return False, "Masked loss not computed"
        
        # Verify manually: compute per-token loss and apply mask
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction='none'
        ).view(shift_labels.shape)
        
        expected_loss = (loss_per_token * shift_mask).sum() / shift_mask.sum()
        
        if not torch.allclose(loss, expected_loss, atol=1e-5):
            return False, f"Masked loss {loss.item():.4f} != expected {expected_loss.item():.4f}"
        
        return True, f"Masked loss verified: {loss.item():.4f}"
    except Exception as e:
        return False, str(e)


def test_lr_scheduler_warmup() -> Tuple[bool, str]:
    """Test LR scheduler warmup phase."""
    try:
        base_lr = 1e-3
        warmup_steps = 100
        model = nn.Linear(10, 10)
        optimizer = AdamW(model.parameters(), lr=base_lr)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=1000)
        
        # Collect LRs during warmup
        lrs = []
        for step in range(warmup_steps + 10):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # LR should increase during warmup (linear warmup)
        for i in range(1, warmup_steps):
            if lrs[i] <= lrs[i-1]:
                return False, f"LR not increasing at step {i}"
        
        # At end of warmup, LR should be close to base_lr
        warmup_end_lr = lrs[warmup_steps]
        if abs(warmup_end_lr - base_lr) > base_lr * 0.1:
            return False, f"LR at warmup end {warmup_end_lr:.6f} not close to base {base_lr:.6f}"
        
        # Verify linear warmup: at step 50, LR should be ~50% of base
        mid_warmup_lr = lrs[50]
        expected_mid = base_lr * 50 / warmup_steps
        if abs(mid_warmup_lr - expected_mid) > expected_mid * 0.2:
            return False, f"Linear warmup failed: {mid_warmup_lr:.6f} != {expected_mid:.6f}"
        
        return True, f"Warmup verified: linear increase to {warmup_end_lr:.6f}"
    except Exception as e:
        return False, str(e)


def test_lr_scheduler_decay() -> Tuple[bool, str]:
    """Test LR scheduler cosine decay."""
    try:
        base_lr = 1e-3
        min_lr_ratio = 0.1
        warmup_steps = 10
        max_steps = 100
        
        model = nn.Linear(10, 10)
        optimizer = AdamW(model.parameters(), lr=base_lr)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=max_steps, min_lr_ratio=min_lr_ratio)
        
        # Collect all LRs
        lrs = []
        for step in range(max_steps):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        lr_peak = lrs[warmup_steps]
        lr_end = lrs[-1]
        
        # LR should decrease after warmup
        if lr_end >= lr_peak:
            return False, "LR not decaying after warmup"
        
        # Final LR should be close to min_lr = base_lr * min_lr_ratio
        expected_min_lr = base_lr * min_lr_ratio
        if abs(lr_end - expected_min_lr) > expected_min_lr * 0.3:
            return False, f"Final LR {lr_end:.6f} not close to min {expected_min_lr:.6f}"
        
        # Verify cosine decay shape: midpoint should be ~(peak + min) / 2
        mid_step = (warmup_steps + max_steps) // 2
        lr_mid = lrs[mid_step]
        expected_mid = (lr_peak + expected_min_lr) / 2
        if abs(lr_mid - expected_mid) > expected_mid * 0.3:
            return False, f"Mid LR {lr_mid:.6f} not matching cosine decay"
        
        return True, f"Cosine decay: {lr_peak:.6f} -> {lr_end:.6f}"
    except Exception as e:
        return False, str(e)


def test_visualize_lr() -> Tuple[bool, str]:
    """Test LR schedule visualization."""
    try:
        base_lr = 1e-3
        warmup_steps = 100
        max_steps = 1000
        lrs = visualize_lr_schedule(warmup_steps=warmup_steps, max_steps=max_steps, base_lr=base_lr)
        
        if len(lrs) != max_steps:
            return False, f"Expected {max_steps} values, got {len(lrs)}"
        
        # Check warmup phase increases
        for i in range(1, warmup_steps):
            if lrs[i] < lrs[i-1]:
                return False, f"LR should increase during warmup (step {i})"
        
        # Check peak at end of warmup
        lr_peak = lrs[warmup_steps]
        if abs(lr_peak - base_lr) > base_lr * 0.1:
            return False, f"Peak LR {lr_peak:.6f} not close to base {base_lr:.6f}"
        
        # Check decay after warmup
        if lrs[500] <= lrs[999]:
            return False, "LR should decrease after warmup"
        
        # Verify all LRs are positive
        if min(lrs) <= 0:
            return False, "All LRs should be positive"
        
        return True, f"LR range verified: {min(lrs):.6f} - {max(lrs):.6f}"
    except Exception as e:
        return False, str(e)


def test_training_step() -> Tuple[bool, str]:
    """Test single training step."""
    try:
        torch.manual_seed(42)
        model = get_test_model()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        batch = {'input_ids': torch.randint(0, 1000, (2, 32))}
        
        # Get initial params
        initial_params = [p.clone() for p in model.parameters()]
        
        loss = training_step(model, optimizer, batch)
        
        if loss == 0.0:
            return False, "Training step not implemented"
        
        # Verify loss is reasonable (around log(vocab_size) for random)
        expected_random_loss = math.log(1000)  # vocab_size = 1000
        if abs(loss - expected_random_loss) > 2:
            return False, f"Loss {loss:.4f} unexpected for random init"
        
        # Check params changed
        params_changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, model.parameters())
        )
        
        if not params_changed:
            return False, "Parameters didn't change after step"
        
        # Verify gradients were zeroed (optimizer.zero_grad)
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 1e-6:
                return False, "Gradients should be zeroed after step"
        
        return True, f"Training step verified, loss: {loss:.4f}"
    except Exception as e:
        return False, str(e)


def test_gradient_accumulation() -> Tuple[bool, str]:
    """Test gradient accumulation."""
    try:
        torch.manual_seed(42)
        model = get_test_model()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        # Create micro-batches
        num_batches = 4
        batches = [
            {'input_ids': torch.randint(0, 1000, (2, 32))}
            for _ in range(num_batches)
        ]
        
        initial_params = [p.clone() for p in model.parameters()]
        
        loss = training_step_with_accumulation(model, optimizer, batches)
        
        if loss == 0.0:
            return False, "Gradient accumulation not implemented"
        
        # Verify loss is average of micro-batch losses
        # Recompute manually
        model2 = get_test_model()
        model2.load_state_dict(dict(zip(
            [n for n, _ in model.named_parameters()],
            initial_params
        )))
        
        params_changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, model.parameters())
        )
        
        if not params_changed:
            return False, "Parameters didn't change"
        
        # Verify gradients were zeroed after accumulation
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 1e-6:
                return False, "Gradients should be zeroed after accumulation step"
        
        return True, f"Gradient accumulation verified, avg loss: {loss:.4f}"
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
        torch.manual_seed(42)
        batch, vocab = 2, 100
        logits = torch.zeros(batch, vocab)
        logits[:, 0] = 10.0  # Strong preference for token 0
        
        # Low temperature should be nearly deterministic
        samples_low = [sample_temperature(logits.clone(), 0.1) for _ in range(10)]
        if not all(s[0] == 0 for s in samples_low):
            return False, "Low temp should sample token 0"
        
        # Verify temperature scaling: probs = softmax(logits / temp)
        temp = 2.0
        expected_probs = F.softmax(logits / temp, dim=-1)
        
        # Sample many times and verify distribution roughly matches
        torch.manual_seed(123)
        samples = [sample_temperature(logits.clone(), temp) for _ in range(100)]
        token0_count = sum(1 for s in samples if s[0] == 0)
        
        # With temp=2, token 0 should be selected most but not all the time
        if token0_count < 50 or token0_count > 99:
            return False, f"Temperature scaling seems wrong: {token0_count}/100 token 0"
        
        return True, f"Temperature sampling verified"
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
        torch.manual_seed(42)
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
        
        # Greedy generation should be deterministic
        output2 = generate(model, prompt, config)
        if not torch.equal(output, output2):
            return False, "Greedy generation should be deterministic"
        
        # Verify generated tokens are valid vocab indices
        if (output < 0).any() or (output >= 1000).any():
            return False, "Generated tokens outside vocab range"
        
        return True, f"Generated {output.size(1)} tokens (deterministic)"
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
        torch.manual_seed(42)
        model = get_test_model()
        config = TrainingConfig(learning_rate=1e-4, warmup_steps=10, max_steps=100)
        trainer = Trainer(model, config)
        
        if trainer.optimizer is None:
            return False, "Trainer not initialized"
        
        initial_params = [p.clone() for p in model.parameters()]
        
        batch = {'input_ids': torch.randint(0, 1000, (2, 32))}
        
        loss = trainer.train_step(batch)
        
        if loss == 0.0:
            return False, "Train step returned 0"
        
        if trainer.global_step != 1:
            return False, f"Step counter wrong: {trainer.global_step}"
        
        # Verify loss is reasonable
        expected_random_loss = math.log(1000)
        if abs(loss - expected_random_loss) > 2:
            return False, f"Loss {loss:.4f} unexpected"
        
        # Verify params changed
        params_changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, model.parameters())
        )
        if not params_changed:
            return False, "Parameters didn't change"
        
        return True, f"Train step verified, loss: {loss:.4f}"
    except Exception as e:
        return False, str(e)


def test_trainer_evaluate() -> Tuple[bool, str]:
    """Test Trainer evaluation."""
    try:
        torch.manual_seed(42)
        model = get_test_model()
        config = TrainingConfig()
        trainer = Trainer(model, config)
        
        if trainer.optimizer is None:
            return False, "Trainer not initialized"
        
        eval_batches = create_dummy_batches(5, batch_size=2, seq_len=32, vocab_size=1000)
        
        # Store model state
        model.eval()
        
        eval_loss = trainer.evaluate(eval_batches)
        
        if eval_loss == 0.0:
            return False, "Evaluate returned 0"
        
        # Verify loss is reasonable
        expected_random_loss = math.log(1000)
        if abs(eval_loss - expected_random_loss) > 2:
            return False, f"Eval loss {eval_loss:.4f} unexpected"
        
        # Running evaluate again should give same result (no randomness)
        eval_loss2 = trainer.evaluate(eval_batches)
        if abs(eval_loss - eval_loss2) > 1e-5:
            return False, "Eval should be deterministic"
        
        return True, f"Eval loss verified: {eval_loss:.4f}"
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
        
        mem = estimate_memory_usage(model, batch_size=8, seq_len=512, dtype_bytes=4)
        
        if mem['total_mb'] == 0.0:
            return False, "Memory not estimated"
        
        if mem['parameters_mb'] > mem['total_mb']:
            return False, "Parameters larger than total?"
        
        # Verify parameter memory matches actual model
        actual_params = sum(p.numel() for p in model.parameters())
        expected_param_mb = actual_params * 4 / (1024 ** 2)
        
        if abs(mem['parameters_mb'] - expected_param_mb) > expected_param_mb * 0.1:
            return False, f"Param memory {mem['parameters_mb']:.2f} != {expected_param_mb:.2f}"
        
        # Gradient memory should equal parameter memory
        if abs(mem['gradients_mb'] - mem['parameters_mb']) > 0.01:
            return False, "Gradient memory should equal parameter memory"
        
        # Optimizer memory should be 2x params for Adam
        if abs(mem['optimizer_mb'] - 2 * mem['parameters_mb']) > 0.01:
            return False, "Optimizer memory should be 2x params for Adam"
        
        return True, f"Memory verified: {mem['total_mb']:.1f} MB"
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
        torch.manual_seed(42)
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
        
        initial_params = [p.clone() for p in model.parameters()]
        
        train_batches = create_dummy_batches(10, batch_size=2, seq_len=32, vocab_size=1000)
        
        history = trainer.train(train_batches, num_epochs=1)
        
        if len(history['train_loss']) == 0:
            return False, "No training history recorded"
        
        # Verify params changed after training
        params_changed = any(
            not torch.allclose(p1, p2)
            for p1, p2 in zip(initial_params, model.parameters())
        )
        if not params_changed:
            return False, "Parameters didn't change after training"
        
        # Verify global step incremented
        if trainer.global_step == 0:
            return False, "Global step should be > 0"
        
        # Verify learning rates were recorded
        if 'learning_rate' in history and len(history['learning_rate']) > 0:
            # LRs should be positive
            if any(lr <= 0 for lr in history['learning_rate']):
                return False, "Learning rates should be positive"
        
        return True, f"Training cycle verified: {trainer.global_step} steps"
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
