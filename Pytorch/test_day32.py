"""Test Suite for Day 32: Training and Inference Pipeline"""

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
import math
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
    IMPORT_ERROR = None
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


def test_lm_loss_basic():
    """Test basic LM loss computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq_len, vocab_size = 2, 16, 100
    
    torch.manual_seed(42)
    logits = torch.randn(batch, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch, seq_len))
    
    loss = compute_lm_loss(logits, labels)
    
    assert loss.item() != 0.0, "Loss not computed (returned 0)"
    
    assert loss.item() >= 0, "Loss should be positive"
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    expected_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1)
    )
    
    assert torch.allclose(loss, expected_loss, atol=1e-5), f"Loss {loss.item():.4f} != expected {expected_loss.item():.4f}"


def test_lm_loss_shape():
    """Test that LM loss shifts correctly."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq_len, vocab_size = 2, 10, 50
    
    logits = torch.zeros(batch, seq_len, vocab_size)
    labels = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
    
    for i in range(seq_len - 1):
        logits[:, i, (i + 1) % vocab_size] = 10.0
    
    loss = compute_lm_loss(logits, labels)
    
    assert loss.item() <= 1.0, f"Loss too high ({loss.item():.4f}) for correct predictions"


def test_masked_loss():
    """Test loss with attention mask."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, seq_len, vocab_size = 2, 16, 100
    
    torch.manual_seed(42)
    logits = torch.randn(batch, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch, seq_len))
    
    mask = torch.ones(batch, seq_len)
    mask[:, seq_len//2:] = 0
    
    loss = compute_loss_with_mask(logits, labels, mask)
    
    assert loss.item() != 0.0, "Masked loss not computed"
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = mask[..., 1:].contiguous()
    
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction='none'
    ).view(shift_labels.shape)
    
    expected_loss = (loss_per_token * shift_mask).sum() / shift_mask.sum()
    
    assert torch.allclose(loss, expected_loss, atol=1e-5), f"Masked loss {loss.item():.4f} != expected {expected_loss.item():.4f}"


def test_lr_scheduler_warmup():
    """Test LR scheduler warmup phase."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    base_lr = 1e-3
    warmup_steps = 100
    model = nn.Linear(10, 10)
    optimizer = AdamW(model.parameters(), lr=base_lr)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=1000)
    
    lrs = []
    for step in range(warmup_steps + 10):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    for i in range(1, warmup_steps):
        assert lrs[i] >= lrs[i-1], f"LR not increasing at step {i}"
    
    warmup_end_lr = lrs[warmup_steps]
    assert abs(warmup_end_lr - base_lr) <= base_lr * 0.1, f"LR at warmup end {warmup_end_lr:.6f} not close to base {base_lr:.6f}"
    
    mid_warmup_lr = lrs[50]
    expected_mid = base_lr * 50 / warmup_steps
    assert abs(mid_warmup_lr - expected_mid) <= expected_mid * 0.2, f"Linear warmup failed: {mid_warmup_lr:.6f} != {expected_mid:.6f}"


def test_lr_scheduler_decay():
    """Test LR scheduler cosine decay."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    base_lr = 1e-3
    min_lr_ratio = 0.1
    warmup_steps = 10
    max_steps = 100
    
    model = nn.Linear(10, 10)
    optimizer = AdamW(model.parameters(), lr=base_lr)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=max_steps, min_lr_ratio=min_lr_ratio)
    
    lrs = []
    for step in range(max_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    
    lr_peak = lrs[warmup_steps]
    lr_end = lrs[-1]
    
    assert lr_end <= lr_peak, "LR not decaying after warmup"
    
    expected_min_lr = base_lr * min_lr_ratio
    assert abs(lr_end - expected_min_lr) <= expected_min_lr * 0.3, f"Final LR {lr_end:.6f} not close to min {expected_min_lr:.6f}"
    
    mid_step = (warmup_steps + max_steps) // 2
    lr_mid = lrs[mid_step]
    expected_mid = (lr_peak + expected_min_lr) / 2
    assert abs(lr_mid - expected_mid) <= expected_mid * 0.3, f"Mid LR {lr_mid:.6f} not matching cosine decay"


def test_visualize_lr():
    """Test LR schedule visualization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    base_lr = 1e-3
    warmup_steps = 100
    max_steps = 1000
    lrs = visualize_lr_schedule(warmup_steps=warmup_steps, max_steps=max_steps, base_lr=base_lr)
    
    assert len(lrs) == max_steps, f"Expected {max_steps} values, got {len(lrs)}"
    
    for i in range(1, warmup_steps):
        assert lrs[i] >= lrs[i-1], f"LR should increase during warmup (step {i})"
    
    lr_peak = lrs[warmup_steps]
    assert abs(lr_peak - base_lr) <= base_lr * 0.1, f"Peak LR {lr_peak:.6f} not close to base {base_lr:.6f}"
    
    assert lrs[500] >= lrs[999], "LR should decrease after warmup"
    
    assert min(lrs) > 0, "All LRs should be positive"


def test_training_step():
    """Test single training step."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = get_test_model()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    batch = {'input_ids': torch.randint(0, 1000, (2, 32))}
    
    initial_params = [p.clone() for p in model.parameters()]
    
    loss = training_step(model, optimizer, batch)
    
    assert loss != 0.0, "Training step not implemented"
    
    expected_random_loss = math.log(1000)
    assert abs(loss - expected_random_loss) <= 2, f"Loss {loss:.4f} unexpected for random init"
    
    params_changed = any(
        not torch.allclose(p1, p2)
        for p1, p2 in zip(initial_params, model.parameters())
    )
    
    assert params_changed, "Parameters didn't change after step"


def test_gradient_accumulation():
    """Test gradient accumulation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = get_test_model()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    num_batches = 4
    batches = [
        {'input_ids': torch.randint(0, 1000, (2, 32))}
        for _ in range(num_batches)
    ]
    
    initial_params = [p.clone() for p in model.parameters()]
    
    loss = training_step_with_accumulation(model, optimizer, batches)
    
    assert loss != 0.0, "Gradient accumulation not implemented"
    
    params_changed = any(
        not torch.allclose(p1, p2)
        for p1, p2 in zip(initial_params, model.parameters())
    )
    
    assert params_changed, "Parameters didn't change"


def test_sample_greedy():
    """Test greedy sampling."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, vocab = 3, 100
    logits = torch.randn(batch, vocab)
    
    logits[0, 42] = 100.0
    logits[1, 7] = 100.0
    logits[2, 99] = 100.0
    
    samples = sample_greedy(logits)
    
    expected = torch.tensor([42, 7, 99])
    assert torch.equal(samples, expected), f"Expected {expected.tolist()}, got {samples.tolist()}"


def test_sample_temperature():
    """Test temperature sampling."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    batch, vocab = 2, 100
    logits = torch.zeros(batch, vocab)
    logits[:, 0] = 10.0
    
    samples_low = [sample_temperature(logits.clone(), 0.1) for _ in range(10)]
    assert all(s[0] == 0 for s in samples_low), "Low temp should sample token 0"
    
    temp = 2.0
    expected_probs = F.softmax(logits / temp, dim=-1)
    
    torch.manual_seed(123)
    samples = [sample_temperature(logits.clone(), temp) for _ in range(100)]
    token0_count = sum(1 for s in samples if s[0] == 0)
    
    assert 50 <= token0_count <= 99, f"Temperature scaling seems wrong: {token0_count}/100 token 0"


def test_sample_top_k():
    """Test top-k sampling."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, vocab = 2, 100
    logits = torch.randn(batch, vocab)
    
    top_indices = [10, 20, 30, 40, 50]
    for idx in top_indices:
        logits[:, idx] = 100.0
    
    samples = [sample_top_k(logits, k=5) for _ in range(20)]
    
    for s in samples:
        assert s[0].item() in top_indices, f"Sampled {s[0].item()} not in top-5"


def test_sample_top_p():
    """Test top-p (nucleus) sampling."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    batch, vocab = 2, 100
    
    logits = torch.ones(batch, vocab) * -10
    logits[:, 0] = 5.0
    logits[:, 1] = 4.0
    logits[:, 2] = 3.0
    
    samples = [sample_top_p(logits, p=0.9) for _ in range(30)]
    
    in_nucleus = sum(1 for s in samples if s[0].item() < 3)
    
    assert in_nucleus >= 25, f"Only {in_nucleus}/30 samples from nucleus"


def test_generate_basic():
    """Test basic generation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = get_test_model()
    config = GenerationConfig(max_new_tokens=5, do_sample=False)
    
    prompt = torch.randint(0, 1000, (1, 3))
    
    output = generate(model, prompt, config)
    
    expected_len = prompt.size(1) + config.max_new_tokens
    assert output.size(1) == expected_len, f"Output length {output.size(1)} != {expected_len}"
    
    assert torch.equal(output[:, :3], prompt), "Prompt not preserved in output"
    
    output2 = generate(model, prompt, config)
    assert torch.equal(output, output2), "Greedy generation should be deterministic"
    
    assert not (output < 0).any() and not (output >= 1000).any(), "Generated tokens outside vocab range"


def test_generate_with_sampling():
    """Test generation with sampling."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = get_test_model()
    config = GenerationConfig(
        max_new_tokens=10,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        do_sample=True
    )
    
    prompt = torch.randint(0, 1000, (2, 5))
    
    output = generate(model, prompt, config)
    
    assert output.size(0) == 2, "Batch size not preserved"
    
    expected_len = prompt.size(1) + config.max_new_tokens
    assert output.size(1) == expected_len, f"Length {output.size(1)} != {expected_len}"


def test_trainer_init():
    """Test Trainer initialization."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = get_test_model()
    config = TrainingConfig(learning_rate=1e-4, warmup_steps=10, max_steps=100)
    
    trainer = Trainer(model, config)
    
    assert trainer.optimizer is not None, "Optimizer not initialized"
    assert trainer.scheduler is not None, "Scheduler not initialized"


def test_trainer_step():
    """Test Trainer train_step."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = get_test_model()
    config = TrainingConfig(learning_rate=1e-4, warmup_steps=10, max_steps=100)
    trainer = Trainer(model, config)
    
    assert trainer.optimizer is not None, "Trainer not initialized"
    
    initial_params = [p.clone() for p in model.parameters()]
    
    batch = {'input_ids': torch.randint(0, 1000, (2, 32))}
    
    loss = trainer.train_step(batch)
    
    assert loss != 0.0, "Train step returned 0"
    
    assert trainer.global_step == 1, f"Step counter wrong: {trainer.global_step}"
    
    expected_random_loss = math.log(1000)
    assert abs(loss - expected_random_loss) <= 2, f"Loss {loss:.4f} unexpected"
    
    params_changed = any(
        not torch.allclose(p1, p2)
        for p1, p2 in zip(initial_params, model.parameters())
    )
    assert params_changed, "Parameters didn't change"


def test_trainer_evaluate():
    """Test Trainer evaluation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = get_test_model()
    config = TrainingConfig()
    trainer = Trainer(model, config)
    
    assert trainer.optimizer is not None, "Trainer not initialized"
    
    eval_batches = create_dummy_batches(5, batch_size=2, seq_len=32, vocab_size=1000)
    
    model.eval()
    
    eval_loss = trainer.evaluate(eval_batches)
    
    assert eval_loss != 0.0, "Evaluate returned 0"
    
    expected_random_loss = math.log(1000)
    assert abs(eval_loss - expected_random_loss) <= 2, f"Eval loss {eval_loss:.4f} unexpected"
    
    eval_loss2 = trainer.evaluate(eval_batches)
    assert abs(eval_loss - eval_loss2) <= 1e-5, "Eval should be deterministic"


def test_perplexity():
    """Test perplexity computation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    ppl = compute_perplexity(1.0)
    
    assert ppl != 0.0, "Perplexity not computed"
    
    expected = math.exp(1.0)
    assert abs(ppl - expected) <= 0.01, f"Perplexity {ppl:.3f} != {expected:.3f}"


def test_memory_estimation():
    """Test memory estimation."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = get_test_model()
    
    mem = estimate_memory_usage(model, batch_size=8, seq_len=512, dtype_bytes=4)
    
    assert mem['total_mb'] != 0.0, "Memory not estimated"
    
    assert mem['parameters_mb'] <= mem['total_mb'], "Parameters larger than total?"
    
    actual_params = sum(p.numel() for p in model.parameters())
    expected_param_mb = actual_params * 4 / (1024 ** 2)
    
    assert abs(mem['parameters_mb'] - expected_param_mb) <= expected_param_mb * 0.1, f"Param memory {mem['parameters_mb']:.2f} != {expected_param_mb:.2f}"
    
    assert abs(mem['gradients_mb'] - mem['parameters_mb']) <= 0.01, "Gradient memory should equal parameter memory"
    
    assert abs(mem['optimizer_mb'] - 2 * mem['parameters_mb']) <= 0.01, "Optimizer memory should be 2x params for Adam"


def test_throughput():
    """Test throughput measurement."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    model = get_test_model()
    
    tps = count_tokens_per_second(model, batch_size=4, seq_len=64, num_runs=3)
    
    assert tps != 0.0, "Throughput not measured"


def test_full_training_cycle():
    """Test complete training cycle."""
    if not IMPORT_SUCCESS:
        pytest.fail(f"Import error: {IMPORT_ERROR}")
    
    torch.manual_seed(42)
    model = get_test_model()
    config = TrainingConfig(
        learning_rate=1e-4,
        warmup_steps=5,
        max_steps=20,
        log_interval=5
    )
    trainer = Trainer(model, config)
    
    assert trainer.optimizer is not None, "Trainer not initialized"
    
    initial_params = [p.clone() for p in model.parameters()]
    
    train_batches = create_dummy_batches(10, batch_size=2, seq_len=32, vocab_size=1000)
    
    history = trainer.train(train_batches, num_epochs=1)
    
    assert len(history['train_loss']) != 0, "No training history recorded"
    
    params_changed = any(
        not torch.allclose(p1, p2)
        for p1, p2 in zip(initial_params, model.parameters())
    )
    assert params_changed, "Parameters didn't change after training"
    
    assert trainer.global_step != 0, "Global step should be > 0"
    
    if 'learning_rate' in history and len(history['learning_rate']) > 0:
        assert all(lr > 0 for lr in history['learning_rate']), "Learning rates should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
