"""
Day 32: Training and Inference Pipeline for DeepSeek Math
=========================================================
Estimated time: 1.5-2 hours
Prerequisites: Day 31 (complete model), Day 23 (KV cache)

Learning objectives:
- Implement the training loop with proper loss computation
- Build gradient accumulation for memory-efficient training
- Create the autoregressive generation pipeline
- Implement various sampling strategies (greedy, temperature, top-k, top-p)
- Build a complete training/eval cycle
- Understand learning rate scheduling for transformers

Key Concepts:
-------------
Training Loop:
    1. Forward pass: input_ids -> logits
    2. Compute cross-entropy loss (shifted by 1 position)
    3. Backward pass: compute gradients
    4. Optimizer step with gradient clipping
    5. Learning rate scheduling (warmup + cosine decay)

Loss Computation:
    For language modeling, we predict the next token:
    - Input:  [BOS, tok1, tok2, tok3]
    - Target: [tok1, tok2, tok3, EOS]
    
    Loss = CrossEntropy(logits[:-1], input_ids[1:])

Gradient Accumulation:
    To simulate larger batch sizes with limited memory:
    1. Run forward/backward for micro_batch_size
    2. Accumulate gradients (don't zero)
    3. After N steps, divide gradients by N and update

Autoregressive Generation:
    1. Prefill: Process prompt, get KV cache
    2. Decode: Generate one token at a time
       - Get logits for last position
       - Sample next token
       - Append to cache and continue

Sampling Strategies:
    - Greedy: argmax(logits)
    - Temperature: softmax(logits / temp)
    - Top-K: Sample from top K tokens
    - Top-P (nucleus): Sample from smallest set with cumulative prob >= p
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
from typing import Optional, List, Tuple, Callable, Dict
from dataclasses import dataclass
import time


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Batch and sequence
    batch_size: int = 8
    micro_batch_size: int = 2      # For gradient accumulation
    max_seq_len: int = 512
    
    # Training schedule
    warmup_steps: int = 100
    max_steps: int = 10000
    
    # Regularization
    gradient_clip: float = 1.0
    dropout: float = 0.0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 100


@dataclass 
class GenerationConfig:
    """Configuration for text generation."""
    
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = 0


# ============================================================================
# Simplified Model Components (from Day 31)
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SimplifiedDeepSeekModel(nn.Module):
    """Simplified model for demonstrating training/inference pipelines."""
    
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, 
                 num_heads: int, max_seq_len: int = 512):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'),
            diagonal=1
        )
        
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits


# ============================================================================
# Exercise 1: Language Model Loss
# ============================================================================

def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor,
                    ignore_index: int = -100) -> torch.Tensor:
    """
    Compute language modeling loss (next token prediction).
    
    For autoregressive LM, we predict position i+1 from position i:
    - logits[..., :-1, :] predicts labels[..., 1:]
    
    Args:
        logits: Model output (batch, seq_len, vocab_size)
        labels: Target token IDs (batch, seq_len)
        ignore_index: Index to ignore in loss computation
    
    Returns:
        Scalar loss value
    """
    # API hints:
    # - logits[..., :-1, :].contiguous() -> shift logits
    # - labels[..., 1:].contiguous() -> shift labels
    # - F.cross_entropy(logits.view(-1, vocab), labels.view(-1), ignore_index)
    
    return None


def compute_loss_with_mask(logits: torch.Tensor, labels: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute loss with explicit attention mask.
    
    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)
        attention_mask: (batch, seq_len) - 1 for valid, 0 for padding
    
    Returns:
        Average loss over valid tokens only
    """
    # API hints:
    # - shift_logits = logits[..., :-1, :], shift_labels = labels[..., 1:]
    # - shift_mask = attention_mask[..., 1:].contiguous()
    # - F.cross_entropy(..., reduction='none') -> per-token loss
    # - loss_per_token.view(shift_labels.shape) -> reshape
    # - (loss_per_token * shift_mask).sum() / shift_mask.sum() -> masked avg
    
    return None


# ============================================================================
# Exercise 2: Learning Rate Scheduler
# ============================================================================

def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int,
                     min_lr_ratio: float = 0.1) -> LambdaLR:
    """
    Create learning rate scheduler with warmup and cosine decay.
    
    Schedule:
    1. Linear warmup from 0 to lr over warmup_steps
    2. Cosine decay from lr to min_lr over remaining steps
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        min_lr_ratio: Minimum LR as fraction of initial LR
    
    Returns:
        LambdaLR scheduler
    """
    # API hints:
    # - if step < warmup_steps: return step / warmup_steps -> linear warmup
    # - progress = (step - warmup_steps) / (max_steps - warmup_steps)
    # - cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    # - return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    # - LambdaLR(optimizer, lr_lambda) -> create scheduler
    
    return None


def visualize_lr_schedule(warmup_steps: int, max_steps: int, 
                          base_lr: float = 3e-4) -> List[float]:
    """
    Generate LR values for visualization.
    
    """
    # API hints:
    # - dummy = torch.nn.Linear(1, 1) -> create dummy layer
    # - optimizer = AdamW([dummy.weight], lr=base_lr)
    # - scheduler = get_lr_scheduler(optimizer, warmup_steps, max_steps)
    # - lrs.append(optimizer.param_groups[0]['lr']); scheduler.step()
    
    return None


# ============================================================================
# Exercise 3: Training Step with Gradient Accumulation
# ============================================================================

def training_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                  batch: Dict[str, torch.Tensor], 
                  gradient_clip: float = 1.0) -> float:
    """
    Single training step.
    
    Args:
        model: The model to train
        optimizer: Optimizer
        batch: Dictionary with 'input_ids' and optionally 'attention_mask'
        gradient_clip: Maximum gradient norm
    
    Returns:
        Loss value as float
    """
    # API hints:
    # - model.train() -> set to training mode
    # - logits = model(input_ids) -> forward pass
    # - compute_lm_loss(logits, input_ids) or compute_loss_with_mask(...)
    # - loss.backward() -> backward pass
    # - torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    # - optimizer.step(); optimizer.zero_grad() -> update weights
    # - loss.item() -> return scalar
    
    return None


def training_step_with_accumulation(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
    batches: List[Dict[str, torch.Tensor]],
    gradient_clip: float = 1.0
) -> float:
    """
    Training with gradient accumulation over multiple micro-batches.
    
    Args:
        model: The model
        optimizer: Optimizer
        batches: List of micro-batches
        gradient_clip: Maximum gradient norm
    
    Returns:
        Average loss across micro-batches
    """
    # API hints:
    # - model.train() -> training mode
    # - for batch in batches: forward + loss + (loss / num_batches).backward()
    # - torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    # - optimizer.step(); optimizer.zero_grad() -> single update
    # - total_loss / num_batches -> average loss
    
    return None


# ============================================================================
# Exercise 4: Sampling Strategies
# ============================================================================

def sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    """
    Greedy sampling - select highest probability token.
    
    Args:
        logits: (batch, vocab_size)
    
    Returns:
        Token indices (batch,)
    
    This is a reference implementation - not an exercise.
    """
    return logits.argmax(dim=-1)


def sample_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Sample with temperature scaling.
    
    Args:
        logits: (batch, vocab_size)
        temperature: Temperature (>1 = more random, <1 = more focused)
    
    Returns:
        Sampled token indices (batch,)
    
    """
    # API hints:
    # - if temperature <= 0: return sample_greedy(logits)
    # - F.softmax(logits / temperature, dim=-1) -> scaled probabilities
    # - torch.multinomial(probs, num_samples=1).squeeze(-1) -> sample
    
    return None


def sample_top_k(logits: torch.Tensor, k: int, 
                 temperature: float = 1.0) -> torch.Tensor:
    """
    Top-K sampling - sample from top K tokens only.
    
    Args:
        logits: (batch, vocab_size)
        k: Number of top tokens to consider
        temperature: Temperature scaling
    
    Returns:
        Sampled token indices (batch,)
    
    """
    # API hints:
    # - if k <= 0: return sample_temperature(logits, temperature)
    # - torch.topk(logits, k, dim=-1) -> (top_k_logits, top_k_indices)
    # - F.softmax(top_k_logits / temperature, dim=-1) -> probs
    # - torch.multinomial(probs, num_samples=1).squeeze(-1) -> sampled_idx
    # - top_k_indices[batch_indices, sampled_idx] -> map back to vocab
    
    return None


def sample_top_p(logits: torch.Tensor, p: float,
                 temperature: float = 1.0) -> torch.Tensor:
    """
    Top-P (nucleus) sampling - sample from smallest set with cumulative prob >= p.
    
    Args:
        logits: (batch, vocab_size)
        p: Cumulative probability threshold
        temperature: Temperature scaling
    
    Returns:
        Sampled token indices (batch,)
    
    """
    # API hints:
    # - if p >= 1.0: return sample_temperature(logits, temperature)
    # - logits / temperature -> apply temperature
    # - torch.sort(logits, descending=True, dim=-1) -> sorted_logits, sorted_indices
    # - torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) -> cumulative probs
    # - sorted_mask = cumulative_probs <= p -> find cutoff
    # - sorted_logits[~sorted_mask] = float('-inf') -> mask beyond threshold
    # - torch.multinomial(F.softmax(sorted_logits), 1) -> sample
    # - sorted_indices[batch_indices, sampled_idx] -> map back
    
    return None


# ============================================================================
# Exercise 5: Autoregressive Generation
# ============================================================================

@torch.no_grad()
def generate(model: nn.Module, 
             prompt_ids: torch.Tensor,
             config: GenerationConfig,
             device: torch.device = None) -> torch.Tensor:
    """
    Autoregressive text generation.
    
    Args:
        model: Language model
        prompt_ids: Initial token IDs (batch, prompt_len)
        config: Generation configuration
        device: Device to run on
    
    Returns:
        Generated token IDs (batch, prompt_len + new_tokens)
    """
    # API hints:
    # - model.eval() -> eval mode
    # - device = next(model.parameters()).device
    # - logits = model(generated); next_logits = logits[:, -1, :]
    # - sample_greedy/sample_temperature/sample_top_k/sample_top_p based on config
    # - torch.cat([generated, next_token.unsqueeze(1)], dim=1) -> append
    # - if (next_token == eos_token_id).all(): break -> check EOS
    
    return None


# ============================================================================
# Exercise 6: Complete Training Loop
# ============================================================================

class Trainer:
    """
    Complete trainer for DeepSeek Math.
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig, device: str = 'cpu'):
        """
        Args:
            model: Model to train
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # TODO: Initialize optimizer and scheduler
        # API hints:
        # - AdamW(model.parameters(), lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
        # - get_lr_scheduler(optimizer, warmup_steps, max_steps)
        
        self.optimizer = None
        self.scheduler = None
        
        self.global_step = 0
        self.total_loss = 0.0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.
        """
        # API hints:
        # - self.model.train() -> training mode
        # - logits = self.model(input_ids.to(self.device))
        # - loss = compute_lm_loss(logits, input_ids)
        # - self.optimizer.zero_grad(); loss.backward()
        # - torch.nn.utils.clip_grad_norm_(params, gradient_clip)
        # - self.optimizer.step(); self.scheduler.step()
        # - self.global_step += 1; return loss.item()
        
        return None
    
    def evaluate(self, eval_batches: List[Dict[str, torch.Tensor]]) -> float:
        """
        Evaluate model on validation data.
        """
        # API hints:
        # - self.model.eval() -> eval mode
        # - with torch.no_grad(): -> no gradient computation
        # - for batch: logits = model(input_ids); loss = compute_lm_loss(...)
        # - total_loss / len(eval_batches) -> average loss
        
        return None
    
    def train(self, train_batches: List[Dict[str, torch.Tensor]],
              eval_batches: Optional[List[Dict[str, torch.Tensor]]] = None,
              num_epochs: int = 1) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        """
        # API hints:
        # - history = {'train_loss': [], 'eval_loss': [], 'learning_rate': []}
        # - for epoch in range(num_epochs): for batch: loss = self.train_step(batch)
        # - if step % log_interval == 0: log and append to history
        # - if eval_batches and step % eval_interval == 0: self.evaluate(...)
        # - self.optimizer.param_groups[0]['lr'] -> current LR
        
        return None


# ============================================================================
# Exercise 7: Metrics and Utilities
# ============================================================================

def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Perplexity = exp(loss)
    """
    # API hints:
    # - math.exp(loss) -> perplexity
    
    return None


def count_tokens_per_second(model: nn.Module, batch_size: int, seq_len: int,
                            device: str = 'cpu', num_runs: int = 10) -> float:
    """
    Benchmark model throughput.
    
    """
    # API hints:
    # - model.eval(); model.to(device)
    # - input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    # - Warmup: for _ in range(3): model(input_ids)
    # - torch.cuda.synchronize() if device == 'cuda'
    # - start = time.time(); for _ in range(num_runs): model(input_ids)
    # - total_tokens = batch_size * seq_len * num_runs
    # - return total_tokens / elapsed
    
    return None


def estimate_memory_usage(model: nn.Module, batch_size: int, seq_len: int,
                          dtype_bytes: int = 4) -> Dict[str, float]:
    """
    Estimate memory usage for training.
    
    """
    # API hints:
    # - param_memory = sum(p.numel() * dtype_bytes for p in model.parameters())
    # - grad_memory = param_memory (same size)
    # - optimizer_memory = param_memory * 2 (Adam: momentum + variance)
    # - d_model = model.d_model; num_layers = len(model.layers)
    # - activation_memory = batch_size * seq_len * d_model * num_layers * dtype_bytes
    # - total / (1024**2) -> convert to MB
    
    return None


# ============================================================================
# Demo and Testing
# ============================================================================

def create_dummy_batches(num_batches: int, batch_size: int, seq_len: int,
                         vocab_size: int) -> List[Dict[str, torch.Tensor]]:
    """Create dummy batches for testing."""
    return [
        {'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len))}
        for _ in range(num_batches)
    ]


if __name__ == "__main__":
    print("Day 32: Training and Inference Pipeline for DeepSeek Math")
    print("=" * 65)
    
    # Create test model
    vocab_size = 1000
    d_model = 128
    num_layers = 2
    num_heads = 4
    
    model = SimplifiedDeepSeekModel(vocab_size, d_model, num_layers, num_heads)
    print(f"\nTest model created:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_model: {d_model}")
    print(f"  num_layers: {num_layers}")
    
    # Test loss computation
    print("\nTesting LM Loss:")
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    loss = compute_lm_loss(logits, input_ids)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Perplexity: {compute_perplexity(loss.item()):.2f}")
    
    # Test LR scheduler
    print("\nTesting LR Scheduler:")
    lrs = visualize_lr_schedule(warmup_steps=100, max_steps=1000)
    print(f"  Warmup end LR: {lrs[100]:.6f}")
    print(f"  Final LR: {lrs[-1]:.6f}")
    
    # Test sampling
    print("\nTesting Sampling Strategies:")
    test_logits = torch.randn(2, vocab_size)
    
    greedy = sample_greedy(test_logits)
    print(f"  Greedy: {greedy}")
    
    temp = sample_temperature(test_logits, 0.7)
    print(f"  Temperature (0.7): {temp}")
    
    topk = sample_top_k(test_logits, k=10)
    print(f"  Top-K (10): {topk}")
    
    topp = sample_top_p(test_logits, p=0.9)
    print(f"  Top-P (0.9): {topp}")
    
    # Test generation
    print("\nTesting Generation:")
    gen_config = GenerationConfig(max_new_tokens=10, temperature=0.8, top_k=50)
    prompt = torch.randint(0, vocab_size, (1, 5))
    print(f"  Prompt: {prompt[0].tolist()}")
    
    generated = generate(model, prompt, gen_config)
    print(f"  Generated: {generated[0].tolist()}")
    
    # Test training
    print("\nTesting Training Loop:")
    train_config = TrainingConfig(
        learning_rate=1e-4,
        warmup_steps=10,
        max_steps=50,
        log_interval=10
    )
    
    trainer = Trainer(model, train_config)
    if trainer.optimizer is not None:
        train_batches = create_dummy_batches(20, batch_size=4, seq_len=32, vocab_size=vocab_size)
        history = trainer.train(train_batches, num_epochs=1)
        print(f"  Training completed, final step: {trainer.global_step}")
    
    # Memory estimate
    print("\nMemory Estimation:")
    mem = estimate_memory_usage(model, batch_size=8, seq_len=512)
    for k, v in mem.items():
        print(f"  {k}: {v:.2f} MB")
    
    print("\nRun test_day32.py to verify your implementations!")
