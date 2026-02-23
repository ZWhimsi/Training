# 30+ Day ML Training Program

A comprehensive, hands-on training program covering three parallel tracks:
- **Triton**: GPU kernel programming from basics to Flash Attention
- **PyTorch**: Deep learning from tensors to DeepSeek Math with Multi-head Latent Attention
- **Autodiff**: Building an automatic differentiation library from scratch

## Overview

This training follows a LeetCode-style approach: each day provides exercises with TODO sections, hints, and test cases. Start with foundational concepts and progressively build toward advanced implementations.

**Time commitment**: ~1-2 hours per track per day (3-6 hours total daily)

## Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support (for Triton track)
- Basic Python and linear algebra knowledge

### Setup

```bash
# Clone the repository
git clone https://github.com/ZWhimsi/Training.git
cd Training

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Exercises

```bash
# Navigate to a track
cd triton

# Run a day's exercises
python day01.py

# Run tests to verify your solutions
pytest day01.py -v
```

## Curriculum

### Triton Track (35 days)
*From GPU basics to Flash Attention implementation*

| Week | Days | Topics |
|------|------|--------|
| 1 | 1-7 | Foundations: kernels, memory ops, grids, blocks |
| 2 | 8-14 | Basic ops: vectors, reductions, fused softmax |
| 3 | 15-21 | Matrix ops: tiled matmul, cache optimization |
| 4 | 22-28 | Advanced: layer norm, attention mechanisms |
| 5 | 29-35 | Flash Attention: forward and backward pass |

**Final project**: Complete Flash Attention with forward and backward passes

### PyTorch Track (32 days)
*From tensors to DeepSeek Math with MLA*

| Week | Days | Topics |
|------|------|--------|
| 1 | 1-7 | Fundamentals: tensors, autograd, nn.Module |
| 2 | 8-14 | Deep learning: CNNs, RNNs, custom modules |
| 3 | 15-21 | Attention: self-attention, MHA, Transformers |
| 4 | 22-28 | Advanced: GQA, MQA, KV-cache, MLA |
| 5 | 29-32 | DeepSeek Math: full architecture with MLA |

**Final project**: DeepSeek Math implementation with Multi-head Latent Attention

### Autodiff Track (35 days)
*Building a PyTorch-like library from scratch*

| Week | Days | Topics |
|------|------|--------|
| 1 | 1-7 | Foundations: derivatives, chain rule, graphs |
| 2 | 8-14 | Scalar engine: Value class, basic operations |
| 3 | 15-21 | Tensor ops: multi-dim, matmul, broadcasting |
| 4 | 22-28 | Autograd: Function class, graph management |
| 5 | 29-35 | Neural nets: Module system, full training |

**Final project**: Complete autograd library with nn.Module equivalent

## Exercise Format

Each `dayXX.py` file contains:

```python
"""
Day XX: Topic Name
==================
Estimated time: 1-2 hours
Prerequisites: Day XX-1

Learning objectives:
- What you'll learn

Hints:
- Useful functions and tips
"""

# Exercise 1: Description
# TODO: Implement this function
# HINT: Use specific_function() to...
def exercise_1():
    pass

# Tests to verify your solution
def test_exercise_1():
    assert exercise_1() == expected
```

## Progress Tracking

Track your progress by checking off completed days:

- [ ] Triton: Day 1-35
- [ ] PyTorch: Day 1-32
- [ ] Autodiff: Day 1-35

## Resources

### Triton
- [Triton Documentation](https://triton-lang.org/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2 Paper](https://arxiv.org/abs/2307.08691)

### PyTorch
- [PyTorch Documentation](https://pytorch.org/docs/)
- [DeepSeek Math Paper](https://arxiv.org/abs/2402.03300)
- [DeepSeek-V2 Paper (MLA)](https://arxiv.org/abs/2405.04434)

### Autodiff
- [Automatic Differentiation Survey](https://arxiv.org/abs/1502.05767)
- [micrograd by Karpathy](https://github.com/karpathy/micrograd)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

MIT License - feel free to use, modify, and distribute.

## Acknowledgments

- Andrej Karpathy for micrograd inspiration
- Triton team at OpenAI
- DeepSeek AI team for MLA innovation
