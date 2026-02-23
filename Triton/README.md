# Triton Track: GPU Kernel Programming

Master GPU programming with Triton, from basic kernels to implementing Flash Attention.

## Overview

Triton is a language and compiler for writing efficient GPU code. This track takes you from writing your first kernel to implementing the complete Flash Attention algorithm with both forward and backward passes.

## Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- Basic understanding of parallel computing concepts
- Linear algebra (matrix multiplication, dot products)

## Curriculum

### Phase 1: Foundations (Days 1-7)

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 1 | Hello Triton | `@triton.jit`, basic kernel structure |
| 2 | Memory Operations | `tl.load()`, `tl.store()`, pointer arithmetic |
| 3 | Masking | Bounds checking, mask creation |
| 4 | Program IDs | `tl.program_id()`, grid launch |
| 5 | Block Programming | Block-level parallelism |
| 6 | Multi-dimensional Grids | 2D/3D program organization |
| 7 | Memory Coalescing | Efficient memory access patterns |

### Phase 2: Basic Operations (Days 8-14)

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 8 | Vector Addition | Element-wise operations |
| 9 | Vector Operations | Broadcasting, vectorized loads |
| 10 | Reduction Basics | `tl.sum()`, `tl.max()` |
| 11 | Row-wise Reductions | 2D reduction patterns |
| 12 | Numerical Stability | Max-subtraction trick |
| 13 | Fused Softmax | Kernel fusion, online algorithms |
| 14 | Softmax Optimization | Memory bandwidth optimization |

### Phase 3: Matrix Operations (Days 15-21)

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 15 | Matrix Basics | 2D memory access, strides |
| 16 | Blocked Matmul | `tl.dot()`, tiling |
| 17 | K-dimension Loop | Accumulation patterns |
| 18 | Cache Optimization | L2 cache awareness |
| 19 | Block Size Tuning | Performance profiling |
| 20 | Split-K Algorithm | Alternative parallelization |
| 21 | Advanced Matmul | Grouped GEMM concepts |

### Phase 4: Advanced Kernels (Days 22-28)

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 22 | Layer Normalization | Mean, variance computation |
| 23 | Fused LayerNorm | Two-pass statistics |
| 24 | Dropout | Random number generation |
| 25 | Low-memory Dropout | Memory-efficient patterns |
| 26 | Extern Functions | `tl.extra.libdevice` |
| 27 | Standard Attention | QKV attention basics |
| 28 | Attention Memory | Memory bottleneck analysis |

### Phase 5: Flash Attention (Days 29-35)

| Day | Topic | Key Concepts |
|-----|-------|--------------|
| 29 | Flash Attention Theory | I/O-aware algorithm design |
| 30 | Tiling Strategy | Q, K, V block decomposition |
| 31 | Online Softmax | Incremental normalization |
| 32 | Flash Forward Pass | Complete forward implementation |
| 33 | Backward Theory | Gradient computation strategy |
| 34 | Flash Backward Pass | dQ, dK, dV computation |
| 35 | Complete Flash Attention | Integration and optimization |

## Key Triton Functions

```python
# Kernel decorator
@triton.jit
def kernel(x_ptr, ...):
    pass

# Memory operations
tl.load(ptr, mask=None)      # Load from memory
tl.store(ptr, value, mask=None)  # Store to memory

# Indexing
tl.arange(0, BLOCK_SIZE)     # Generate indices
tl.program_id(axis)          # Get block ID

# Reductions
tl.sum(x, axis=0)            # Sum reduction
tl.max(x, axis=0)            # Max reduction

# Math operations
tl.dot(a, b)                 # Block matrix multiply
tl.exp(x)                    # Exponential
tl.log(x)                    # Logarithm

# Control flow
tl.where(cond, x, y)         # Conditional select
```

## Resources

- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub](https://github.com/triton-lang/triton)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)
- [GPU Memory Hierarchy](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)

## Tips for Success

1. **Understand memory**: GPU performance is often memory-bound
2. **Think in blocks**: Operations happen at the block level
3. **Profile early**: Use Triton's profiling tools
4. **Start simple**: Get correctness before optimization
5. **Compare with PyTorch**: Verify results against reference implementations
