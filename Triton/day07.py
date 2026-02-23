"""
Day 7: Memory Coalescing and Access Patterns
============================================
Estimated time: 1-2 hours
Prerequisites: Day 6 (multi-dimensional grids)

Learning objectives:
- Understand memory coalescing for performance
- Optimize memory access patterns
- Compare coalesced vs non-coalesced access
- Profile memory bandwidth

Hints:
- Coalesced access: consecutive threads access consecutive memory
- Row-major layout: adjacent columns are adjacent in memory
- Use contiguous() to ensure proper memory layout
- Poor access patterns can 10x slow down your kernel!
"""

import torch
import triton
import triton.language as tl
import time


# ============================================================================
# Exercise 1: Row-wise Access (Coalesced)
# ============================================================================

@triton.jit
def row_access_kernel(
    x_ptr,
    out_ptr,
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    """
    Access matrix row by row (good memory access pattern).
    Each program handles one row.
    """
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return
    
    col_offs = tl.arange(0, BLOCK_N)
    mask = col_offs < N
    
    # TODO: Calculate row pointer (coalesced access)
    row_ptr = None  # Replace: x_ptr + row_idx * stride_m
    
    # TODO: Load row (adjacent elements in memory)
    row = None  # Replace
    
    # Process (e.g., double the values)
    out_row = row * 2.0
    
    # TODO: Store
    out_row_ptr = None  # Replace
    pass


def row_access(x: torch.Tensor) -> torch.Tensor:
    """Process matrix row by row."""
    assert x.is_cuda
    M, N = x.shape
    out = torch.empty_like(x)
    
    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    
    # TODO: Launch kernel
    
    return out


# ============================================================================
# Exercise 2: Column-wise Access (Non-coalesced - Educational)
# ============================================================================

@triton.jit
def col_access_kernel(
    x_ptr,
    out_ptr,
    M, N,
    stride_m,
    BLOCK_M: tl.constexpr,
):
    """
    Access matrix column by column (poor memory access pattern).
    This is for educational purposes - shows what NOT to do for performance.
    """
    col_idx = tl.program_id(0)
    
    if col_idx >= N:
        return
    
    row_offs = tl.arange(0, BLOCK_M)
    mask = row_offs < M
    
    # TODO: Calculate indices (strided access - not coalesced!)
    # Each load jumps by stride_m in memory
    indices = None  # Replace: row_offs * stride_m + col_idx
    
    # TODO: Load column (non-adjacent elements)
    col = None  # Replace
    
    # Process
    out_col = col * 2.0
    
    # TODO: Store (also strided)
    pass


def col_access(x: torch.Tensor) -> torch.Tensor:
    """Process matrix column by column."""
    assert x.is_cuda
    M, N = x.shape
    out = torch.empty_like(x)
    
    BLOCK_M = triton.next_power_of_2(M)
    grid = (N,)
    
    # TODO: Launch kernel
    
    return out


# ============================================================================
# Exercise 3: Tiled Access (Balanced)
# ============================================================================

@triton.jit
def tiled_access_kernel(
    x_ptr,
    out_ptr,
    M, N,
    stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Tiled access pattern - good balance of coalescing and parallelism.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    # TODO: Calculate 2D indices
    indices = None  # Replace: offs_m[:, None] * stride_m + offs_n[None, :]
    
    # TODO: Load tile
    tile = None  # Replace
    
    # Process
    out_tile = tile * 2.0
    
    # TODO: Store tile
    pass


def tiled_access(x: torch.Tensor) -> torch.Tensor:
    """Process matrix with tiled access."""
    assert x.is_cuda
    M, N = x.shape
    out = torch.empty_like(x)
    
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # TODO: Launch kernel
    
    return out


# ============================================================================
# Exercise 4: Vectorized Load
# ============================================================================

@triton.jit
def vectorized_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Use vectorized loads for better memory throughput.
    Triton automatically vectorizes when possible.
    """
    pid = tl.program_id(0)
    
    # Use larger offsets for vectorization
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # TODO: Load (Triton will vectorize automatically if aligned)
    x = None  # Replace
    
    # Process
    out = x * 2.0
    
    # TODO: Store
    pass


def vectorized_op(x: torch.Tensor) -> torch.Tensor:
    """Vectorized operation."""
    assert x.is_cuda
    n = x.numel()
    out = torch.empty_like(x)
    
    # Larger block size for better vectorization
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # TODO: Launch kernel
    
    return out


# ============================================================================
# Exercise 5: Benchmark Different Access Patterns
# ============================================================================

def benchmark_access_patterns():
    """
    Compare performance of different access patterns.
    Run this to see the impact of memory coalescing!
    """
    print("\nBenchmarking Memory Access Patterns")
    print("=" * 50)
    
    # Create test data
    M, N = 1024, 1024
    x = torch.randn(M, N, device='cuda')
    
    # Warmup
    _ = row_access(x)
    _ = col_access(x)
    _ = tiled_access(x)
    torch.cuda.synchronize()
    
    n_iters = 100
    
    # Benchmark row access
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = row_access(x)
    torch.cuda.synchronize()
    row_time = (time.perf_counter() - start) / n_iters * 1000
    
    # Benchmark column access  
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = col_access(x)
    torch.cuda.synchronize()
    col_time = (time.perf_counter() - start) / n_iters * 1000
    
    # Benchmark tiled access
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = tiled_access(x)
    torch.cuda.synchronize()
    tiled_time = (time.perf_counter() - start) / n_iters * 1000
    
    print(f"Row access (coalesced):     {row_time:.3f} ms")
    print(f"Column access (strided):    {col_time:.3f} ms")
    print(f"Tiled access (balanced):    {tiled_time:.3f} ms")
    print(f"\nColumn/Row ratio: {col_time/row_time:.1f}x slower")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 7: Memory Coalescing and Access Patterns")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print("Run test_day07.py to verify implementations")
        print("Then run benchmark_access_patterns() to see performance impact!")
    else:
        print("CUDA not available")
