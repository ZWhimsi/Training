"""
Day 20: Efficient Transpose
===========================
Estimated time: 1-2 hours
Prerequisites: Day 19 (block operations)

Learning objectives:
- Implement efficient matrix transpose
- Use shared memory for coalesced access
- Understand bank conflicts
- Handle non-square matrices
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Efficient Transpose
# ============================================================================
# Naive transpose has coalesced reads but scattered writes (or vice versa).
# Tiled transpose: read tile, transpose in shared memory, write tile.
# This makes both reads and writes coalesced!
# ============================================================================


# ============================================================================
# Exercise 1: Naive Transpose
# ============================================================================

@triton.jit
def naive_transpose_kernel(
    input_ptr, output_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Naive transpose: output[j,i] = input[i,j]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # TODO: Load input block [BLOCK_M, BLOCK_N]
    input_offs = offs_m[:, None] * N + offs_n[None, :]
    block = tl.load(input_ptr + input_offs, mask=mask, other=0.0)
    
    # TODO: Transpose the block
    # HINT: block_t = tl.trans(block)  # Or manually swap indices
    block_t = None  # Replace
    
    # TODO: Store transposed block
    # Output shape is (N, M), so output[j, i] at position j * M + i
    output_offs = offs_n[:, None] * M + offs_m[None, :]
    mask_t = mask_n[:, None] & mask_m[None, :]
    
    # HINT: tl.store(output_ptr + output_offs, block_t, mask=mask_t)
    pass  # Replace


def naive_transpose(x: torch.Tensor) -> torch.Tensor:
    """Naive transpose implementation."""
    M, N = x.shape
    output = torch.empty((N, M), device=x.device, dtype=x.dtype)
    
    BLOCK_M = 32
    BLOCK_N = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    naive_transpose_kernel[grid](
        x, output, M, N, BLOCK_M, BLOCK_N
    )
    return output


# ============================================================================
# Exercise 2: Coalesced Transpose with Shared Memory
# ============================================================================

@triton.jit
def tiled_transpose_kernel(
    input_ptr, output_ptr,
    M, N,
    BLOCK: tl.constexpr,
):
    """
    Tiled transpose using shared memory for coalesced access.
    
    1. Load tile coalesced (row by row)
    2. Store to shared memory
    3. Read transposed from shared memory
    4. Write coalesced
    
    Note: Triton handles shared memory automatically through register blocking.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Load tile (coalesced reads)
    input_offs = offs_m[:, None] * N + offs_n[None, :]
    tile = tl.load(input_ptr + input_offs, mask=mask, other=0.0)
    
    # Transpose tile
    tile_t = tl.trans(tile)
    
    # Store transposed tile (coalesced writes)
    # Output position: (offs_n, offs_m) in the (N, M) output
    output_offs = offs_n[:, None] * M + offs_m[None, :]
    mask_t = mask_n[:, None] & mask_m[None, :]
    
    tl.store(output_ptr + output_offs, tile_t, mask=mask_t)


def tiled_transpose(x: torch.Tensor) -> torch.Tensor:
    """Tiled transpose for better memory coalescing."""
    M, N = x.shape
    output = torch.empty((N, M), device=x.device, dtype=x.dtype)
    
    BLOCK = 32
    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    
    tiled_transpose_kernel[grid](x, output, M, N, BLOCK=BLOCK)
    return output


# ============================================================================
# Exercise 3: In-Place Transpose (Square Only)
# ============================================================================

@triton.jit
def inplace_transpose_kernel(
    data_ptr,
    N,  # Matrix is N x N
    BLOCK: tl.constexpr,
):
    """
    In-place transpose for square matrix.
    Only process upper triangular blocks to avoid double-swapping.
    """
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Only process upper triangular (i < j)
    if pid_i >= pid_j:
        return
    
    offs_i = pid_i * BLOCK + tl.arange(0, BLOCK)
    offs_j = pid_j * BLOCK + tl.arange(0, BLOCK)
    
    mask_i = offs_i < N
    mask_j = offs_j < N
    
    # Load block (i, j)
    offs_ij = offs_i[:, None] * N + offs_j[None, :]
    mask_ij = mask_i[:, None] & mask_j[None, :]
    block_ij = tl.load(data_ptr + offs_ij, mask=mask_ij, other=0.0)
    
    # Load block (j, i)
    offs_ji = offs_j[:, None] * N + offs_i[None, :]
    mask_ji = mask_j[:, None] & mask_i[None, :]
    block_ji = tl.load(data_ptr + offs_ji, mask=mask_ji, other=0.0)
    
    # Swap: store transposed blocks
    tl.store(data_ptr + offs_ij, tl.trans(block_ji), mask=mask_ij)
    tl.store(data_ptr + offs_ji, tl.trans(block_ij), mask=mask_ji)


def inplace_transpose(x: torch.Tensor) -> torch.Tensor:
    """In-place transpose (modifies input, returns same tensor)."""
    assert x.shape[0] == x.shape[1], "In-place transpose requires square matrix"
    N = x.shape[0]
    
    BLOCK = 32
    n_blocks = triton.cdiv(N, BLOCK)
    grid = (n_blocks, n_blocks)
    
    inplace_transpose_kernel[grid](x, N, BLOCK=BLOCK)
    return x


# ============================================================================
# Exercise 4: Batched Transpose
# ============================================================================

@triton.jit
def batched_transpose_kernel(
    input_ptr, output_ptr,
    B, M, N,
    BLOCK: tl.constexpr,
):
    """
    Transpose each matrix in a batch.
    Input: (B, M, N) -> Output: (B, N, M)
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # Input offset for batch pid_b
    batch_offset_in = pid_b * M * N
    input_offs = batch_offset_in + offs_m[:, None] * N + offs_n[None, :]
    
    tile = tl.load(input_ptr + input_offs, mask=mask, other=0.0)
    tile_t = tl.trans(tile)
    
    # Output offset for batch pid_b (shape N x M)
    batch_offset_out = pid_b * N * M
    output_offs = batch_offset_out + offs_n[:, None] * M + offs_m[None, :]
    mask_t = mask_n[:, None] & mask_m[None, :]
    
    tl.store(output_ptr + output_offs, tile_t, mask=mask_t)


def batched_transpose(x: torch.Tensor) -> torch.Tensor:
    """Transpose each matrix in batch: (B, M, N) -> (B, N, M)."""
    B, M, N = x.shape
    output = torch.empty((B, N, M), device=x.device, dtype=x.dtype)
    
    BLOCK = 32
    grid = (B, triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    
    batched_transpose_kernel[grid](x, output, B, M, N, BLOCK=BLOCK)
    return output


if __name__ == "__main__":
    print("Day 20: Efficient Transpose")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        x = torch.randn(256, 512, device=device)
        
        print("\nTesting naive transpose:")
        result = naive_transpose(x)
        expected = x.T
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        print("\nTesting tiled transpose:")
        result = tiled_transpose(x)
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
        
        print("\nTesting batched transpose:")
        x_batch = torch.randn(8, 64, 128, device=device)
        result = batched_transpose(x_batch)
        expected = x_batch.transpose(1, 2)
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day20.py to verify!")
