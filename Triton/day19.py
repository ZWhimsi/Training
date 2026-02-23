"""
Day 19: Block Matrix Operations
===============================
Estimated time: 1-2 hours
Prerequisites: Day 18 (strided access)

Learning objectives:
- Work with blocked/tiled matrix representations
- Implement block-level operations
- Understand blocking for cache efficiency
- Build foundation for tiled algorithms
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: Block Matrices
# ============================================================================
# Large matrices are often processed in blocks for cache efficiency.
# A (M, N) matrix can be viewed as (M//BM, N//BN) blocks of size (BM, BN).
#
# Block (i, j) contains elements [i*BM:(i+1)*BM, j*BN:(j+1)*BN]
# ============================================================================


# ============================================================================
# Exercise 1: Extract Block
# ============================================================================

@triton.jit
def extract_block_kernel(
    input_ptr, output_ptr,
    M, N,  # Full matrix size
    block_i, block_j,  # Which block to extract
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Extract a single block from a matrix.
    """
    # Block starts at (block_i * BLOCK_M, block_j * BLOCK_N)
    row_start = block_i * BLOCK_M
    col_start = block_j * BLOCK_N
    
    # Local offsets within the block
    local_m = tl.arange(0, BLOCK_M)
    local_n = tl.arange(0, BLOCK_N)
    
    # Global offsets
    global_m = row_start + local_m
    global_n = col_start + local_n
    
    mask_m = global_m < M
    mask_n = global_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # TODO: Compute input offsets (row-major)
    # HINT: input_offs = global_m[:, None] * N + global_n[None, :]
    input_offs = None  # Replace
    
    # TODO: Load block
    block = None  # Replace: tl.load(input_ptr + input_offs, mask=mask, other=0.0)
    
    # Store to output (as contiguous block)
    output_offs = local_m[:, None] * BLOCK_N + local_n[None, :]
    tl.store(output_ptr + output_offs, block, mask=mask)


def extract_block(x: torch.Tensor, block_i: int, block_j: int, 
                  block_m: int, block_n: int) -> torch.Tensor:
    """Extract block (block_i, block_j) from matrix."""
    M, N = x.shape
    output = torch.zeros((block_m, block_n), device=x.device, dtype=x.dtype)
    
    extract_block_kernel[(1,)](
        x, output, M, N, block_i, block_j, 
        BLOCK_M=block_m, BLOCK_N=block_n
    )
    return output


# ============================================================================
# Exercise 2: Block Addition
# ============================================================================

@triton.jit
def block_add_kernel(
    a_ptr, b_ptr, output_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Add two matrices block by block.
    Each program handles one block.
    """
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    row_start = block_i * BLOCK_M
    col_start = block_j * BLOCK_N
    
    local_m = tl.arange(0, BLOCK_M)
    local_n = tl.arange(0, BLOCK_N)
    
    global_m = row_start + local_m
    global_n = col_start + local_n
    
    mask_m = global_m < M
    mask_n = global_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    offs = global_m[:, None] * N + global_n[None, :]
    
    # TODO: Load blocks from a and b
    a_block = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b_block = tl.load(b_ptr + offs, mask=mask, other=0.0)
    
    # TODO: Add blocks
    result = None  # Replace: a_block + b_block
    
    # TODO: Store result
    # HINT: tl.store(output_ptr + offs, result, mask=mask)
    pass  # Replace


def block_matrix_add(a: torch.Tensor, b: torch.Tensor, 
                     block_m: int = 32, block_n: int = 32) -> torch.Tensor:
    """Add matrices using block decomposition."""
    M, N = a.shape
    output = torch.empty_like(a)
    
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    block_add_kernel[grid](
        a, b, output, M, N, BLOCK_M=block_m, BLOCK_N=block_n
    )
    return output


# ============================================================================
# Exercise 3: Block Diagonal
# ============================================================================

@triton.jit
def block_diagonal_kernel(
    input_ptr, output_ptr,
    M, N, n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Extract diagonal blocks from a matrix.
    Assumes square blocks along the diagonal.
    """
    block_idx = tl.program_id(0)
    
    # Block (block_idx, block_idx) - diagonal block
    start = block_idx * BLOCK_SIZE
    
    local_i = tl.arange(0, BLOCK_SIZE)
    local_j = tl.arange(0, BLOCK_SIZE)
    
    global_i = start + local_i
    global_j = start + local_j
    
    mask_i = global_i < M
    mask_j = global_j < N
    mask = mask_i[:, None] & mask_j[None, :]
    
    # TODO: Load diagonal block
    input_offs = global_i[:, None] * N + global_j[None, :]
    block = tl.load(input_ptr + input_offs, mask=mask, other=0.0)
    
    # Store to output position
    output_offs = block_idx * BLOCK_SIZE * BLOCK_SIZE + local_i[:, None] * BLOCK_SIZE + local_j[None, :]
    tl.store(output_ptr + output_offs, block, mask=mask)


def extract_block_diagonal(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Extract diagonal blocks as a (n_blocks, block_size, block_size) tensor."""
    M, N = x.shape
    n_blocks = min(M, N) // block_size
    output = torch.zeros((n_blocks, block_size, block_size), device=x.device, dtype=x.dtype)
    
    block_diagonal_kernel[(n_blocks,)](
        x, output, M, N, n_blocks, BLOCK_SIZE=block_size
    )
    return output


# ============================================================================
# Exercise 4: Block Trace
# ============================================================================

@triton.jit
def block_trace_kernel(
    input_ptr, output_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute trace of each diagonal block.
    """
    block_idx = tl.program_id(0)
    
    start = block_idx * BLOCK_SIZE
    
    # Diagonal elements within block
    diag_offs = tl.arange(0, BLOCK_SIZE)
    global_idx = start + diag_offs
    
    mask = global_idx < min(M, N)
    
    # TODO: Load diagonal elements
    # For element (i, i): offset is i * N + i = i * (N + 1)
    input_offs = global_idx * N + global_idx
    diag_elements = tl.load(input_ptr + input_offs, mask=mask, other=0.0)
    
    # TODO: Sum diagonal elements (trace)
    trace = tl.sum(diag_elements, axis=0)
    
    # Store trace for this block
    tl.store(output_ptr + block_idx, trace)


def block_traces(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Compute trace of each diagonal block."""
    M, N = x.shape
    n_blocks = min(M, N) // block_size
    output = torch.zeros(n_blocks, device=x.device, dtype=x.dtype)
    
    block_trace_kernel[(n_blocks,)](
        x, output, M, N, BLOCK_SIZE=block_size
    )
    return output


if __name__ == "__main__":
    print("Day 19: Block Matrix Operations")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        M, N = 128, 128
        x = torch.randn(M, N, device=device)
        
        print("\nTesting block extraction:")
        block = extract_block(x, 1, 2, 32, 32)
        expected = x[32:64, 64:96]
        print(f"  Max error: {(block - expected).abs().max().item():.6f}")
        
        print("\nTesting block addition:")
        a = torch.randn(M, N, device=device)
        b = torch.randn(M, N, device=device)
        result = block_matrix_add(a, b)
        expected = a + b
        print(f"  Max error: {(result - expected).abs().max().item():.6f}")
    else:
        print("CUDA not available")
    
    print("\nRun test_day19.py to verify!")
