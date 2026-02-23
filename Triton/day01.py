"""
Day 1: Hello Triton - Your First GPU Kernel
============================================
Estimated time: 1-2 hours
Prerequisites: None (first day!)

Learning objectives:
- Understand what Triton is and why it's useful
- Write and run your first Triton kernel
- Understand the @triton.jit decorator
- Learn basic kernel launch configuration

Hints:
- Use @triton.jit to decorate kernel functions
- Kernels run on GPU, regular Python runs on CPU
- Use triton.language (tl) for GPU operations
- Launch kernels with kernel_name[grid](args)

Resources:
- https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# CONCEPT: What is Triton?
# ============================================================================
# Triton is a language and compiler for writing GPU kernels in Python.
# Unlike CUDA, Triton handles many low-level details automatically.
# 
# Key ideas:
# - Kernels are functions that run on the GPU
# - Many copies of the kernel run in parallel (called "programs")
# - Each program processes a block of data
# ============================================================================


# ============================================================================
# Exercise 1: Your First Kernel
# ============================================================================
# Write a kernel that adds 1 to every element in an array.
# This is the "Hello World" of GPU programming.

@triton.jit
def add_one_kernel(
    x_ptr,      # Pointer to input array
    out_ptr,    # Pointer to output array  
    n_elements, # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Number of elements per block
):
    """
    Add 1 to every element in the input array.
    
    Each program instance processes BLOCK_SIZE elements.
    """
    # TODO: Get the program ID (which block are we?)
    # HINT: Use tl.program_id(axis=0)
    pid = None  # Replace with correct code
    
    # TODO: Calculate the starting index for this block
    # HINT: block_start = pid * BLOCK_SIZE
    block_start = None  # Replace with correct code
    
    # TODO: Generate offsets for this block [0, 1, 2, ..., BLOCK_SIZE-1]
    # HINT: Use tl.arange(0, BLOCK_SIZE)
    offsets = None  # Replace with correct code
    
    # TODO: Create a mask for bounds checking (important for last block!)
    # HINT: mask = (block_start + offsets) < n_elements
    mask = None  # Replace with correct code
    
    # TODO: Load data from input pointer
    # HINT: Use tl.load(x_ptr + block_start + offsets, mask=mask)
    x = None  # Replace with correct code
    
    # TODO: Add 1 to the loaded values
    output = None  # Replace with correct code
    
    # TODO: Store the result
    # HINT: Use tl.store(out_ptr + block_start + offsets, output, mask=mask)
    pass  # Replace with correct code


def add_one(x: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to launch the kernel.
    """
    # Ensure input is on GPU
    assert x.is_cuda, "Input must be on GPU"
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Get total elements
    n_elements = x.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size (number of blocks needed)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # TODO: Launch the kernel
    # HINT: add_one_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    pass  # Replace with correct code
    
    return output


# ============================================================================
# Exercise 2: Understanding Grid and Blocks
# ============================================================================
# Modify the kernel to print (conceptually) which program is processing which data.
# This helps understand parallelism.

@triton.jit
def print_ids_kernel(
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    A kernel that helps visualize how work is distributed.
    
    In practice, printing from GPU kernels isn't straightforward,
    but this exercise helps you understand the concept.
    """
    # TODO: Get program ID
    pid = None  # Replace with correct code
    
    # TODO: Calculate start and end indices this program handles
    start_idx = None  # Replace with correct code
    end_idx = None    # Replace with correct code (min of start + BLOCK_SIZE, n_elements)


# ============================================================================
# Exercise 3: Simple Element-wise Operation
# ============================================================================
# Write a kernel that squares every element.

@triton.jit
def square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Square every element: out[i] = x[i] * x[i]
    """
    # TODO: Implement the kernel
    # Follow the same pattern as add_one_kernel:
    # 1. Get program ID
    # 2. Calculate block start
    # 3. Generate offsets
    # 4. Create mask
    # 5. Load data
    # 6. Compute (square the values)
    # 7. Store result
    pass


def square(x: torch.Tensor) -> torch.Tensor:
    """Launch the square kernel."""
    assert x.is_cuda
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    # TODO: Calculate grid and launch kernel
    grid = None  # Replace with correct code
    # Launch kernel here
    
    return output


# ============================================================================
# Tests
# ============================================================================

def test_add_one():
    """Test the add_one function."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    result = add_one(x)
    expected = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0], device='cuda')
    
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_add_one PASSED")


def test_add_one_large():
    """Test with larger array to ensure blocking works."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    x = torch.arange(10000, dtype=torch.float32, device='cuda')
    result = add_one(x)
    expected = x + 1
    
    assert torch.allclose(result, expected), "Large array test failed"
    print("test_add_one_large PASSED")


def test_square():
    """Test the square function."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device='cuda')
    result = square(x)
    expected = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0], device='cuda')
    
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_square PASSED")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Day 1: Hello Triton")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Install CUDA to run Triton kernels.")
        print("You can still read and understand the code.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print()
        
        print("Running tests...")
        test_add_one()
        test_add_one_large()
        test_square()
        print()
        print("All tests passed! Great job on your first Triton kernels!")
