#!/usr/bin/env python3
"""
Example usage of the local pinned memory pool implementation
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server_pool import ServerPinnedMemoryPool

def main():
    """Example usage of ServerPinnedMemoryPool with local pinned memory"""
    print("=== Local Pinned Memory Pool Example ===")
    
    # Create a local pinned memory pool
    pool = ServerPinnedMemoryPool(
        dtype=torch.bfloat16,
        pool_size=512,  # 512MB local pinned memory
        device="cuda:0"
    )
    
    print(f"✓ Created local pinned memory pool")
    print(f"  - Pool size: {pool.pool_size}MB")
    print(f"  - Local memory size: {pool.server_memory_size} bytes")
    print(f"  - Free blocks: {len(pool.free_blocks)}")
    
    # Create some test tensors
    tensors = [
        torch.randn(100, 100, dtype=torch.bfloat16),
        torch.randn(50, 200, dtype=torch.bfloat16),
        torch.randn(75, 75, dtype=torch.bfloat16)
    ]
    
    memory_blocks = []
    
    # Allocate memory blocks and copy data
    for i, tensor in enumerate(tensors):
        print(f"\n--- Processing tensor {i+1} ---")
        print(f"  - Shape: {tensor.shape}")
        print(f"  - Size: {tensor.numel()} elements")
        
        # Allocate memory block
        memory_block = pool.alloc_same_pin_tensor(tensor)
        memory_blocks.append(memory_block)
        print(f"  ✓ Allocated memory block")
        
        # Copy data to local pinned memory (no network call)
        memory_block.copy_(tensor)
        print(f"  ✓ Copied data to local pinned memory")
        
        # Read data back (no network call)
        result = memory_block.to("cuda:0")
        print(f"  ✓ Read data back from local memory")
        
        # Verify data integrity
        if torch.allclose(tensor, result, rtol=1e-3):
            print(f"  ✓ Data integrity verified")
        else:
            print(f"  ✗ Data integrity check failed")
            return False
    
    # Test memory usage info
    print(f"\n--- Memory Usage Info ---")
    usage_info = pool.get_usage_info()
    for key, value in usage_info.items():
        print(f"  - {key}: {value}")
    
    # Test view operation
    print(f"\n--- Testing View Operation ---")
    view_block = memory_blocks[0].view(50, 200)
    print(f"  ✓ Created view: {view_block.numel()} elements")
    
    # Free memory blocks
    print(f"\n--- Freeing Memory Blocks ---")
    for i, memory_block in enumerate(memory_blocks):
        pool.free(memory_block)
        print(f"  ✓ Freed memory block {i+1}")
    
    # Test re-allocation after freeing
    print(f"\n--- Testing Re-allocation ---")
    new_tensor = torch.randn(30, 30, dtype=torch.bfloat16)
    new_block = pool.alloc_same_pin_tensor(new_tensor)
    new_block.copy_(new_tensor)
    new_result = new_block.to("cuda:0")
    
    if torch.allclose(new_tensor, new_result, rtol=1e-3):
        print(f"  ✓ Re-allocation test passed")
    else:
        print(f"  ✗ Re-allocation test failed")
        return False
    
    pool.free(new_block)
    
    # Final usage info
    print(f"\n--- Final Memory Usage ---")
    final_usage = pool.get_usage_info()
    print(f"  - Active allocations: {final_usage['active_allocations']}")
    print(f"  - Free bytes: {final_usage['free_bytes']}")
    print(f"  - Utilization: {final_usage['utilization']:.2%}")
    
    print(f"\n🎉 All operations completed successfully!")
    print(f"✓ Local pinned memory pool works without any network communication")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

