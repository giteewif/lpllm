#!/usr/bin/env python3
"""
Example usage of the server pre-allocated memory pool implementation
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server_pool import ServerPinnedMemoryPool

def main():
    """Example usage of ServerPinnedMemoryPool with server pre-allocated memory"""
    print("=== Server Pre-allocated Memory Pool Example ===")
    print("Note: Make sure the server is running with pre-allocated memory")
    
    # Create a server memory pool client
    pool = ServerPinnedMemoryPool(
        dtype=torch.bfloat16,
        pool_size=512,  # 512MB from server's pre-allocated memory
        device="cuda:0"
    )
    
    print(f"✓ Created server memory pool client")
    print(f"  - Pool size: {pool.pool_size}MB")
    print(f"  - Server memory size: {pool.server_memory_size} bytes")
    print(f"  - Free blocks: {len(pool.free_blocks)}")
    print(f"  - Server allocation ID: {pool.server_allocation_info['allocation_id']}")
    
    # Create some test tensors
    tensors = [
        torch.randn(100, 100, dtype=torch.bfloat16),
        torch.randn(50, 200, dtype=torch.bfloat16),
        torch.randn(75, 75, dtype=torch.bfloat16)
    ]
    
    memory_blocks = []
    
    # Allocate tensors and copy data
    for i, tensor in enumerate(tensors):
        print(f"\n--- Processing tensor {i+1} ---")
        print(f"  - Shape: {tensor.shape}")
        print(f"  - Size: {tensor.numel()} elements")
        
        # Allocate tensor directly from server memory
        allocated_tensor = pool.alloc_same_pin_tensor(tensor)
        memory_blocks.append(allocated_tensor)
        print(f"  ✓ Allocated tensor from server memory")
        print(f"  - Has server pool info: {hasattr(allocated_tensor, '_server_pool_info')}")
        
        # Copy data to allocated tensor
        allocated_tensor.copy_(tensor)
        print(f"  ✓ Copied data to allocated tensor")
        
        # Move to GPU
        result = allocated_tensor.to("cuda:0")
        print(f"  ✓ Moved tensor to GPU")
        
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
    view_tensor = memory_blocks[0].view(50, 200)
    print(f"  ✓ Created view: {view_tensor.shape}, {view_tensor.numel()} elements")
    
    # Free memory blocks
    print(f"\n--- Freeing Memory Blocks ---")
    for i, memory_block in enumerate(memory_blocks):
        pool.free(memory_block)
        print(f"  ✓ Freed tensor {i+1}")
    
    # Test re-allocation after freeing
    print(f"\n--- Testing Re-allocation ---")
    new_tensor = torch.randn(30, 30, dtype=torch.bfloat16)
    new_allocated = pool.alloc_same_pin_tensor(new_tensor)
    new_allocated.copy_(new_tensor)
    new_result = new_allocated.to("cuda:0")
    
    if torch.allclose(new_tensor, new_result, rtol=1e-3):
        print(f"  ✓ Re-allocation test passed")
    else:
        print(f"  ✗ Re-allocation test failed")
        return False
    
    pool.free(new_allocated)
    
    # Final usage info
    print(f"\n--- Final Memory Usage ---")
    final_usage = pool.get_usage_info()
    print(f"  - Active allocations: {final_usage['active_allocations']}")
    print(f"  - Free bytes: {final_usage['free_bytes']}")
    print(f"  - Utilization: {final_usage['utilization']:.2%}")
    
    print(f"\n🎉 All operations completed successfully!")
    print(f"✓ Server pre-allocated memory pool returns tensors directly")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

