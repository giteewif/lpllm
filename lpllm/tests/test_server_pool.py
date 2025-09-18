#!/usr/bin/env python3
"""
Test script for ServerPinnedMemoryPool
"""
import torch
import time
import sys
import os

# Add current directory to Python path
sys.path.insert(0, '/mnt/zhengcf3/lpllm')

from lpllm.server_pool import ServerPinnedMemoryPool
from lpllm.pinpool import PinnedMemoryPool
from lpllm.cuda_memcpy_utils import cuda_copy_, safe_copy_

def test_server_pool_basic():
    """Basic functionality test for ServerPinnedMemoryPool"""
    print("Testing ServerPinnedMemoryPool basic functionality...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available, skipping test")
        return
    
    try:
        # Create server pool
        pool = ServerPinnedMemoryPool(
            dtype=torch.bfloat16,
            pool_size=20480,  # 20480MB
            device=device
        )
        
        print(f"Created pool: {pool.get_usage_info()}")
        
        # Test allocation
        test_tensor = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
        allocated_block = pool.alloc_same_pin_tensor(test_tensor)
        
        print(f"Allocated block shape: {allocated_block.shape}")
        print(f"Pool usage after allocation: {pool.get_usage_info()}")
        
        # Test copy operation
        start_time = time.time()
        cuda_copy_(allocated_block, test_tensor, non_blocking=True)
        copy_time = time.time() - start_time
        
        print(f"Copy time: {copy_time:.6f} seconds")
        
        # Test free operation
        pool.free(allocated_block)
        print(f"Pool usage after free: {pool.get_usage_info()}")
        
        print("✓ ServerPinnedMemoryPool basic test passed!")
        
    except Exception as e:
        print(f"✗ ServerPinnedMemoryPool test failed: {e}")
        import traceback
        traceback.print_exc()

def test_compare_pools():
    """Compare ServerPinnedMemoryPool vs regular PinnedMemoryPool"""
    print("\nComparing ServerPinnedMemoryPool vs PinnedMemoryPool...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("CUDA not available, skipping comparison test")
        return
        
    try:
        # Test data
        test_tensor = torch.randn(2048, 2048, dtype=torch.bfloat16, device=device)
        num_iterations = 10
        
        # Test regular pool
        regular_pool = PinnedMemoryPool(dtype=torch.bfloat16, pool_size=64)
        regular_times = []
        
        for _ in range(num_iterations):
            start = time.time()
            block = regular_pool.alloc_same_pin_tensor(test_tensor)
            cuda_copy_(block, test_tensor, non_blocking=True)
            regular_pool.free(block)
            regular_times.append(time.time() - start)
        
        # Test server pool
        server_pool = ServerPinnedMemoryPool(
            dtype=torch.bfloat16,
            pool_size=64,
            device=device
        )
        server_times = []
        
        for _ in range(num_iterations):
            start = time.time()
            block = server_pool.alloc_same_pin_tensor(test_tensor)
            cuda_copy_(block, test_tensor, non_blocking=True)
            server_pool.free(block)
            server_times.append(time.time() - start)
        
        avg_regular = sum(regular_times) / len(regular_times)
        avg_server = sum(server_times) / len(server_times)
        
        print(f"Regular pool average time: {avg_regular:.6f}s")
        print(f"Server pool average time: {avg_server:.6f}s")
        print(f"Performance ratio: {avg_server/avg_regular:.2f}x")
        
        print("✓ Pool comparison test completed!")
        
    except Exception as e:
        print(f"✗ Pool comparison test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Starting ServerPinnedMemoryPool tests...")
    print("=" * 50)
    
    test_server_pool_basic()
    test_compare_pools()
    
    print("=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    main()
