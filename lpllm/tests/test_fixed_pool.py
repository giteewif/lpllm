#!/usr/bin/env python3
"""
Test script for fixed ServerPinnedMemoryPool
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server_pool import ServerPinnedMemoryPool

def test_fixed_pool():
    """Test the fixed ServerPinnedMemoryPool"""
    print("=== Testing Fixed ServerPinnedMemoryPool ===")
    
    try:
        # Create a server memory pool client
        pool = ServerPinnedMemoryPool(
            dtype=torch.bfloat16,
            pool_size=256,  # 256MB from server
            device="cuda:0"
        )
        
        print(f"✓ Pool created successfully")
        print(f"  - Client ID: {pool.client_id}")
        print(f"  - Server memory size: {pool.server_memory_size} bytes")
        print(f"  - Free blocks: {len(pool.free_blocks)}")
        
        # Create test tensor
        test_tensor = torch.randn(10, 10, dtype=torch.bfloat16)
        print(f"✓ Created test tensor: {test_tensor.shape}, {test_tensor.dtype}")
        
        # Allocate tensor from server memory
        allocated_tensor = pool.alloc_same_pin_tensor(test_tensor)
        print(f"✓ Allocated tensor: {allocated_tensor.shape}, {allocated_tensor.dtype}")
        print(f"  - Has server pool info: {hasattr(allocated_tensor, '_server_pool_info')}")
        
        # Test copy operation
        allocated_tensor.copy_(test_tensor)
        print(f"✓ Copied data to allocated tensor")
        
        # Test device transfer
        gpu_tensor = allocated_tensor.to("cuda:0")
        print(f"✓ Moved to GPU: {gpu_tensor.device}")
        
        # Verify data integrity
        if torch.allclose(test_tensor, gpu_tensor, rtol=1e-3):
            print(f"✓ Data integrity verified")
        else:
            print(f"✗ Data integrity check failed")
            return False
        
        # Test view operation
        view_tensor = allocated_tensor.view(5, 20)
        print(f"✓ Created view: {view_tensor.shape}")
        
        # Test free operation
        pool.free(allocated_tensor)
        print(f"✓ Freed allocated tensor")
        
        print(f"\n🎉 All tests passed! Fixed ServerPinnedMemoryPool works correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            del pool
        except:
            pass

if __name__ == "__main__":
    success = test_fixed_pool()
    sys.exit(0 if success else 1)
