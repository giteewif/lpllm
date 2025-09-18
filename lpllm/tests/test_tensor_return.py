#!/usr/bin/env python3
"""
Test script to verify that alloc_same_pin_tensor returns tensor directly
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '/mnt/zhengcf3/lpllm')

from lpllm.server_pool import ServerPinnedMemoryPool

def test_tensor_return():
    """Test that alloc_same_pin_tensor returns tensor directly"""
    print("=== Testing Tensor Return ===")
    
    try:
        # Create a server memory pool client
        pool = ServerPinnedMemoryPool(
            dtype=torch.bfloat16,
            pool_size=1,  # GB from server
            device="cuda:0"
        )
        
        print(f"✓ Created server memory pool client")
        
        # Create test tensor
        test_tensor = torch.randn(160, 16, 512, 128, dtype=torch.bfloat16)
        print(f"✓ Created test tensor: {test_tensor.shape}, {test_tensor.dtype}")
        
        # Allocate tensor from server memory
        allocated_tensor = pool.alloc_same_pin_tensor(test_tensor)
        print(f"✓ Allocated tensor: {allocated_tensor.shape}, {allocated_tensor.dtype}")
        print(f"  - Type: {type(allocated_tensor)}")
        print(f"  - Is tensor: {torch.is_tensor(allocated_tensor)}")
        print(f"  - Has server pool info: {hasattr(allocated_tensor, '_server_pool_info')}")
        
        
        # Test tensor operations
        allocated_tensor.copy_(test_tensor)
        # print(allocated_tensor)
        print(f"✓ Copied data to allocated tensor")
        
        # Test view operation
        view_tensor = allocated_tensor.view(160, 16, 512, 128)
        print(f"✓ Created view: {view_tensor.shape}")
        
        # Test device transfer
        gpu_tensor = allocated_tensor.to("cuda:0")
        print(f"✓ Moved to GPU: {gpu_tensor.device}")
        
        # Verify data integrity - move test_tensor to same device as gpu_tensor
        test_tensor_gpu = test_tensor.to(gpu_tensor.device)
        if torch.allclose(test_tensor_gpu, gpu_tensor, rtol=1e-3):
            print(f"✓ Data integrity verified")
        else:
            print(f"✗ Data integrity check failed")
            return False
        
        # Test free operation
        pool.free(allocated_tensor)
        print(f"✓ Freed allocated tensor")
        
        print(f"\n🎉 All tests passed! alloc_same_pin_tensor returns tensor directly.")
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
    success = test_tensor_return()
    sys.exit(0 if success else 1)
