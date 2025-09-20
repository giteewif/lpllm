#!/usr/bin/env python3
"""
Simple test for PinnedMemoryPool
"""

import torch
import sys
sys.path.append('/mnt/zhengcf3/lpllm')
import time
try:
    from lpllm.lpllm.pinpool import PinnedMemoryPool
    print("✅ Successfully imported PinnedMemoryPool")

    # Test basic functionality
    time_start = time.time()
    pool = PinnedMemoryPool(dtype=torch.bfloat16, pool_size=2)
    print(f"create pin cost {time.time()-time_start} s")
    print(f"✅ Created pool: {pool.get_usage_info()}")

    # Test allocation
    tensor = pool.alloc(100)  # 100MB
    print(f"✅ Allocated tensor: {tensor.shape}")

    # Test free
    pool.free(tensor)
    print("✅ Freed tensor")
    print(f"✅ Final state: {pool.get_usage_info()}")

    print("🎉 Basic test passed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
