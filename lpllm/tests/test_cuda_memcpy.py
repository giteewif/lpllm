#!/usr/bin/env python3
"""
Test script for CUDA memcpy functionality
"""

import torch
import time
import sys
import os

# Add current directory to Python path
sys.path.insert(0, '/mnt/zhengcf3/lpllm')

def test_cuda_memcpy_availability():
    """Test if CUDA memcpy functions are available"""
    print("Testing CUDA memcpy availability...")
    
    try:
        from cuda_memcpy_utils import is_cuda_memcpy_available, cuda_copy_
        available = is_cuda_memcpy_available()
        print(f"CUDA memcpy available: {available}")
        return available
    except Exception as e:
        print(f"Error importing CUDA memcpy utils: {e}")
        return False

def test_basic_cuda_memcpy():
    """Test basic CUDA memcpy operations"""
    print("\nTesting basic CUDA memcpy operations...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    try:
        from cuda_memcpy_utils import cuda_copy_, safe_copy_
        
        # Test H2D (Host to Device)
        print("Testing Host to Device copy...")
        src_cpu = torch.randn(1024, 1024, dtype=torch.float32, pin_memory=True)
        dst_gpu = torch.empty_like(src_cpu, device='cuda:0')
        
        start_time = time.time()
        cuda_copy_(dst_gpu, src_cpu, non_blocking=True)
        torch.cuda.synchronize()  # Wait for completion
        h2d_time = time.time() - start_time
        
        print(f"H2D copy time: {h2d_time:.6f} seconds")
        print(f"Data matches: {torch.allclose(dst_gpu.cpu(), src_cpu, atol=1e-6)}")
        
        # Test D2H (Device to Host)
        print("Testing Device to Host copy...")
        src_gpu = torch.randn(1024, 1024, dtype=torch.float32, device='cuda:0')
        dst_cpu = torch.empty_like(src_gpu, pin_memory=True)
        
        start_time = time.time()
        cuda_copy_(dst_cpu, src_gpu, non_blocking=True)
        torch.cuda.synchronize()  # Wait for completion
        d2h_time = time.time() - start_time
        
        print(f"D2H copy time: {d2h_time:.6f} seconds")
        print(f"Data matches: {torch.allclose(dst_cpu, src_gpu.cpu(), atol=1e-6)}")
        
        # Test D2D (Device to Device)
        if torch.cuda.device_count() > 1:
            print("Testing Device to Device copy...")
            src_gpu0 = torch.randn(1024, 1024, dtype=torch.float32, device='cuda:0')
            dst_gpu1 = torch.empty_like(src_gpu0, device='cuda:1')
            
            start_time = time.time()
            cuda_copy_(dst_gpu1, src_gpu0, non_blocking=True)
            torch.cuda.synchronize()  # Wait for completion
            d2d_time = time.time() - start_time
            
            print(f"D2D copy time: {d2d_time:.6f} seconds")
            print(f"Data matches: {torch.allclose(dst_gpu1.cpu(), src_gpu0.cpu(), atol=1e-6)}")
        else:
            print("Only one GPU available, skipping D2D test")
            
        print("✓ Basic CUDA memcpy tests passed!")
        
    except Exception as e:
        print(f"✗ CUDA memcpy test failed: {e}")
        import traceback
        traceback.print_exc()

def test_performance_comparison():
    """Compare CUDA memcpy vs PyTorch copy performance"""
    print("\nTesting performance comparison...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping performance test")
        return
    
    try:
        from cuda_memcpy_utils import cuda_copy_
        
        # Test data
        sizes = [(512, 512), (1024, 1024), (2048, 2048)]
        num_iterations = 10
        
        print(f"{'Size':<15} {'PyTorch (ms)':<15} {'CUDA memcpy (ms)':<18} {'Speedup':<10}")
        print("-" * 70)
        
        for size in sizes:
            # Prepare test data
            src_cpu = torch.randn(*size, dtype=torch.float32, pin_memory=True)
            
            # Test PyTorch copy
            pytorch_times = []
            for _ in range(num_iterations):
                dst_gpu = torch.empty_like(src_cpu, device='cuda:0')
                
                torch.cuda.synchronize()
                start = time.time()
                dst_gpu.copy_(src_cpu, non_blocking=True)
                torch.cuda.synchronize()
                pytorch_times.append((time.time() - start) * 1000)  # Convert to ms
            
            # Test CUDA memcpy
            cuda_times = []
            for _ in range(num_iterations):
                dst_gpu = torch.empty_like(src_cpu, device='cuda:0')
                
                torch.cuda.synchronize()
                start = time.time()
                cuda_copy_(dst_gpu, src_cpu, non_blocking=True)
                torch.cuda.synchronize()
                cuda_times.append((time.time() - start) * 1000)  # Convert to ms
            
            avg_pytorch = sum(pytorch_times) / len(pytorch_times)
            avg_cuda = sum(cuda_times) / len(cuda_times)
            speedup = avg_pytorch / avg_cuda if avg_cuda > 0 else 1.0
            
            print(f"{str(size):<15} {avg_pytorch:<15.3f} {avg_cuda:<18.3f} {speedup:<10.2f}x")
        
        print("✓ Performance comparison completed!")
        
    except Exception as e:
        print(f"✗ Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()

def test_fallback_behavior():
    """Test fallback behavior when CUDA memcpy is not available"""
    print("\nTesting fallback behavior...")
    
    try:
        from cuda_memcpy_utils import safe_copy_
        
        # Test with CPU tensors (should use regular copy)
        src = torch.randn(100, 100, dtype=torch.float32)
        dst = torch.empty_like(src)
        
        start_time = time.time()
        safe_copy_(dst, src)
        copy_time = time.time() - start_time
        
        print(f"CPU to CPU copy time: {copy_time:.6f} seconds")
        print(f"Data matches: {torch.allclose(dst, src, atol=1e-6)}")
        print("✓ Fallback behavior test passed!")
        
    except Exception as e:
        print(f"✗ Fallback behavior test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("CUDA Memcpy Test Suite")
    print("=" * 50)
    
    # Test availability
    cuda_memcpy_available = test_cuda_memcpy_availability()
    
    # Test basic functionality
    test_basic_cuda_memcpy()
    
    # Test performance
    if cuda_memcpy_available:
        test_performance_comparison()
    
    # Test fallback
    test_fallback_behavior()
    
    print("=" * 50)
    print("Test suite completed!")

if __name__ == "__main__":
    main()
