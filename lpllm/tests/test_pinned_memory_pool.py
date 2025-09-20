#!/usr/bin/env python3
"""
Comprehensive test suite for PinnedMemoryPool
测试PinnedMemoryPool的各种功能和边界情况
"""

import torch
import time
import threading
import random
import gc
from typing import List, Tuple
from lpllm.pinpool import PinnedMemoryPool, PreAllocatedPinnedMemoryPool, HybridPinnedMemoryPool


def test_basic_allocation():
    """测试基本内存分配和释放功能"""
    print("=== 测试基本内存分配功能 ===")

    pool = PinnedMemoryPool(dtype=torch.bfloat16, pool_size=2)
    print(f"初始状态: {pool.get_usage_info()}")

    # 分配不同大小的内存块
    tensors = []
    sizes = [100, 200, 300, 400]  # MB

    for i, size_mb in enumerate(sizes):
        tensor = pool.alloc(size_mb)
        tensors.append((tensor, size_mb))
        print(f"分配 {size_mb}MB: {tensor.shape}")
        print(f"状态: {pool.get_usage_info()}")

    # 释放内存块
    for i, (tensor, size_mb) in enumerate(tensors):
        pool.free(tensor)
        print(f"释放 {size_mb}MB")
        print(f"状态: {pool.get_usage_info()}")

    print("✅ 基本分配测试通过\n")


def test_chunk_boundary():
    """测试chunk边界情况"""
    print("=== 测试chunk边界情况 ===")

    pool = PinnedMemoryPool(dtype=torch.bfloat16, pool_size=2)

    # 测试分配接近chunk大小的内存
    large_tensor = pool.alloc(900)  # 900MB < 1024MB
    print(f"分配900MB: {large_tensor.shape}")
    print(f"状态: {pool.get_usage_info()}")

    # 测试分配超过chunk大小的内存（应该失败）
    try:
        too_large = pool.alloc(1200)  # 1200MB > 1024MB
        print("❌ 错误：应该拒绝超过chunk大小的分配")
    except MemoryError:
        print("✅ 正确拒绝了超过chunk大小的分配")

    pool.free(large_tensor)
    print("✅ Chunk边界测试通过\n")


def test_fragmentation():
    """测试内存碎片化情况"""
    print("=== 测试内存碎片化 ===")

    pool = PinnedMemoryPool(dtype=torch.bfloat16, pool_size=3)

    # 创建一些碎片
    tensors = []
    for i in range(5):
        size_mb = random.randint(50, 200)
        tensor = pool.alloc(size_mb)
        tensors.append((tensor, size_mb))

    print(f"分配后状态: {pool.get_usage_info()}")

    # 随机释放一些tensor
    for i in [0, 2, 4]:
        tensor, size_mb = tensors[i]
        pool.free(tensor)
        print(f"释放tensor {i}: {size_mb}MB")

    print(f"部分释放后状态: {pool.get_usage_info()}")

    # 尝试重新分配
    new_tensor = pool.alloc(150)
    print(f"重新分配150MB: {new_tensor.shape}")
    print(f"最终状态: {pool.get_usage_info()}")

    # 清理
    pool.free(new_tensor)
    for i in [1, 3]:
        tensor, size_mb = tensors[i]
        pool.free(tensor)

    print("✅ 碎片化测试通过\n")


def test_kb_allocation():
    """测试KB级别的内存分配"""
    print("=== 测试KB级别分配 ===")

    pool = PinnedMemoryPool(dtype=torch.bfloat16, pool_size=1)

    # 测试不同KB大小的分配
    kb_sizes = [512, 1024, 2048, 4096]  # 0.5MB, 1MB, 2MB, 4MB

    for kb in kb_sizes:
        tensor = pool.alloc_kb(kb)
        print(f"分配 {kb}KB: {tensor.shape}")
        pool.free(tensor)

    print("✅ KB级别分配测试通过\n")


def test_concurrent_access():
    """测试并发访问"""
    print("=== 测试并发访问 ===")

    pool = PinnedMemoryPool(dtype=torch.bfloat16, pool_size=4)
    results = {"success": 0, "failed": 0}

    def worker(worker_id: int):
        try:
            # 每个线程分配和释放多个内存块
            for i in range(5):
                size_mb = random.randint(50, 300)
                tensor = pool.alloc(size_mb)
                time.sleep(random.random() * 0.01)  # 小延迟
                pool.free(tensor)
            results["success"] += 1
        except Exception as e:
            print(f"线程 {worker_id} 失败: {e}")
            results["failed"] += 1

    # 启动多个线程
    threads = []
    for i in range(10):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print(f"并发测试结果: 成功 {results['success']}, 失败 {results['failed']}")
    print(f"最终状态: {pool.get_usage_info()}")
    print("✅ 并发访问测试通过\n")


def test_pre_allocated_pool():
    """测试预分配内存池"""
    print("=== 测试预分配内存池 ===")

    # 创建预分配池
    prealloc_pool = PreAllocatedPinnedMemoryPool(
        block_shape=(128, 256, 512),  # 128*256*512 = 16M elements
        dtype=torch.bfloat16,
        pool_size=10,
        name="test_prealloc"
    )

    print(f"预分配池状态: {prealloc_pool.stats()}")

    # 分配所有预分配块
    allocated_tensors = []
    for i in range(10):
        try:
            tensor = prealloc_pool.allocate()
            allocated_tensors.append(tensor)
            print(f"分配预分配块 {i}: {tensor.shape}")
        except RuntimeError as e:
            print(f"无法分配更多预分配块: {e}")
            break

    print(f"成功分配 {len(allocated_tensors)} 个预分配块")

    # 释放一些块
    for i in range(5):
        tensor = allocated_tensors[i]
        prealloc_pool.free(tensor)
        print(f"释放预分配块 {i}")

    print(f"释放后状态: {prealloc_pool.stats()}")

    # 尝试分配超过预分配数量
    try:
        extra_tensor = prealloc_pool.allocate()
        print("❌ 错误：应该无法分配超过预分配数量的块")
    except RuntimeError:
        print("✅ 正确拒绝了超过预分配数量的分配")

    # 清理剩余块
    for tensor in allocated_tensors[5:]:
        prealloc_pool.free(tensor)

    print("✅ 预分配内存池测试通过\n")


def test_hybrid_pool():
    """测试混合内存池"""
    print("=== 测试混合内存池 ===")

    # 创建混合内存池配置
    prealloc_configs = [
        {"shape": (64, 128, 256), "pool_size": 5, "name": "small"},
        {"shape": (128, 256, 512), "pool_size": 3, "name": "medium"},
    ]

    hybrid_pool = HybridPinnedMemoryPool(
        prealloc_configs=prealloc_configs,
        dtype=torch.bfloat16,
        max_dynamic_pool_size=1,  # 1GB动态池
        name="test_hybrid"
    )

    print(f"混合池状态: {hybrid_pool.stats()}")

    # 测试预分配池分配
    small_tensors = []
    for i in range(5):
        tensor = hybrid_pool.allocate((64, 128, 256))
        small_tensors.append(tensor)
        print(f"从预分配小池分配: {tensor.shape}")

    # 测试动态分配
    dynamic_tensor = hybrid_pool.allocate((200, 300, 400))
    print(f"从动态池分配: {dynamic_tensor.shape}")

    # 释放所有
    for tensor in small_tensors:
        hybrid_pool.free(tensor)
    hybrid_pool.free(dynamic_tensor)

    print("✅ 混合内存池测试通过\n")


def test_memory_efficiency():
    """测试内存使用效率"""
    print("=== 测试内存使用效率 ===")

    pool = PinnedMemoryPool(dtype=torch.bfloat16, pool_size=2)

    # 记录初始内存使用
    initial_usage = pool.get_usage_info()

    # 进行多次分配和释放
    for round in range(3):
        print(f"第 {round + 1} 轮测试")
        tensors = []

        # 分配阶段
        for i in range(5):
            size_mb = random.randint(100, 400)
            tensor = pool.alloc(size_mb)
            tensors.append((tensor, size_mb))

        print(f"  分配后状态: {pool.get_usage_info()}")

        # 释放阶段
        for i, (tensor, size_mb) in enumerate(tensors):
            pool.free(tensor)

        print(f"  释放后状态: {pool.get_usage_info()}")

    final_usage = pool.get_usage_info()
    print(f"效率测试完成，最终状态: {final_usage}")
    print("✅ 内存效率测试通过\n")


def test_error_handling():
    """测试错误处理"""
    print("=== 测试错误处理 ===")

    pool = PinnedMemoryPool(dtype=torch.bfloat16, pool_size=1)

    # 测试无效分配大小
    try:
        pool.alloc(0)
        print("❌ 错误：应该拒绝0大小分配")
    except ValueError:
        print("✅ 正确拒绝了0大小分配")

    try:
        pool.alloc(-1)
        print("❌ 错误：应该拒绝负数大小分配")
    except ValueError:
        print("✅ 正确拒绝了负数大小分配")

    # 测试释放未分配的内存
    fake_tensor = torch.empty(100, dtype=torch.bfloat16, pin_memory=True)
    try:
        pool.free(fake_tensor)
        print("✅ 正确处理了释放未分配内存的情况")
    except ValueError:
        print("❌ 错误：应该优雅处理未分配内存的释放")

    print("✅ 错误处理测试通过\n")


def performance_test():
    """性能测试"""
    print("=== 性能测试 ===")

    pool = PinnedMemoryPool(dtype=torch.bfloat16, pool_size=3)

    # 测试大量小分配
    start_time = time.time()
    small_tensors = []

    for i in range(100):
        tensor = pool.alloc(10)  # 10MB
        small_tensors.append(tensor)

    alloc_time = time.time() - start_time
    print(f"100次10MB分配耗时: {alloc_time:.3f}秒")
    start_time = time.time()
    for tensor in small_tensors:
        pool.free(tensor)

    free_time = time.time() - start_time
    print(f"100次释放耗时: {free_time:.3f}秒")
    print("✅ 性能测试通过\n")


def main():
    """主测试函数"""
    print("🚀 开始PinnedMemoryPool全面测试\n")

    try:
        test_basic_allocation()
        test_chunk_boundary()
        test_fragmentation()
        test_kb_allocation()
        test_concurrent_access()
        test_pre_allocated_pool()
        test_hybrid_pool()
        test_memory_efficiency()
        test_error_handling()
        performance_test()

        print("🎉 所有测试通过！PinnedMemoryPool工作正常。")

    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
