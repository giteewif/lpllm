import torch
import threading
import collections
from typing import Dict, List, Optional, Tuple, Union, Deque

import threading
import torch

class PinnedMemoryPool:
    def __init__(self,
                 dtype: torch.dtype = torch.bfloat16,
                 pool_size: int = 2):
        self.dtype = dtype
        self.pool_size = pool_size*1024*1024*1024  # GB
        
        # 计算总字节数和元素数量
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        total_bytes = pool_size * 1024 * 1024 * 1024
        self.total_elements = total_bytes // self.element_size
        
        # 分配固定内存
        self.pool_memory = torch.empty(self.total_elements, dtype=dtype, pin_memory=True)
        
        # 内存管理数据结构
        self.lock = threading.RLock()
        self.allocated_blocks = []  # 记录已分配的内存块 (start, size)
        self.free_list = [(0, self.total_elements)]  # 初始时整个内存池都是空闲的
    def alloc_kb(self, size_kb: int):
        """分配指定KB大小的连续内存块"""
        if size_kb <= 0:
            raise ValueError("分配大小必须为正数")
        required_bytes = size_kb * 1024
        required_elements = (required_bytes + self.element_size - 1 ) // self.element_size
        required_elements = int(required_elements)
        with self.lock:
            # 寻找合适的空闲块（首次适应算法）
            for i, (start, size) in enumerate(self.free_list):
                if size >= required_elements:
                    # 分配这块内存
                    allocated_start = start
                    allocated_size = required_elements
                    
                    # 更新空闲列表
                    if size == required_elements:
                        # 完全匹配，移除这个空闲块
                        self.free_list.pop(i)
                    else:
                        # 部分使用，更新空闲块起始位置和大小
                        self.free_list[i] = (start + required_elements, size - required_elements)
                    
                    # 记录已分配块
                    self.allocated_blocks.append((allocated_start, allocated_size))
                    
                    # print(allocated_start, allocated_size)
                    # 返回内存视图（不拷贝数据）
                    return self.pool_memory[allocated_start:allocated_start+allocated_size]
            
            raise MemoryError("内存池中没有足够的连续空间")
    def alloc_same_pin_tensor(self, tensor: torch.Tensor):
        """分配与给定Tensor形状和类型匹配的内存块"""
        # 直接按元素数量分配，避免KB对齐导致的大小不匹配
        required_elements = tensor.numel()
        
        with self.lock:
            # 寻找合适的空闲块（首次适应算法）
            for i, (start, size) in enumerate(self.free_list):
                if size >= required_elements:
                    # 分配这块内存
                    allocated_start = start
                    allocated_size = required_elements
                    
                    # 更新空闲列表
                    if size == required_elements:
                        # 完全匹配，移除这个空闲块
                        self.free_list.pop(i)
                    else:
                        # 部分使用，更新空闲块起始位置和大小
                        self.free_list[i] = (start + required_elements, size - required_elements)
                    
                    # 记录已分配块
                    self.allocated_blocks.append((allocated_start, allocated_size))
                    
                    # 返回内存视图并reshape为正确形状
                    memory_block = self.pool_memory[allocated_start:allocated_start+allocated_size]
                    return memory_block.view(tensor.shape)
            
            raise MemoryError("内存池中没有足够的连续空间")
    def alloc(self, size_mb: int):
        """分配指定MB大小的连续内存块"""
        if size_mb <= 0:
            raise ValueError("分配大小必须为正数")
            
        # 计算需要的元素数量
        required_bytes = size_mb * 1024 * 1024
        required_elements = (required_bytes + self.element_size - 1 ) // self.element_size
        required_elements = int(required_elements)
        print(required_bytes, self.element_size,required_elements)
        with self.lock:
            # 寻找合适的空闲块（首次适应算法）
            for i, (start, size) in enumerate(self.free_list):
                if size >= required_elements:
                    # 分配这块内存
                    allocated_start = start
                    allocated_size = required_elements
                    
                    # 更新空闲列表
                    if size == required_elements:
                        # 完全匹配，移除这个空闲块
                        self.free_list.pop(i)
                    else:
                        # 部分使用，更新空闲块起始位置和大小
                        self.free_list[i] = (start + required_elements, size - required_elements)
                    
                    # 记录已分配块
                    self.allocated_blocks.append((allocated_start, allocated_size))
                    
                    # print(allocated_start, allocated_size)
                    # 返回内存视图（不拷贝数据）
                    return self.pool_memory[allocated_start:allocated_start+allocated_size]
            
            raise MemoryError("内存池中没有足够的连续空间")
    
    def free(self, memory_block):
        """释放之前分配的内存块"""
        with self.lock:
            # 找到这个内存块对应的分配记录
            block_start = None
            block_size = None
            
            for i, (start, size) in enumerate(self.allocated_blocks):
                if (self.pool_memory[start:start+size].data_ptr() == 
                    memory_block.data_ptr()):
                    block_start = start
                    block_size = size
                    self.allocated_blocks.pop(i)
                    break
            
            if block_start is None:
                # if for test
                return
                raise ValueError("尝试释放未分配的内存块")
            
            # 将释放的块加入空闲列表并合并相邻空闲块
            self.free_list.append((block_start, block_size))
            self.free_list.sort(key=lambda x: x[0])  # 按起始地址排序
            
            # 合并相邻空闲块
            merged_list = []
            current_start, current_size = self.free_list[0]
            
            for i in range(1, len(self.free_list)):
                start, size = self.free_list[i]
                if current_start + current_size == start:
                    # 相邻块，合并
                    current_size += size
                else:
                    # 不相邻，添加当前块并开始新的块
                    merged_list.append((current_start, current_size))
                    current_start, current_size = start, size
            
            merged_list.append((current_start, current_size))
            self.free_list = merged_list
    
    def copy_func(self, src, dst,non_blocking=False):
        """从src拷贝数据到dst"""
        if src.numel() != dst.numel():
            raise ValueError("源和目标内存块大小不匹配")
        dst = self.reshape(dst, src.shape)
        with torch.no_grad():
            dst.copy_(src, non_blocking=non_blocking)
        return dst
    def reshape(self, memory_block, new_shape):
        """将内存块reshape为新形状，不进行内存拷贝"""
        # 计算新形状需要的总元素数
        new_size = 1
        for dim in new_shape:
            new_size *= dim
        
        # 检查新形状是否与原始内存块大小兼容
        if new_size != memory_block.numel():
            raise ValueError(f"新形状 {new_shape} 需要 {new_size} 个元素，但内存块有 {memory_block.numel()} 个元素")
        
        # 返回reshape后的视图（不拷贝数据）
        return memory_block.view(new_shape)
    
    def get_usage_info(self):
        """获取内存池使用情况信息"""
        with self.lock:
            total_elements = self.total_elements
            allocated_elements = sum(size for _, size in self.allocated_blocks)
            free_elements = sum(size for _, size in self.free_list)
            
            return {
                "total_mb": self.pool_size//(1024*1024*1024),
                "allocated_mb": (allocated_elements * self.element_size) / (1024 * 1024),
                "free_mb": (free_elements * self.element_size) / (1024 * 1024),
                "fragmentation": len(self.free_list)
            }
        