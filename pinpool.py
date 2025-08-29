import torch
import threading
import collections
from typing import Dict, List, Optional, Tuple, Union, Deque

import threading
import torch

class PinnedMemoryPool:
    def __init__(self,
                 dtype: torch.dtype = torch.bfloat16,
                 pool_size: int = 1024):
        self.dtype = dtype
        self.pool_size = pool_size  # MB
        
        # 计算总字节数和元素数量
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        total_bytes = pool_size * 1024 * 1024
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
        KB_size = 1024
        size_kb = (tensor.numel() * tensor.element_size() + KB_size - 1) // KB_size
        size_kb = int(size_kb)
        # print(tensor.element_size(), tensor.numel(), size_kb)
        block = self.alloc_kb(size_kb)
        return self.reshape(block, tensor.shape)
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
                "total_mb": self.pool_size,
                "allocated_mb": (allocated_elements * self.element_size) / (1024 * 1024),
                "free_mb": (free_elements * self.element_size) / (1024 * 1024),
                "fragmentation": len(self.free_list)
            }
        
        

class FixedSizePinnedMemoryPool:
    """
    固定大小双类型固定内存池
    只分配两种预定义大小的固定内存块，所有块使用相同的数据类型
    """
    
    def __init__(self, 
                 small_block_shape: Tuple[int, ...], 
                 large_block_shape: Tuple[int, ...],
                 dtype: torch.dtype = torch.float32,
                 small_pool_size: int = 100,
                 large_pool_size: int = 50):
        """
        初始化固定大小内存池
        
        Args:
            small_block_shape: 小块Tensor的形状
            large_block_shape: 大块Tensor的形状
            dtype: 所有块的数据类型
            small_pool_size: 小块池的大小（块数量）
            large_pool_size: 大块池的大小（块数量）
        """
        self.small_block_shape = small_block_shape
        self.large_block_shape = large_block_shape
        self.dtype = dtype
        
        # 计算块大小（元素数量）
        self.small_block_size = 1
        for dim in small_block_shape:
            self.small_block_size *= dim
            
        self.large_block_size = 1
        for dim in large_block_shape:
            self.large_block_size *= dim
        
        # 计算字节大小
        element_size = torch.tensor([], dtype=dtype).element_size()
        self.small_block_bytes = self.small_block_size * element_size
        self.large_block_bytes = self.large_block_size * element_size
        
        # 锁用于线程安全
        self.lock = threading.RLock()
        
        # 预分配内存块
        self.small_blocks: Deque[torch.Tensor] = collections.deque()
        self.large_blocks: Deque[torch.Tensor] = collections.deque()
        
        # 分配小块
        for _ in range(small_pool_size):
            block = torch.empty(self.small_block_shape, dtype=dtype, pin_memory=True)
            self.small_blocks.append(block)
        
        # 分配大块
        for _ in range(large_pool_size):
            block = torch.empty(self.large_block_shape, dtype=dtype, pin_memory=True)
            self.large_blocks.append(block)
        
        # 记录已分配块的使用情况
        self.allocated_small_blocks: Dict[int, torch.Tensor] = {}
        self.allocated_large_blocks: Dict[int, torch.Tensor] = {}
    
    def allocate(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        分配指定形状的固定内存Tensor
        
        Args:
            shape: Tensor的形状（必须与预定义的小块或大块形状匹配）
            device: 目标设备（默认为当前GPU设备）
            
        Returns:
            torch.Tensor: 分配的固定内存Tensor
            
        Raises:
            ValueError: 如果请求的形状与预定义形状不匹配
            RuntimeError: 如果没有可用内存块
        """
        
        # 检查请求的形状是否匹配预定义形状
        if shape == self.small_block_shape:
            block_type = 'small'
        elif shape == self.large_block_shape:
            block_type = 'large'
        else:
            raise ValueError(f"Requested shape {shape} does not match predefined shapes: "
                           f"small={self.small_block_shape}, large={self.large_block_shape}")
        
        with self.lock:
            if block_type == 'small':
                if not self.small_blocks:
                    raise RuntimeError("No available small blocks in the pool")
                
                # 获取一个小块
                tensor = self.small_blocks.popleft()
                self.allocated_small_blocks[id(tensor)] = tensor
            else:
                if not self.large_blocks:
                    raise RuntimeError("No available large blocks in the pool")
                
                # 获取一个大块
                tensor = self.large_blocks.popleft()
                self.allocated_large_blocks[id(tensor)] = tensor
            
            # 移动到目标设备
            return tensor
    
    def free(self, tensor: torch.Tensor) -> None:
        """
        释放Tensor回内存池
        
        Args:
            tensor: 要释放的Tensor
        """
        tensor_id = id(tensor)
        
        with self.lock:
            # 检查是小块还是大块
            if tensor_id in self.allocated_small_blocks:
                # 是小块，放回小块池
                self.small_blocks.append(self.allocated_small_blocks[tensor_id])
                del self.allocated_small_blocks[tensor_id]
            elif tensor_id in self.allocated_large_blocks:
                # 是大块，放回大块池
                self.large_blocks.append(self.allocated_large_blocks[tensor_id])
                del self.allocated_large_blocks[tensor_id]
            else:
                raise ValueError("Tensor was not allocated by this memory pool")
    
    def clear(self) -> None:
        """清空内存池中的所有分配"""
        with self.lock:
            # 将所有已分配块放回池中
            for tensor in list(self.allocated_small_blocks.values()):
                self.small_blocks.append(tensor)
            for tensor in list(self.allocated_large_blocks.values()):
                self.large_blocks.append(tensor)
            
            self.allocated_small_blocks.clear()
            self.allocated_large_blocks.clear()
    
    def resize(self, 
               small_pool_size: Optional[int] = None, 
               large_pool_size: Optional[int] = None) -> None:
        """
        调整内存池大小
        
        Args:
            small_pool_size: 新的小块池大小（如果为None则不改变）
            large_pool_size: 新的大块池大小（如果为None则不改变）
        """
        with self.lock:
            # 调整小块池大小
            if small_pool_size is not None:
                current_size = len(self.small_blocks) + len(self.allocated_small_blocks)
                
                if small_pool_size > current_size:
                    # 需要增加小块
                    for _ in range(small_pool_size - current_size):
                        block = torch.empty(self.small_block_shape, dtype=self.dtype, pin_memory=True)
                        self.small_blocks.append(block)
                elif small_pool_size < current_size:
                    # 需要减少小块
                    # 首先尝试从空闲块中移除
                    while len(self.small_blocks) > 0 and current_size > small_pool_size:
                        self.small_blocks.popleft()
                        current_size -= 1
                    
                    # 如果还需要更多减少，需要等待已分配块被释放
                    # 这里我们只是记录目标大小，实际减少会在块被释放时进行
                    self._target_small_pool_size = small_pool_size
            
            # 调整大块池大小
            if large_pool_size is not None:
                current_size = len(self.large_blocks) + len(self.allocated_large_blocks)
                
                if large_pool_size > current_size:
                    # 需要增加大块
                    for _ in range(large_pool_size - current_size):
                        block = torch.empty(self.large_block_shape, dtype=self.dtype, pin_memory=True)
                        self.large_blocks.append(block)
                elif large_pool_size < current_size:
                    # 需要减少大块
                    # 首先尝试从空闲块中移除
                    while len(self.large_blocks) > 0 and current_size > large_pool_size:
                        self.large_blocks.popleft()
                        current_size -= 1
                    
                    # 如果还需要更多减少，需要等待已分配块被释放
                    self._target_large_pool_size = large_pool_size
    
    def stats(self) -> Dict[str, int]:
        """
        获取内存池统计信息
        
        Returns:
            包含统计信息的字典
        """
        with self.lock:
            return {
                'small_block_shape': self.small_block_shape,
                'large_block_shape': self.large_block_shape,
                'dtype': str(self.dtype),
                'small_blocks_available': len(self.small_blocks),
                'small_blocks_allocated': len(self.allocated_small_blocks),
                'large_blocks_available': len(self.large_blocks),
                'large_blocks_allocated': len(self.allocated_large_blocks),
                'total_blocks_available': len(self.small_blocks) + len(self.large_blocks),
                'total_blocks_allocated': len(self.allocated_small_blocks) + len(self.allocated_large_blocks),
            }
def calculate_size_in_mb(shape: Tuple[int, ...], dtype: torch.dtype) -> float:
    """计算给定形状和数据类型的Tensor所需的内存大小（MB）"""
    element_size = torch.tensor([], dtype=dtype).element_size()
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    total_bytes = num_elements * element_size
    return total_bytes / (1024 * 1024)
import time
# 使用示例
def main_fixed():
    # 创建内存池 - 定义两种大小的块
    small_shape = (2048, 2048)  # 用于小图像
    large_shape = (8, 10, 512, 2048)  # 用于大图像
    dtype = torch.bfloat16
    
    pool = FixedSizePinnedMemoryPool(
        small_block_shape=small_shape,
        large_block_shape=large_shape,
        dtype=dtype,
        small_pool_size=3,
        large_pool_size=16
    )
    
    print("Initial pool stats:")
    print(pool.stats())
    
    # 分配和使用小块
    small_tensors = []
    for i in range(3):
        tensor = pool.allocate(small_shape)
        
        # 使用copy_从GPU Tensor拷贝数据
        if torch.cuda.is_available():
            gpu_tensor = torch.randn(small_shape, dtype=torch.bfloat16, device='cuda:1')
            time_start = time.time()
            tensor.copy_(gpu_tensor, non_blocking=False)
            print(f"Copy time for small tensor {i}: {time.time() - time_start:.6f} seconds")
            print(f"Allocated small tensor {i}: shape {tensor.shape}")
        
        small_tensors.append(tensor)
    
    # 分配和使用大块
    large_tensors = []
    for i in range(4):
        tensor = pool.allocate(large_shape)
        
        # 使用copy_从GPU Tensor拷贝数据
        if torch.cuda.is_available():
            gpu_tensor = torch.randn(large_shape, dtype=torch.bfloat16,device='cuda:1')
            time_start = time.time()
            tensor.copy_(gpu_tensor, non_blocking=False)
            print(f"Copy time for large tensor {i}: {time.time() - time_start:.6f} seconds")
            print(f"Allocated large tensor {i}: shape {tensor.shape}, sum {tensor.sum().item()}")
        
        large_tensors.append(tensor)
    
    print("After allocation:")
    print(pool.stats())
    
    # 释放一些Tensor
    for i, tensor in enumerate(small_tensors):
        if i % 2 == 0:  # 释放一半的小块
            pool.free(tensor)
            print(f"Freed small tensor {i}")
    
    for i, tensor in enumerate(large_tensors):
        if i % 2 == 0:  # 释放一半的大块
            pool.free(tensor)
            print(f"Freed large tensor {i}")
    
    print("After freeing some tensors:")
    print(pool.stats())
    
    # 调整池大小
    pool.resize(small_pool_size=15, large_pool_size=8)
    print("After resizing:")
    print(pool.stats())
    
    # 清空内存池
    pool.clear()
    print("After clearing:")
    print(pool.stats())
    
def test_pin():
    pool = PinnedMemoryPool(
        dtype=torch.bfloat16,
        pool_size=4*1024  # 10MB
    )
    print("Initial usage:", pool.get_usage_info())
    
    key_states = torch.randn((8, 512, 2048), dtype=torch.bfloat16)
    
    # 一致的 key_states_size = calculate_size_in_mb(key_states.shape, key_states.dtype)
    key_states_size = key_states.numel()*key_states.element_size()/(1024*1024)
    # print(key_states_size)
    print(pool.get_usage_info())
    block1 = pool.alloc_tensor(key_states)  # 1MB
    print("After alloc 1:", pool.get_usage_info())
    print(block1.shape)
    
    time_start=time.time()
    block1.copy_(key_states, non_blocking=False)
    print("Copy time:", time.time()-time_start)
    print("After copy 1:", pool.get_usage_info())
    pool.free(block1)
    
if __name__ == "__main__":
    test_pin()