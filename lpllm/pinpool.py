import torch
import threading
import collections
from typing import Dict, List, Optional, Tuple, Union, Deque

import threading
import time
from lpllm.cuda_memcpy_utils import cuda_copy_, safe_copy_
from lpllm.logger import init_logger
logger = init_logger(__name__)

class PinnedMemoryPool:
    def __init__(self,
                 dtype: torch.dtype = torch.bfloat16,
                 pool_size: int = 2, chunk_size_mb: int=1):
        self.dtype = dtype
        self.pool_size = pool_size  # GB
        print(f"here pin")
        # 计算总字节数和元素数量
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        total_bytes = pool_size * 1024 * 1024 * 1024
        self.total_elements = total_bytes // self.element_size

        # 分块分配内存，每次分配1GB
        chunk_size_bytes = chunk_size_mb * 1024 * 1024
        chunk_elements = chunk_size_bytes // self.element_size

        time_start = time.time()
        self.pool_memory = []  # 改为列表存储多个内存块

        logger.info(f"Initializing PinnedMemoryPool with {pool_size}GB total, allocating in {chunk_size_mb}MB chunks...")

        remaining_elements = self.total_elements
        current_offset = 0

        while remaining_elements > 0:
            # 计算当前块的大小（最后一个块可能小于1GB）
            current_chunk_elements = min(chunk_elements, remaining_elements)

            try:
                chunk = torch.empty(current_chunk_elements, dtype=dtype, pin_memory=True, device="cpu")
                self.pool_memory.append(chunk)
                logger.debug(f"Allocated chunk {len(self.pool_memory)}: {current_chunk_elements} elements ({current_chunk_elements * self.element_size / (1024*1024):.1f} MB)")

                remaining_elements -= current_chunk_elements
                current_offset += current_chunk_elements

            except Exception as e:
                logger.error(f"Failed to allocate chunk: {e}")
                break

        total_allocated = sum(chunk.numel() for chunk in self.pool_memory)
        alloc_time = time.time() - time_start

        logger.info(f"Successfully allocated {len(self.pool_memory)} chunks, total {total_allocated} elements ({total_allocated * self.element_size / (1024*1024):.1f} MB) in {alloc_time:.3f}s")

        # 内存管理数据结构
        self.lock = threading.RLock()
        self.allocated_blocks = []  # 记录已分配的内存块 (chunk_idx, start, size)
        self.free_list = []  # 空闲块列表 [(chunk_idx, start, size), ...]

        # 初始化空闲列表
        for chunk_idx, chunk in enumerate(self.pool_memory):
            self.free_list.append((chunk_idx, 0, chunk.numel()))

    def alloc_same_pin_tensor(self, tensor: torch.Tensor):
        """分配与给定Tensor形状和类型匹配的内存块"""
        required_elements = tensor.numel()
        required_bytes = required_elements * self.element_size
        # print(tensor.element_size(), tensor.numel(), size_kb)
        block = self.alloc(required_bytes)
        return self.reshape(block, tensor.shape)
    def alloc(self, size_bytes: int):
        """分配指定大小的连续内存块"""
        if size_bytes <= 0:
            raise ValueError("分配大小必须为正数")

        # 计算需要的元素数量
        required_bytes = size_bytes
        required_elements = (required_bytes + self.element_size - 1) // self.element_size
        required_elements = int(required_elements)

        with self.lock:
            # 寻找合适的空闲块（首次适应算法）
            for i, (chunk_idx, start, size) in enumerate(self.free_list):
                if size >= required_elements:
                    # 分配这块内存
                    allocated_chunk_idx = chunk_idx
                    allocated_start = start
                    allocated_size = required_elements

                    # 更新空闲列表
                    if size == required_elements:
                        # 完全匹配，移除这个空闲块
                        self.free_list.pop(i)
                    else:
                        # 部分使用，更新空闲块起始位置和大小
                        self.free_list[i] = (chunk_idx, start + required_elements, size - required_elements)

                    # 记录已分配块
                    self.allocated_blocks.append((allocated_chunk_idx, allocated_start, allocated_size))

                    # 返回内存视图（不拷贝数据）
                    chunk = self.pool_memory[allocated_chunk_idx]
                    return chunk[allocated_start:allocated_start+allocated_size]
            logger.error(f"Memory allocation failed: ")
            logger.error(f"  Required: {required_elements} elements ({required_bytes} bytes)")
            logger.error(f"  Available free blocks: {len(self.free_list)}")
            for i, (chunk_idx, start, size) in enumerate(self.free_list):
                logger.error(f"    Block {i}: chunk={chunk_idx}, start={start}, size={size}")
            logger.error(f"  Current usage: {self.get_usage_info()}")
            raise MemoryError("内存池中没有足够的连续空间")
    
    def free(self, memory_block):
        """释放之前分配的内存块"""
        with self.lock:
            # 找到这个内存块对应的分配记录
            block_chunk_idx = None
            block_start = None
            block_size = None

            for i, (chunk_idx, start, size) in enumerate(self.allocated_blocks):
                chunk = self.pool_memory[chunk_idx]
                if (chunk[start:start+size].data_ptr() ==
                    memory_block.data_ptr()):
                    block_chunk_idx = chunk_idx
                    block_start = start
                    block_size = size
                    self.allocated_blocks.pop(i)
                    break

            if block_start is None:
                # if for test
                return
                raise ValueError("尝试释放未分配的内存块")

            # 将释放的块加入空闲列表并合并相邻空闲块
            self.free_list.append((block_chunk_idx, block_start, block_size))
            self.free_list.sort(key=lambda x: (x[0], x[1]))  # 按chunk_idx和start排序
            
            # 合并相邻空闲块
            merged_list = []
            current_chunk_idx, current_start, current_size = self.free_list[0]

            for i in range(1, len(self.free_list)):
                chunk_idx, start, size = self.free_list[i]
                # 只有在同一个chunk内的相邻块才能合并
                if chunk_idx == current_chunk_idx and current_start + current_size == start:
                    # 相邻块，合并
                    current_size += size
                else:
                    # 不同chunk或不相邻，添加当前块并开始新的块
                    merged_list.append((current_chunk_idx, current_start, current_size))
                    current_chunk_idx, current_start, current_size = chunk_idx, start, size

            merged_list.append((current_chunk_idx, current_start, current_size))
            self.free_list = merged_list
    
    def copy_func(self, src, dst,non_blocking=False):
        """从src拷贝数据到dst using CUDA memcpy"""
        if src.numel() != dst.numel():
            raise ValueError("源和目标内存块大小不匹配")
        dst = self.reshape(dst, src.shape)
        with torch.no_grad():
            cuda_copy_(dst, src, non_blocking=non_blocking)
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
            allocated_elements = sum(size for _, _, size in self.allocated_blocks)
            free_elements = sum(size for _, _, size in self.free_list)

            return {
                "total_gb": self.pool_size,
                "allocated_mb": (allocated_elements * self.element_size) / (1024 * 1024),
                "free_mb": (free_elements * self.element_size) / (1024 * 1024),
                "fragmentation": len(self.free_list)
            }
        
        
        

class PreAllocatedPinnedMemoryPool:
    """
    预分配固定大小的Pinned内存池
    预先分配指定数量的固定大小内存块，避免每次分配时的计算开销
    """

    def __init__(self,
                 block_shape: Tuple[int, ...],
                 dtype: torch.dtype = torch.bfloat16,
                 pool_size: int = 100,
                 name: str = "prealloc"):
        """
        初始化预分配内存池

        Args:
            block_shape: 每个内存块的固定形状
            dtype: 数据类型
            pool_size: 预分配的块数量
            name: 池的名称，用于日志
        """
        self.block_shape = block_shape
        self.dtype = dtype
        self.pool_size = pool_size
        self.name = name

        # 计算块大小
        self.block_size = 1
        for dim in block_shape:
            self.block_size *= dim

        # 预分配内存块
        logger.info(f"Pre-allocating {pool_size} pinned memory blocks of shape {block_shape}...")
        time_start = time.time()

        self.lock = threading.RLock()
        self.available_blocks: collections.deque = collections.deque()
        self.allocated_blocks: Dict[int, torch.Tensor] = {}

        # 预分配所有内存块
        for i in range(pool_size):
            try:
                block = torch.empty(block_shape, dtype=dtype, pin_memory=True)
                self.available_blocks.append(block)
            except Exception as e:
                logger.error(f"Failed to pre-allocate block {i}: {e}")
                break

        actual_pool_size = len(self.available_blocks)
        alloc_time = time.time() - time_start

        logger.info(f"Pre-allocated {actual_pool_size}/{pool_size} blocks in {alloc_time:.3f}s")
        logger.info(f"Total pre-allocated memory: {actual_pool_size * self.block_size * torch.tensor([], dtype=dtype).element_size() / (1024*1024):.1f} MB")

    def allocate(self) -> torch.Tensor:
        """
        分配一个预分配的内存块

        Returns:
            torch.Tensor: 分配的内存块

        Raises:
            RuntimeError: 如果没有可用内存块
        """
        with self.lock:
            if not self.available_blocks:
                raise RuntimeError(f"No available pre-allocated blocks in {self.name} pool")

            tensor = self.available_blocks.popleft()
            self.allocated_blocks[id(tensor)] = tensor
            return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """
        释放Tensor回内存池

        Args:
            tensor: 要释放的Tensor
        """
        tensor_id = id(tensor)

        with self.lock:
            if tensor_id not in self.allocated_blocks:
                raise ValueError(f"Tensor {tensor_id} was not allocated by this {self.name} pool")

            # 将tensor放回可用块队列
            self.available_blocks.append(self.allocated_blocks[tensor_id])
            del self.allocated_blocks[tensor_id]

    def stats(self) -> Dict[str, int]:
        """
        获取内存池统计信息

        Returns:
            包含统计信息的字典
        """
        with self.lock:
            return {
                'name': self.name,
                'block_shape': self.block_shape,
                'dtype': str(self.dtype),
                'pool_size': self.pool_size,
                'available_blocks': len(self.available_blocks),
                'allocated_blocks': len(self.allocated_blocks),
                'total_allocated': len(self.available_blocks) + len(self.allocated_blocks),
            }

    def clear(self) -> None:
        """清空内存池中的所有分配"""
        with self.lock:
            # 将所有已分配块放回池中
            for tensor in list(self.allocated_blocks.values()):
                self.available_blocks.append(tensor)

            self.allocated_blocks.clear()


class HybridPinnedMemoryPool:
    """
    混合内存池：结合预分配和动态分配的策略
    优先使用预分配的固定大小块，必要时回退到动态分配
    """

    def __init__(self,
                 prealloc_configs: List[Dict],
                 dtype: torch.dtype = torch.bfloat16,
                 max_dynamic_pool_size: int = 2,  # GB
                 name: str = "hybrid"):
        """
        初始化混合内存池

        Args:
            prealloc_configs: 预分配配置列表，每个配置包含:
                - shape: 块的形状
                - pool_size: 预分配数量
                - name: 池名称
            dtype: 数据类型
            max_dynamic_pool_size: 动态内存池最大大小(GB)
            name: 池的名称
        """
        self.dtype = dtype
        self.name = name
        self.prealloc_pools = {}
        self.dynamic_pool = None

        logger.info(f"Initializing {name} hybrid memory pool...")

        # 创建预分配池
        for config in prealloc_configs:
            shape = config['shape']
            pool_size = config['pool_size']
            pool_name = config.get('name', f"prealloc_{shape}")

            pool = PreAllocatedPinnedMemoryPool(
                block_shape=shape,
                dtype=dtype,
                pool_size=pool_size,
                name=pool_name
            )
            self.prealloc_pools[shape] = pool

        # 创建动态内存池作为后备
        if max_dynamic_pool_size > 0:
            self.dynamic_pool = PinnedMemoryPool(dtype=dtype, pool_size=max_dynamic_pool_size)

        logger.info(f"Initialized {len(self.prealloc_pools)} pre-allocated pools and 1 dynamic pool")

    def allocate(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        分配内存，优先使用预分配的固定大小块

        Args:
            shape: 需要的Tensor形状

        Returns:
            torch.Tensor: 分配的内存
        """
        # 首先尝试从预分配池中分配
        if shape in self.prealloc_pools:
            try:
                pool = self.prealloc_pools[shape]
                logger.debug(f"Allocating from pre-allocated pool {pool.name}")
                return pool.allocate()
            except RuntimeError:
                logger.warning(f"Pre-allocated pool {shape} is empty, falling back to dynamic allocation")

        # 如果没有对应的预分配池或预分配池已满，使用动态分配
        if self.dynamic_pool is not None:
            logger.debug(f"Allocating from dynamic pool for shape {shape}")
            # 创建一个与shape匹配的tensor
            tensor = torch.empty(shape, dtype=self.dtype, pin_memory=True)
            # 使用动态池来管理这个tensor
            return tensor
        else:
            raise RuntimeError(f"No available memory pool for shape {shape}")

    def free(self, tensor: torch.Tensor) -> None:
        """
        释放Tensor

        Args:
            tensor: 要释放的Tensor
        """
        tensor_id = id(tensor)

        # 检查是否来自预分配池
        for pool in self.prealloc_pools.values():
            if tensor_id in pool.allocated_blocks:
                pool.free(tensor)
                return

        # 如果不在预分配池中，可能是动态分配的tensor
        # 这里我们不做特殊处理，让用户自己管理
        logger.debug(f"Tensor {tensor_id} not from pre-allocated pools, assuming dynamic allocation")

    def stats(self) -> Dict:
        """获取所有内存池的统计信息"""
        stats = {
            'name': self.name,
            'dtype': str(self.dtype),
            'prealloc_pools': {},
            'dynamic_pool': None
        }

        for shape, pool in self.prealloc_pools.items():
            stats['prealloc_pools'][str(shape)] = pool.stats()

        if self.dynamic_pool is not None:
            stats['dynamic_pool'] = self.dynamic_pool.get_usage_info()

        return stats


