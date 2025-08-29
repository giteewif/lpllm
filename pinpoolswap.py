import ctypes
import threading
import numpy as np
import os
import tempfile
import pickle
import time
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from collections import OrderedDict

class MemoryState(Enum):
    FREE = 0
    ALLOCATED = 1
    SWAPPED = 2

@dataclass
class MemoryChunk:
    """内存块数据结构"""
    start_address: int
    size: int  # 以MB为单位
    state: MemoryState
    data: Optional[ctypes.Array] = None
    swap_file: Optional[str] = None
    last_access_time: float = 0
    access_count: int = 0

class PinMemoryPool:
    """
    Pin Memory Pool 类用于分配和管理固定大小的连续固定内存。
    支持当内存不足时将内存交换到硬盘。
    """
    
    def __init__(self, total_mb: int = 1024, chunk_size_mb: int = 1, 
                 swap_dir: Optional[str] = None, max_swap_mb: int = 4096):
        """
        初始化 Pin Memory Pool
        
        参数:
            total_mb: 内存池总大小（MB），默认为 1024MB (1GB)
            chunk_size_mb: 每个内存块的大小（MB），默认为 1MB
            swap_dir: 交换文件目录，如果为None则使用系统临时目录
            max_swap_mb: 最大交换空间（MB），默认为 4096MB (4GB)
        """
        self.total_mb = total_mb
        self.chunk_size_mb = chunk_size_mb
        self.max_swap_mb = max_swap_mb
        self._lock = threading.RLock()
        
        # 计算实际字节大小
        self.byte_size = total_mb * 1024 * 1024
        self.chunk_byte_size = chunk_size_mb * 1024 * 1024
        self.num_chunks = total_mb // chunk_size_mb
        
        # 分配固定内存
        self._memory = (ctypes.c_byte * self.byte_size)()
        self._base_address = ctypes.addressof(self._memory)
        
        # 初始化内存块列表
        self._chunks: List[MemoryChunk] = []
        for i in range(self.num_chunks):
            chunk_address = self._base_address + i * self.chunk_byte_size
            chunk = MemoryChunk(
                start_address=chunk_address,
                size=chunk_size_mb,
                state=MemoryState.FREE
            )
            self._chunks.append(chunk)
        
        # 空闲块索引列表
        self._free_indices = list(range(self.num_chunks))
        
        # 已分配块映射 (地址 -> 块索引)
        self._allocated_map: Dict[int, int] = {}
        
        # 交换文件目录
        self.swap_dir = swap_dir or tempfile.gettempdir()
        self._swap_files = OrderedDict()  # 最近最少使用的交换文件
        self._current_swap_size = 0  # 当前交换文件总大小（MB）
        
        # 统计信息
        self._swap_operations = 0
        self._total_swap_time = 0
    
    def allocate(self, size_mb: int = 1) -> Optional[ctypes.Array]:
        """
        从内存池中分配指定大小的连续内存
        
        参数:
            size_mb: 需要分配的内存大小（MB），必须是 chunk_size_mb 的整数倍
            
        返回:
            分配的内存块，如果分配失败则返回 None
        """
        if size_mb <= 0 or size_mb % self.chunk_size_mb != 0:
            return None
            
        num_chunks_needed = size_mb // self.chunk_size_mb
        
        with self._lock:
            # 如果内存不足，尝试交换出一些内存
            if len(self._free_indices) < num_chunks_needed:
                if not self._swap_out_memory(num_chunks_needed):
                    return None
            
            # 查找连续的空闲块
            start_index = self._find_contiguous_free_blocks(num_chunks_needed)
            if start_index is None:
                return None
            
            # 标记这些块为已分配
            for i in range(start_index, start_index + num_chunks_needed):
                self._chunks[i].state = MemoryState.ALLOCATED
                self._chunks[i].last_access_time = time.time()
                self._chunks[i].access_count = 1
                if i in self._free_indices:
                    self._free_indices.remove(i)
            
            # 创建连续内存视图
            start_address = self._chunks[start_index].start_address
            total_size = num_chunks_needed * self.chunk_byte_size
            memory_view = ctypes.cast(start_address, ctypes.POINTER(ctypes.c_byte * total_size)).contents
            
            # 记录分配信息
            self._allocated_map[start_address] = num_chunks_needed
            
            return memory_view
    
    def _swap_out_memory(self, num_chunks_needed: int) -> bool:
        """
        将内存交换到硬盘，释放指定数量的内存块
        
        参数:
            num_chunks_needed: 需要释放的内存块数量
            
        返回:
            成功释放返回 True，否则返回 False
        """
        # 计算需要交换出的内存块数量
        chunks_to_swap = num_chunks_needed - len(self._free_indices)
        
        # 获取最近最少使用的已分配内存块
        allocated_indices = []
        for i in range(self.num_chunks):
            if self._chunks[i].state == MemoryState.ALLOCATED:
                allocated_indices.append(i)
        
        # 按最后访问时间排序（最近最少使用优先）
        allocated_indices.sort(key=lambda i: self._chunks[i].last_access_time)
        
        # 检查交换空间是否足够
        swap_space_needed = chunks_to_swap * self.chunk_size_mb
        if self._current_swap_size + swap_space_needed > self.max_swap_mb:
            return False
        
        # 交换出内存
        swapped_count = 0
        for idx in allocated_indices:
            if swapped_count >= chunks_to_swap:
                break
                
            if self._swap_chunk_to_disk(idx):
                swapped_count += 1
        
        return swapped_count >= chunks_to_swap
    
    def _swap_chunk_to_disk(self, chunk_index: int) -> bool:
        """
        将指定内存块交换到硬盘
        
        参数:
            chunk_index: 内存块索引
            
        返回:
            成功交换返回 True，否则返回 False
        """
        chunk = self._chunks[chunk_index]
        
        try:
            # 创建交换文件
            swap_file = os.path.join(self.swap_dir, f"pin_memory_swap_{chunk.start_address}_{int(time.time())}.dat")
            
            # 将内存数据写入文件
            start_time = time.time()
            with open(swap_file, 'wb') as f:
                # 将内存数据转换为字节数组并写入文件
                data_bytes = bytearray(chunk.data)
                f.write(data_bytes)
            
            # 更新统计信息
            swap_time = time.time() - start_time
            self._swap_operations += 1
            self._total_swap_time += swap_time
            
            # 更新块状态
            chunk.state = MemoryState.SWAPPED
            chunk.swap_file = swap_file
            chunk.data = None
            
            # 添加到空闲列表
            self._free_indices.append(chunk_index)
            
            # 更新交换文件列表
            self._swap_files[chunk.start_address] = {
                'file': swap_file,
                'size': chunk.size,
                'access_time': chunk.last_access_time
            }
            self._current_swap_size += chunk.size
            
            return True
        except Exception as e:
            print(f"交换内存到硬盘失败: {e}")
            return False
    
    def _swap_chunk_to_memory(self, chunk_index: int) -> bool:
        """
        将指定内存块从硬盘交换回内存
        
        参数:
            chunk_index: 内存块索引
            
        返回:
            成功交换返回 True，否则返回 False
        """
        chunk = self._chunks[chunk_index]
        
        if chunk.state != MemoryState.SWAPPED or not chunk.swap_file:
            return False
        
        try:
            # 从文件读取数据
            start_time = time.time()
            with open(chunk.swap_file, 'rb') as f:
                data_bytes = f.read()
            
            # 将数据写回内存
            memory_view = ctypes.cast(chunk.start_address, ctypes.POINTER(ctypes.c_byte * len(data_bytes))).contents
            for i, byte in enumerate(data_bytes):
                memory_view[i] = byte
            
            # 更新统计信息
            swap_time = time.time() - start_time
            self._swap_operations += 1
            self._total_swap_time += swap_time
            
            # 更新块状态
            chunk.state = MemoryState.ALLOCATED
            chunk.data = memory_view
            chunk.last_access_time = time.time()
            chunk.access_count += 1
            
            # 从空闲列表中移除
            if chunk_index in self._free_indices:
                self._free_indices.remove(chunk_index)
            
            # 更新交换文件列表
            if chunk.start_address in self._swap_files:
                self._current_swap_size -= self._swap_files[chunk.start_address]['size']
                del self._swap_files[chunk.start_address]
            
            # 删除交换文件
            try:
                os.remove(chunk.swap_file)
            except:
                pass
            
            chunk.swap_file = None
            
            return True
        except Exception as e:
            print(f"从硬盘交换内存回内存失败: {e}")
            return False
    
    def _find_contiguous_free_blocks(self, num_blocks: int) -> Optional[int]:
        """
        查找连续的空闲块
        
        参数:
            num_blocks: 需要的连续块数量
            
        返回:
            起始块索引，如果找不到返回 None
        """
        # 先对空闲索引排序
        sorted_indices = sorted(self._free_indices)
        
        # 查找连续块
        for i in range(len(sorted_indices) - num_blocks + 1):
            # 检查是否连续
            is_contiguous = True
            for j in range(1, num_blocks):
                if sorted_indices[i + j] != sorted_indices[i] + j:
                    is_contiguous = False
                    break
            
            if is_contiguous:
                return sorted_indices[i]
        
        return None
    
    def free(self, memory: ctypes.Array) -> bool:
        """
        释放内存回内存池
        
        参数:
            memory: 要释放的内存
            
        返回:
            成功释放返回 True，否则返回 False
        """
        # 获取内存地址
        address = ctypes.addressof(memory)
        
        with self._lock:
            if address not in self._allocated_map:
                return False
            
            # 获取分配的块数
            num_chunks = self._allocated_map[address]
            
            # 计算起始块索引
            start_index = (address - self._base_address) // self.chunk_byte_size
            
            # 标记这些块为空闲
            for i in range(start_index, start_index + num_chunks):
                # 如果块已被交换到硬盘，删除交换文件
                if self._chunks[i].state == MemoryState.SWAPPED and self._chunks[i].swap_file:
                    try:
                        os.remove(self._chunks[i].swap_file)
                    except:
                        pass
                    
                    # 更新交换文件列表
                    if self._chunks[i].start_address in self._swap_files:
                        self._current_swap_size -= self._swap_files[self._chunks[i].start_address]['size']
                        del self._swap_files[self._chunks[i].start_address]
                
                self._chunks[i].state = MemoryState.FREE
                self._chunks[i].swap_file = None
                # 添加到空闲列表（保持有序）
                self._free_indices.append(i)
            
            # 从已分配映射中移除
            del self._allocated_map[address]
            
            # 对空闲索引排序，便于后续查找连续块
            self._free_indices.sort()
            
            return True
    
    def access_memory(self, memory: ctypes.Array) -> bool:
        """
        访问内存，如果内存已被交换到硬盘，则将其交换回内存
        
        参数:
            memory: 要访问的内存
            
        返回:
            成功访问返回 True，否则返回 False
        """
        # 获取内存地址
        address = ctypes.addressof(memory)
        
        with self._lock:
            if address not in self._allocated_map:
                return False
            
            # 获取分配的块数
            num_chunks = self._allocated_map[address]
            
            # 计算起始块索引
            start_index = (address - self._base_address) // self.chunk_byte_size
            
            # 检查是否有块被交换到硬盘
            needs_swap = False
            for i in range(start_index, start_index + num_chunks):
                if self._chunks[i].state == MemoryState.SWAPPED:
                    needs_swap = True
                    break
            
            # 如果有块被交换，将它们交换回内存
            if needs_swap:
                # 先确保有足够的空闲空间
                swapped_count = sum(1 for i in range(start_index, start_index + num_chunks) 
                                  if self._chunks[i].state == MemoryState.SWAPPED)
                
                if len(self._free_indices) < swapped_count:
                    if not self._swap_out_memory(swapped_count):
                        return False
                
                # 交换回内存
                for i in range(start_index, start_index + num_chunks):
                    if self._chunks[i].state == MemoryState.SWAPPED:
                        if not self._swap_chunk_to_memory(i):
                            return False
            
            # 更新访问时间
            current_time = time.time()
            for i in range(start_index, start_index + num_chunks):
                self._chunks[i].last_access_time = current_time
                self._chunks[i].access_count += 1
            
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取内存池统计信息
        
        返回:
            包含内存池统计信息的字典
        """
        with self._lock:
            total_allocated_mb = 0
            total_swapped_mb = 0
            
            for chunk in self._chunks:
                if chunk.state == MemoryState.ALLOCATED:
                    total_allocated_mb += chunk.size
                elif chunk.state == MemoryState.SWAPPED:
                    total_swapped_mb += chunk.size
            
            total_free_mb = self.total_mb - total_allocated_mb
            fragmentation = self._calculate_fragmentation()
            
            avg_swap_time = self._total_swap_time / self._swap_operations if self._swap_operations > 0 else 0
            
            return {
                "total_mb": self.total_mb,
                "chunk_size_mb": self.chunk_size_mb,
                "allocated_mb": total_allocated_mb,
                "swapped_mb": total_swapped_mb,
                "free_mb": total_free_mb,
                "fragmentation": fragmentation,
                "allocated_blocks": len(self._allocated_map),
                "free_blocks": len(self._free_indices),
                "contiguous_free_blocks": self._max_contiguous_free_blocks(),
                "swap_operations": self._swap_operations,
                "avg_swap_time_ms": avg_swap_time * 1000,
                "swap_space_used_mb": self._current_swap_size,
                "max_swap_space_mb": self.max_swap_mb
            }
    
    def _calculate_fragmentation(self) -> float:
        """计算内存碎片化程度"""
        if not self._free_indices:
            return 0.0
        
        # 计算空闲块之间的平均间隔
        sorted_indices = sorted(self._free_indices)
        gaps = []
        
        for i in range(1, len(sorted_indices)):
            gap = sorted_indices[i] - sorted_indices[i-1] - 1
            if gap > 0:
                gaps.append(gap)
        
        if not gaps:
            return 0.0
        
        # 碎片化程度 = 平均间隔 / 总块数
        avg_gap = sum(gaps) / len(gaps)
        return avg_gap / self.num_chunks
    
    def _max_contiguous_free_blocks(self) -> int:
        """计算最大连续空闲块数量"""
        if not self._free_indices:
            return 0
        
        sorted_indices = sorted(self._free_indices)
        max_contiguous = 1
        current_contiguous = 1
        
        for i in range(1, len(sorted_indices)):
            if sorted_indices[i] == sorted_indices[i-1] + 1:
                current_contiguous += 1
                max_contiguous = max(max_contiguous, current_contiguous)
            else:
                current_contiguous = 1
        
        return max_contiguous
    
    def release_all(self) -> None:
        """释放所有内存块"""
        with self._lock:
            # 删除所有交换文件
            for chunk in self._chunks:
                if chunk.state == MemoryState.SWAPPED and chunk.swap_file:
                    try:
                        os.remove(chunk.swap_file)
                    except:
                        pass
            
            # 重置所有块状态
            for chunk in self._chunks:
                chunk.state = MemoryState.FREE
                chunk.swap_file = None
            
            # 重置空闲索引
            self._free_indices = list(range(self.num_chunks))
            
            # 清空已分配映射
            self._allocated_map.clear()
            
            # 清空交换文件列表
            self._swap_files.clear()
            self._current_swap_size = 0
            
            # 重置统计信息
            self._swap_operations = 0
            self._total_swap_time = 0


# 示例使用
if __name__ == "__main__":
    # 创建内存池，总大小16MB，每个块1MB，最大交换空间64MB
    pool = PinMemoryPool(total_mb=16, chunk_size_mb=1, max_swap_mb=64)
    
    # 分配内存，超过物理内存限制
    allocated_mem = []
    for i in range(20):  # 尝试分配20MB，超过物理内存16MB
        mem = pool.allocate(1)
        if mem:
            allocated_mem.append(mem)
            print(f"成功分配 {i+1}MB 内存")
        else:
            print(f"分配 {i+1}MB 内存失败")
            break
    
    # 获取统计信息
    stats = pool.get_stats()
    print(f"内存池统计: 总大小={stats['total_mb']}MB, 已分配={stats['allocated_mb']}MB, "
          f"交换={stats['swapped_mb']}MB, 空闲={stats['free_mb']}MB")
    print(f"交换操作: {stats['swap_operations']}次, 平均交换时间={stats['avg_swap_time_ms']:.2f}ms")
    
    # 访问已被交换的内存
    if len(allocated_mem) > 0:
        print("访问已被交换的内存...")
        for i, mem in enumerate(allocated_mem):
            if pool.access_memory(mem):
                # 使用内存
                for j in range(min(100, len(mem))):
                    mem[j] = (i + j) % 256
                print(f"成功访问第 {i+1} 块内存")
            else:
                print(f"访问第 {i+1} 块内存失败")
    
    # 再次获取统计信息
    stats = pool.get_stats()
    print(f"访问后统计: 已分配={stats['allocated_mb']}MB, 交换={stats['swapped_mb']}MB")
    print(f"交换操作: {stats['swap_operations']}次, 平均交换时间={stats['avg_swap_time_ms']:.2f}ms")
    
    # 释放所有内存
    for mem in allocated_mem:
        pool.free(mem)
    
    # 最终统计
    stats = pool.get_stats()
    print(f"最终统计: 已分配={stats['allocated_mb']}MB, 交换={stats['swapped_mb']}MB")