import os
import json
import fcntl
import mmap
import numpy as np
import torch
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List
from safetensors import safe_open
import math
from typing import Dict, Tuple, Optional
import concurrent
@dataclass
class TensorInfo:
    name: str
    dtype: torch.dtype
    shape: Tuple[int, ...]
    offset: int
    size: int
    view: Optional[torch.Tensor] = None  # 添加view字段存储tensor视图

class SafetensorReader:
    def __init__(self):
        self.header_size = None
        self.total_size = 0
        
    def _dtype_from_str(self, dtype_str: str) -> torch.dtype:
        """Convert safetensors dtype string to torch dtype"""
        dtype_map = {
            'BF16': torch.bfloat16,
            'F16': torch.float16,
            'F32': torch.float32,
            'I32': torch.int32,
        }
        return dtype_map.get(dtype_str, torch.float32)
        
    def load_metadata(self, layer_file):
        """Load metadata from safetensors file header"""
        tensor_map = {}
        with open(layer_file, 'rb') as f:
            # Read header size (8 bytes)
            header_size = int.from_bytes(f.read(8), 'little')
            
            # Read JSON header
            header = f.read(header_size).decode('utf-8')
            metadata = json.loads(header)
            
            # Parse tensor metadata
            for name, info in metadata.items():
                dtype = self._dtype_from_str(info['dtype'])
                shape = tuple(info['shape'])
                
                # Calculate aligned offset and size
                offset = info['data_offsets'][0] + 8 + header_size  # Add 8 for header size bytes
                end_offset = info['data_offsets'][1] + 8 + header_size
                size = end_offset - offset
                
                tensor_map[name] = TensorInfo(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    offset=offset,
                    size=size
                )
        # total_size = os.path.getsize(layer_file)
        return tensor_map
    def load_path(self, path, pin_pool, pool_size):
        index_path = path + "/model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)
            weight_map = index["weight_map"]
        start = 0
        layers_tensor_map = {}
        metadatas_map = {}
        for file_name in set(weight_map.values()):
            
            file_path = f"{path}/{file_name}"
            metadata_map = self.load_metadata(file_path)
            
            layer_x = file_name.split(".")[0].split("_")[1]
            layer_num = int(layer_x)
            if layer_num not in layers_tensor_map:
                layers_tensor_map[layer_num] = {}
            metadatas_map[layer_num] = metadata_map

            file_size = os.path.getsize(file_path)
            if start + file_size > pool_size:
                raise ValueError(
                    f"not enough pin memory size pin {pool_size}, need {start+file_size}"
                )
            fd = self.safe_open(layer_file=file_path)
            
            pin_buffer=pin_pool[start:]
            try:
                # 读取整个文件到pinned memory
                view = memoryview(pin_buffer.numpy()).cast('B')
                time_start_read = time.time()
                bytes_read = os.readv(fd, [view[start:]])
                print(f"file read cost {time.time()-time_start_read} seconds")
                if bytes_read != file_size:
                    raise IOError(f"读取不完整: {bytes_read} vs {file_size}")
                    
                # 为每个张量创建视图
                for name, info in metadata_map.items():
                    ksplits = name.split(".")
                    # models.layers.1.[]
                    layer_key = ".".join(ksplits[3])

                    tensor_bytes = pin_buffer[info.offset:info.offset + info.size]                
                    info.view = torch.frombuffer(
                        tensor_bytes.numpy(),
                        dtype=info.dtype
                    ).reshape(info.shape)
                    layers_tensor_map[layer_num][layer_key] = info.view
            finally:
                os.close(fd)
            start += bytes_read
        return layers_tensor_map, metadatas_map

    def load_to_pinned_memory_withpin(self, layer_file, pin_buffer, start):
        """将整个文件加载到连续的pinned memory中"""
        time_all=time.time()
        tensor_map, total_size=self.load_metadata(layer_file)
            
        # 计算对齐后的总大小
        aligned_size = math.ceil(total_size / 4096) * 4096
        
        # 分配连续的pinned memory
        pinned_buffer = torch.empty(aligned_size, dtype=torch.int8, pin_memory=True)

        time_start=time.time()
        # 使用O_DIRECT读取整个文件
        fd = self.safe_open(layer_file=layer_file)
        try:
            # 读取整个文件到pinned memory
            view = memoryview(pinned_buffer.numpy()).cast('B')
            bytes_read = os.readv(fd, [view[:aligned_size]])
            print(f"file read cost {time.time()-time_start} seconds")
            if bytes_read != total_size:
                raise IOError(f"读取不完整: {bytes_read} vs {aligned_size}")
                
            # 为每个张量创建视图
            for name, info in tensor_map.items():
                tensor_bytes = pinned_buffer[info.offset:info.offset + info.size]                
                info.view = torch.frombuffer(
                    tensor_bytes.numpy(),
                    dtype=info.dtype
                ).reshape(info.shape)
                
        finally:
            os.close(fd)
        print(f"load all {time.time()-time_all} s")
        return pinned_buffer
    def load_to_pinned_memory(self, layer_file):
        """将整个文件加载到连续的pinned memory中"""
        time_all=time.time()
        tensor_map, total_size=self.load_metadata(layer_file)

        # 计算对齐后的总大小
        aligned_size = math.ceil(total_size / 4096) * 4096
        
        # 分配连续的pinned memory
        pinned_buffer = torch.empty(aligned_size, dtype=torch.int8, pin_memory=True)

        time_start=time.time()
        # 使用O_DIRECT读取整个文件
        fd = self.safe_open(layer_file=layer_file)
        try:
            # 读取整个文件到pinned memory
            view = memoryview(pinned_buffer.numpy()).cast('B')
            bytes_read = os.readv(fd, [view[:aligned_size]])
            print(f"file read cost {time.time()-time_start} seconds")
            if bytes_read != total_size:
                raise IOError(f"读取不完整: {bytes_read} vs {aligned_size}")
                
            # 为每个张量创建视图
            for name, info in tensor_map.items():
                tensor_bytes = pinned_buffer[info.offset:info.offset + info.size]                
                info.view = torch.frombuffer(
                    tensor_bytes.numpy(),
                    dtype=info.dtype
                ).reshape(info.shape)
                
        finally:
            os.close(fd)
        print(f"load all {time.time()-time_all} s")
        return pinned_buffer
    def safe_open(self, layer_file):
        """Open safetensors file with optimized settings"""
        fd = os.open(layer_file, os.O_RDONLY | os.O_DIRECT)
        if fd < 0:
            raise IOError(f"Failed to open {layer_file}")
        
        # Set sequential read hint
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        return fd
        
    def load_tensor(self, name: str) -> torch.Tensor:
        """Load specific tensor into pinned memory"""
        if name not in self.tensor_map:
            raise KeyError(f"Tensor {name} not found")
            
        info = self.tensor_map[name]
        pinned_buffer = torch.empty(info.shape, dtype=info.dtype, pin_memory=True)
        
        fd = self.safe_open()
        try:
            # Seek to tensor position
            os.lseek(fd, info.offset, os.SEEK_SET)
            
            # Read directly into pinned memory
            view = memoryview(pinned_buffer.numpy()).cast('B')
            aligned_size = ((info.size + 4095) // 4096) * 4096
            bytes_read = os.readv(fd, [view[:aligned_size]])
            
            if bytes_read != aligned_size:
                raise IOError(f"Short read: {bytes_read} vs {aligned_size}")
                
        finally:
            os.close(fd)
            
        return pinned_buffer
    def get_tensor(self, name: str) -> torch.Tensor:
        """获取张量视图"""
        if self.pinned_buffer is None:
            raise RuntimeError("请先调用load_to_pinned_memory()")
        if name not in self.tensor_map:
            raise KeyError(f"未找到张量 {name}")
        return self.tensor_map[name].view
    
    def to_cuda(self, name: str, device="cuda:0", non_blocking=True) -> torch.Tensor:
        """将张量传输到CUDA设备"""
        return self.get_tensor(name).to(device=device, non_blocking=non_blocking)
        
    # def __del__(self):
    #     """清理pinned memory"""
    #     if self.pinned_buffer is not None:
    #         del self.pinned_buffer
    #         self.pinned_buffer = None
if __name__ == "__main__":
    layer_file = "/models/layer_1.safetensors"
    reader = SafetensorReader()
    
    # 加载到pinned memory
    print(f"\n加载 {os.path.basename(layer_file)} 到pinned memory...")
    start_time = time.time()

    # for i in range(28):
    #     if i == 0:
    #         continue
    #     file_path = format(f"/models/layer_{i}.safetensors")
    #     print(file_path)
    #     pin = reader.load_to_pinned_memory(file_path)
    # print(f"加载完成，耗时 {time.time() - start_time:.6f} 秒")
    
    time_start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        future_list = []
        for i in range(2):
            if i == 0:
                continue
            file_path = format(f"/models/layer_{i}.safetensors")
            print(file_path)
            future = executor.submit(reader.load_to_pinned_memory, file_path)
            future_list.append(future)
        
        for i in future_list:
            i.result()
    print(f"cost {time.time()-time_start}")
    # 打印内存映射
    # print("\n张量映射:")
    # total_gb = 0
    # for name, info in reader.tensor_map.items():
    #     size_gb = info.size / (1024**3)
    #     total_gb += size_gb
    #     print(f"{name}:")
    #     print(f"  形状: {info.shape}")
    #     print(f"  大小: {size_gb:.2f} GB")
    #     print(f"  偏移: {info.offset}")
    # print(f"\n总大小: {total_gb:.2f} GB")
    