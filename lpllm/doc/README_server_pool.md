# ServerPinnedMemoryPool 实现说明
sllm-store start --storage-path /mnt/zhengcf3/models/models --mem-pool-size 40GB
## 概述

本实现将原有的 `pool_memory` 改为使用 server 端额外申请的空间作为 pool memory。

## 实现架构

### 1. Server端改进 (`sllm_store/server.py`)

- 在 `StorageServicer` 中添加了 pool memory 管理功能
- 新增参数 `pool_memory_size` (默认4GB) 用于控制pool内存大小
- 新增方法:
  - `AllocatePoolMemory`: 为客户端分配pool内存
  - `FreePoolMemory`: 释放客户端的pool内存
- 使用 `allocate_cuda_memory` 和 `get_cuda_memory_handles` 管理CUDA内存

### 2. Client端接口 (`sllm_store/client.py`)

- 新增方法:
  - `allocate_pool_memory(client_id, device_id, size_mb)`: 从server申请pool内存
  - `free_pool_memory(client_id)`: 释放pool内存

### 3. ServerPinnedMemoryPool (`server_pool.py`)

- 新建类，保持与 `PinnedMemoryPool` 相同的接口
- 构造时自动从server申请CUDA内存
- 目前使用本地pinned内存作为fallback (待完整IPC实现)
- 析构时自动释放server端内存

### 4. AttnManager集成 (`lpllm.py`)

- 添加 `use_server_pool` 参数控制是否使用server pool
- 默认启用 server pool (`use_server_pool=True`)

## 使用方式

### 启动Server

```python
# server启动时可指定pool_memory_size
await serve(
    host="127.0.0.1",
    port=8073,
    storage_path="/path/to/storage",
    pool_memory_size=4096,  # 4GB pool memory
    # ... other params
)
```

### Client使用

```python
# 现有代码无需修改，AttnManager会自动使用server pool
attn_manager = AttnManager(
    lpmodule_class=lpmodule_class, 
    device="cuda:0", 
    config=config, 
    pool_size=1024,  # 1GB
    use_server_pool=True  # 使用server pool
)
```

### 直接使用ServerPinnedMemoryPool

```python
from server_pool import ServerPinnedMemoryPool

# 创建server pool
pool = ServerPinnedMemoryPool(
    dtype=torch.bfloat16,
    pool_size=1024,  # 1GB
    device="cuda:0"
)

# 使用方式与PinnedMemoryPool完全相同
tensor = torch.randn(1024, 1024, dtype=torch.bfloat16)
allocated_block = pool.alloc_same_pin_tensor(tensor)
allocated_block.copy_(tensor)
pool.free(allocated_block)
```

## 技术细节

### 内存管理

1. **Server端**: 使用 `allocate_cuda_memory` 分配CUDA内存
2. **IPC通信**: 通过 `get_cuda_memory_handles` 获取内存句柄
3. **Client端**: 接收内存句柄并映射到本地地址空间

### 当前限制

- IPC内存映射部分使用本地内存作为fallback
- 需要完整的C++层面IPC支持才能实现真正的server端内存共享

### 兼容性

- 保持与原有 `PinnedMemoryPool` 完全相同的接口
- 现有代码只需在 `AttnManager` 创建时设置 `use_server_pool=True`

## 优势

1. **集中管理**: Server统一管理所有client的pool内存
2. **资源优化**: 避免每个client单独申请大量内存
3. **灵活配置**: 可在server端统一配置pool大小
4. **无缝集成**: 与现有代码完全兼容

## 后续工作

1. 完善C++层面的IPC内存映射实现
2. 添加更多的内存管理策略(如动态调整)
3. 增加监控和统计功能
4. 优化多client场景下的内存分配效率


## 安装
pip install grpcio-tools
### 生成protobuf
cd /mnt/zhengcf3/lpllm/lpllm/sllm_store && python -m grpc_tools.protoc --proto_path=proto --python_out=sllm_store --grpc_python_out=sllm_store proto/storage.proto  

### 检查新字段
grep -E "memory_ptr|start_offset|size_bytes" /mnt/zhengcf3/lpllm/lpllm/sllm_store/sllm_store/storage_pb2.py