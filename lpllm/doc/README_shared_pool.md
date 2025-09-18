# Server Pre-allocated Memory Pool Implementation

## 概述

本实现将 `ServerPinnedMemoryPool` 修改为服务器预先分配一大段连续的 pinned memory，然后客户端向服务器申请其中的一部分供自己使用。客户端在本地管理从服务器申请的内存块，通过网络通信进行数据读写操作。

## 主要变更

### 1. ServerPinnedMemoryPool (`server_pool.py`)

**变更前:**
- 每次分配都向服务器请求小块内存
- 频繁的网络通信进行内存分配和释放

**变更后:**
- 初始化时向服务器申请一大段连续内存
- 客户端本地管理从服务器申请的内存块
- `copy_()` 和 `to()` 方法通过网络操作服务器内存

**关键方法变更:**
- `_initialize_server_connection()`: 获取服务器配置
- `_allocate_memory_from_server()`: 从服务器预分配内存中申请部分
- `alloc_kb()`: 本地内存管理，使用 first-fit 算法
- `free()`: 本地内存释放，支持相邻块合并
- `copy_()` / `to()`: 通过网络操作服务器内存
- `__del__()`: 释放服务器内存分配

### 2. Server端 (`sllm_store/sllm_store/server.py`)

**变更前:**
- 为每个客户端分配独立的内存块
- 复杂的客户端内存管理

**变更后:**
- 服务器启动时预分配一大段连续 pinned memory
- 客户端从预分配内存中申请部分使用
- 服务器统一管理预分配内存的分配和释放

**关键变更:**
- 新增 `_pre_allocate_memory()`: 服务器启动时预分配大段内存
- `_allocate_from_pre_allocated()`: 从预分配内存中分配给客户端
- `_free_from_pre_allocated()`: 将客户端内存释放回预分配池
- `CopyToPool()` / `CopyFromPool()`: 直接操作预分配内存

## 架构优势

### 1. 内存效率
- **预分配**: 服务器启动时预分配大段连续内存
- **统一管理**: 服务器统一管理内存分配和释放
- **减少碎片**: 使用 first-fit 算法和块合并减少内存碎片
- **多客户端共享**: 多个客户端共享预分配的内存池

### 2. 性能优势
- **连续内存**: 大段连续 pinned memory 提高缓存局部性
- **pinned memory**: 优化的 GPU 数据传输
- **减少分配**: 客户端本地管理，减少服务器分配请求
- **网络优化**: 批量数据传输，减少网络开销

### 3. 架构优势
- **集中管理**: 服务器集中管理内存资源
- **资源复用**: 客户端释放的内存可以被其他客户端使用
- **负载均衡**: 服务器可以根据使用情况优化内存分配
- **监控友好**: 服务器可以监控整体内存使用情况

### 4. 兼容性
- **接口不变**: 保持与原有 `PinnedMemoryPool` 相同的接口
- **透明使用**: 客户端代码无需修改
- **易于调试**: 服务器端统一管理，便于调试

## 使用方式

### 启动服务器
```bash
# 启动服务器时指定预分配内存大小
python -m sllm_store.server --pool-memory-size 4096  # 预分配4GB内存
```

### 客户端使用
```python
from server_pool import ServerPinnedMemoryPool

# 创建服务器内存池客户端
pool = ServerPinnedMemoryPool(
    dtype=torch.bfloat16,
    pool_size=1024,  # 从服务器申请1GB内存
    device="cuda:0"
)

# 使用方式与原来完全相同
memory_block = pool.alloc_same_pin_tensor(tensor)
memory_block.copy_(tensor)  # 通过网络写入服务器内存
result = memory_block.to("cuda:0")  # 通过网络读取服务器内存
pool.free(memory_block)
```

**注意**: 需要先启动服务器，客户端通过网络与服务器通信。

## 测试

运行测试脚本验证实现：
```bash
python test_shared_pool.py
```

## 配置参数

### 服务器端
- `pool_memory_size`: 共享池大小（MB），默认4GB
- `chunk_size`: 内存块大小，从存储配置获取

### 客户端
- `dtype`: 数据类型，默认 `torch.bfloat16`
- `device`: 设备，默认 `"cuda:0"`
- `pool_size`: 被忽略（用于兼容性）

## 注意事项

1. **服务器启动**: 服务器必须在客户端之前启动并初始化共享池
2. **内存管理**: 客户端必须正确释放分配的内存块
3. **并发安全**: 服务器端使用锁保证线程安全
4. **错误处理**: 包含完整的错误处理和日志记录

## 性能考虑

1. **内存碎片**: 使用基于块的分配减少碎片
2. **网络开销**: 通过gRPC进行内存操作，需要网络通信
3. **并发性能**: 多个客户端可以同时分配和释放内存
4. **缓存友好**: 使用pinned memory提高数据传输性能
