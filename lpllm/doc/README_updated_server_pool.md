# 更新版 ServerPinnedMemoryPool 实现说明

## 概述

现在的实现完全基于sllm-store的PinnedMemoryPool的chunk-based方法，不再分配GPU内存，只分配服务端的pinned memory（CPU固定内存）。

## 主要改进

### 1. 使用chunk-based分配策略
- **与sllm-store一致**: 采用与sllm-store的PinnedMemoryPool相同的chunk-based分配方法
- **固定chunk大小**: 使用server端配置的chunk_size，保证内存对齐
- **高效管理**: 通过chunk索引管理内存，避免内存碎片

### 2. Server端实现 (`server.py`)

#### 内存池结构
```python
self.client_pool_memories[client_id] = {
    'pinned_chunks': pinned_chunks,         # 实际的pinned memory chunks
    'chunk_size': chunk_size,               # 每个chunk的大小
    'num_chunks': num_chunks,               # chunk总数
    'free_chunks': free_chunks,             # 空闲chunk集合
    'allocated_chunks': allocated_chunks,    # 已分配chunk映射
    'next_allocation_id': 0                 # 分配ID生成器
}
```

#### 关键方法
- `AllocatePoolMemory`: 为客户端分配chunk-based内存池
- `AllocateFromPool`: 从池中分配指定大小的内存块
- `FreeFromPool`: 释放内存块回池中
- `CopyToPool`/`CopyFromPool`: 跨chunk的数据拷贝操作

### 3. Client端实现 (`client.py`)

#### 更新的接口
- `allocate_from_pool()`: 返回allocation_id和chunk信息
- `free_from_pool()`: 使用allocation_id释放内存
- `copy_to_pool()`/`copy_from_pool()`: 使用chunk-based寻址

### 4. ServerPinnedMemoryPool类

#### 新架构
- **ServerMemoryBlock**: 代理对象，表示服务端的内存块
- **完全兼容**: 保持与原PinnedMemoryPool相同的接口
- **透明操作**: 所有内存操作通过gRPC转发到服务端

#### ServerMemoryBlock特性
```python
class ServerMemoryBlock:
    def copy_(self, src: torch.Tensor, non_blocking=False)  # 拷贝数据到服务端
    def to(self, device: str, non_blocking=False)           # 从服务端读取到设备
    def view(self, *shape)                                  # 创建新形状的视图
    def data_ptr(self)                                      # 唯一标识符
```

## 使用方式

### 基本使用（与原接口完全兼容）
```python
# 创建server pool - 接口不变
pool = ServerPinnedMemoryPool(
    dtype=torch.bfloat16,
    pool_size=1024,  # MB
    device="cuda:0"
)

# 分配内存 - 接口不变
tensor = torch.randn(1024, 1024, dtype=torch.bfloat16)
allocated_block = pool.alloc_same_pin_tensor(tensor)

# 数据操作 - 接口不变
allocated_block.copy_(tensor)
result = allocated_block.to("cuda:0")

# 释放内存 - 接口不变
pool.free(allocated_block)
```

### 高级特性
```python
# 查看使用情况
usage = pool.get_usage_info()
print(f"活跃分配: {usage['active_allocations']}")
print(f"使用的chunks: {usage['allocated_chunks']}")
print(f"Chunk大小: {usage['chunk_size']}")
```

## 技术优势

### 1. **内存效率**
- 使用chunk-based分配，减少内存碎片
- 服务端统一管理，避免客户端重复申请
- 固定chunk大小，提高分配效率

### 2. **与sllm-store一致**
- 采用相同的PinnedMemoryPool设计理念
- 使用相同的chunk_size配置
- 兼容现有的内存管理策略

### 3. **完全兼容**
- 客户端接口与原PinnedMemoryPool完全相同
- 现有代码无需修改
- 透明的服务端操作

### 4. **可扩展性**
- 支持多客户端并发使用
- 每个客户端独立的内存池
- 线程安全的操作

## 内存管理流程

### 分配流程
1. 客户端请求分配N KB内存
2. 服务端计算所需chunks: `ceil(N KB / chunk_size)`
3. 从free_chunks中分配所需数量的chunks
4. 返回allocation_id和chunk_indices
5. 客户端创建ServerMemoryBlock代理对象

### 释放流程  
1. 客户端调用pool.free(block)
2. 使用allocation_id向服务端请求释放
3. 服务端将chunks返回到free_chunks
4. 客户端移除本地跟踪信息

### 数据操作流程
1. `copy_()`: 将tensor转换为bytes，通过gRPC传输到服务端chunks
2. `to()`: 从服务端chunks读取bytes，转换为tensor并移动到目标设备

## 配置说明

### 服务端配置
```python
# 启动时配置
await serve(
    host="127.0.0.1",
    port=8073,
    pool_memory_size=4096,  # 默认pool大小（MB）
    # ... 其他参数
)
```

### 客户端配置
```python
# 自动使用server pool
attn_manager = AttnManager(
    lpmodule_class=lpmodule_class,
    device="cuda:0", 
    config=config,
    pool_size=1024,      # 客户端pool大小（MB）
    use_server_pool=True  # 启用server pool
)
```

## 注意事项

1. **网络延迟**: 数据传输通过gRPC，适合大块数据操作
2. **chunk对齐**: 小于chunk_size的分配会占用完整chunk
3. **生命周期**: ServerMemoryBlock对象销毁时自动释放服务端内存
4. **错误处理**: 网络错误会抛出RuntimeError异常

这个实现完全符合sllm-store的设计理念，同时保持了与现有代码的兼容性。
