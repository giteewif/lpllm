# 重新生成 storage_pb2.py 和 storage_pb2_grpc.py

## 更新内容

已更新 `storage.proto` 文件，添加了以下新的 gRPC 服务方法：

### 新增服务方法
1. `AllocatePoolMemory` - 为客户端分配 pinned memory 池
2. `FreePoolMemory` - 释放客户端的 pinned memory 池
3. `AllocateFromPool` - 从池中分配内存块
4. `FreeFromPool` - 释放内存块回池中
5. `CopyToPool` - 复制数据到池内存
6. `CopyFromPool` - 从池内存复制数据

### 新增消息类型
- `AllocatePoolMemoryRequest/Response`
- `FreePoolMemoryRequest/Response`
- `AllocateFromPoolRequest/Response`
- `FreeFromPoolRequest/Response`
- `CopyToPoolRequest/Response`
- `CopyFromPoolRequest/Response`

## 重新生成步骤

### 方法1：使用 protoc 命令行工具

```bash
cd /mnt/zhengcf3/lpllm/lpllm/sllm_store

# 生成 Python 代码
python -m grpc_tools.protoc \
    --proto_path=proto \
    --python_out=sllm_store \
    --grpc_python_out=sllm_store \
    proto/storage.proto
copy to sllm_store/proto
add sllm_store to package
```


### 方法2：使用 Python grpcio-tools

```python
import grpc_tools.protoc

grpc_tools.protoc.main([
    'grpc_tools.protoc',
    '--proto_path=proto',
    '--python_out=sllm_store',
    '--grpc_python_out=sllm_store',
    'proto/storage.proto'
])
```

### 方法3：检查是否有构建脚本

查看项目中是否有现有的构建脚本：

```bash
find /mnt/zhengcf3/lpllm/lpllm/sllm_store -name "*.py" -o -name "*.sh" -o -name "Makefile" | grep -E "(build|gen|proto)"
```

## 验证生成结果

生成完成后，检查以下文件是否包含新的消息类型：

1. `sllm_store/storage_pb2.py` - 应包含新的消息类
2. `sllm_store/storage_pb2_grpc.py` - 应包含新的服务接口

验证方法：
```bash
grep -E "AllocatePoolMemory|FreePoolMemory|AllocateFromPool|FreeFromPool|CopyToPool|CopyFromPool" sllm_store/storage_pb2*.py
```

## 注意事项

1. **确保安装 grpcio-tools**：
   ```bash
   pip install grpcio-tools
   ```

2. **路径问题**：确保在正确的目录中运行命令

3. **权限问题**：确保有写入权限到目标目录

4. **导入问题**：生成后可能需要调整 import 路径

## 如果遇到问题

如果自动生成遇到问题，也可以手动编辑现有的 `storage_pb2.py` 和 `storage_pb2_grpc.py` 文件，添加必要的类和方法定义。
