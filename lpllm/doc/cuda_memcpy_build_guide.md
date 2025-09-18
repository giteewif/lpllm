# CUDA Memcpy 扩展构建指南

## 概述

已经将所有 `tensor.copy_()` 调用替换为使用 `cudaMemcpy` 的实现。需要重新编译 C++ 扩展以包含新的 CUDA 内存复制函数。

## 新增文件

### 1. C++ 实现文件
- `lpllm/sllm_store/csrc/checkpoint/cuda_memcpy.cpp` - CUDA memcpy 实现
- `lpllm/sllm_store/csrc/checkpoint/cuda_memcpy.h` - 头文件
- 已修改 `checkpoint_py.cpp` 以包含新的 Python 绑定

### 2. Python 包装器
- `cuda_memcpy_utils.py` - 高级 Python 接口

## 构建步骤

### 1. 检查依赖
确保安装了必要的依赖：
```bash
pip install torch pybind11 setuptools
# 确保CUDA工具包已安装
nvcc --version
```

### 2. 修改 CMakeLists.txt 或 setup.py

需要将新的源文件添加到构建配置中：

#### CMakeLists.txt 修改：
```cmake
# 添加到源文件列表
set(CHECKPOINT_SOURCES
    csrc/checkpoint/checkpoint.cpp
    csrc/checkpoint/cuda_memcpy.cpp  # 新增
)
```

#### setup.py 修改：
```python
ext_modules = [
    CppExtension(
        'sllm_store._C',
        [
            'csrc/checkpoint/checkpoint.cpp',
            'csrc/checkpoint/cuda_memcpy.cpp',  # 新增
            'csrc/checkpoint/checkpoint_py.cpp',
        ],
        include_dirs=[
            'csrc',
            'csrc/checkpoint',
        ],
        libraries=['cuda', 'cudart'],  # 确保链接CUDA库
        language='c++'
    )
]
```

### 3. 构建扩展

#### 使用 setuptools：
```bash
cd /mnt/zhengcf3/lpllm/lpllm/sllm_store
python setup.py build_ext --inplace
```

#### 或者使用 pip：
```bash
cd /mnt/zhengcf3/lpllm/lpllm/sllm_store
pip install -e .
```

### 4. 验证构建

验证新函数是否可用：
```python
try:
    from sllm_store._C import (
        cuda_memcpy_h2d,
        cuda_memcpy_d2h, 
        cuda_memcpy_d2d,
        cuda_memcpy_smart
    )
    print("CUDA memcpy functions available!")
except ImportError as e:
    print(f"CUDA memcpy functions not available: {e}")
```

## 代码修改摘要

### 修改的文件：
1. **lpllm.py** - 5个 copy_ 调用替换为 cuda_copy_
2. **server_pool.py** - copy_func 方法使用 cuda_copy_
3. **pinpool.py** - copy_func 方法使用 cuda_copy_
4. **test_server_pool.py** - 测试中的 copy_ 调用替换

### 主要功能：
- **cuda_copy_()** - 智能复制，自动检测方向
- **cuda_copy_h2d()** - Host到Device复制
- **cuda_copy_d2h()** - Device到Host复制
- **cuda_copy_d2d()** - Device到Device复制

### 错误处理：
- 如果CUDA memcpy不可用，自动回退到PyTorch的copy_()
- 提供详细的错误信息和状态码

## 优势

### 性能优势：
1. **直接CUDA API调用** - 避免PyTorch的开销
2. **精确控制** - 可以指定stream和同步行为
3. **更好的调试** - 明确的错误码和消息

### 兼容性：
1. **向后兼容** - 自动回退机制
2. **相同接口** - 与原来的copy_()保持相同的调用方式
3. **异常安全** - 错误时提供清晰的异常信息

## 测试

### 基本测试：
```python
import torch
from cuda_memcpy_utils import cuda_copy_

# CPU to GPU
src = torch.randn(1024, 1024, dtype=torch.float32, pin_memory=True)
dst = torch.empty_like(src, device='cuda:0')
cuda_copy_(dst, src, non_blocking=True)

# GPU to CPU  
src_gpu = torch.randn(1024, 1024, dtype=torch.float32, device='cuda:0')
dst_cpu = torch.empty_like(src_gpu, pin_memory=True)
cuda_copy_(dst_cpu, src_gpu, non_blocking=True)
```

### 性能测试：
运行 `test_server_pool.py` 来比较性能差异。

## 故障排除

### 常见问题：

1. **编译错误**：检查CUDA工具包是否正确安装
2. **链接错误**：确保链接了正确的CUDA库
3. **运行时错误**：检查CUDA驱动和运行时版本兼容性
4. **导入错误**：确保Python能找到编译的扩展

### 调试技巧：
```python
# 检查CUDA可用性
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")

# 检查扩展是否构建正确
from cuda_memcpy_utils import is_cuda_memcpy_available
print(f"CUDA memcpy available: {is_cuda_memcpy_available()}")
```
