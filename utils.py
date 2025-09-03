import torch
import pynvml
import os
from pynvml import *
def get_torch_gpu_memory():
    """获取PyTorch分配的显存"""
    if not torch.cuda.is_available():
        return 0.0
    
    # 清除未使用的显存
    torch.cuda.empty_cache()
    
    # 获取当前进程分配的显存
    allocated = torch.cuda.memory_allocated()
    
    # 转换为MB并返回
    print(f"gpu {round(allocated / (1024 ** 2), 2)} MB ")
    return round(allocated / (1024 ** 2), 2)
def get_gpu_memory_usage(name=""):
    try:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)  # 默认使用第一个GPU
        info = nvmlDeviceGetMemoryInfo(handle)
        
        map = {
            'total_GB': info.total / (1024 ** 3),
            'used_GB': info.used / (1024 ** 3),
            'free_GB': info.free / (1024 ** 3)
        }
        print(f"gpu_usage:-- {name} {map}")
        return {
            'total_GB': info.total / (1024 ** 3),
            'used_GB': info.used / (1024 ** 3),
            'free_GB': info.free / (1024 ** 3)
        }
    except NVMLError as error:
        return f"NVML Error: {error}"
    finally:
        try:
            nvmlShutdown()
        except:
            pass

def get_gpu_memory_usage1():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    
    device = torch.device('cuda:1')
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 转换为MB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)     # 转换为MB
    
    print({
        'allocated_memory_MB': allocated,
        'reserved_memory_MB': reserved,
        'device': device
    })
    return {
        'allocated_memory_MB': allocated,
        'reserved_memory_MB': reserved,
        'device': device
    }
def get_current_process_gpu_memory():
    """获取当前程序在GPU上占用的显存（按进程ID）"""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 默认使用第一个GPU
        
        # 获取当前进程ID
        pid = os.getpid()
        current_process_mem = 0
        
        # 查找当前进程在所有GPU上的显存使用
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for p in processes:
                if p.pid == pid:
                    current_process_mem += p.usedGpuMemory
        
        # 转换为MB并返回
        print(f"gpu {round(current_process_mem / (1024 ** 2), 2) if current_process_mem > 0 else 0.0} MB ")
        return round(current_process_mem / (1024 ** 2), 2) if current_process_mem > 0 else 0.0
        
    except pynvml.NVMLError as e:
        return f"Error: {e}"
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass