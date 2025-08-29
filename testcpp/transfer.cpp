#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cstring>

// CUDA 错误检查宏
#define CUDA_CHECK(cmd) \
do { \
    cudaError_t error = cmd; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 使用固定内存传输数据到 GPU
void transferToGPUWithPinnedMemory(const float* host_data, float* device_ptr, size_t size) {
    CUDA_CHECK(cudaSetDevice(1));
    // 分配固定内存 (pinned memory)
    float* pinned_host_ptr;
    CUDA_CHECK(cudaMallocHost((void**)&pinned_host_ptr, size * sizeof(float)));
    
    // 复制数据到固定内存 (比普通内存快)
    auto start_copy = std::chrono::high_resolution_clock::now();
    memcpy(pinned_host_ptr, host_data, size * sizeof(float));
    auto end_copy = std::chrono::high_resolution_clock::now();
    
    // 异步传输到 GPU
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    auto start_transfer = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(device_ptr, pinned_host_ptr, 
                              size * sizeof(float), 
                              cudaMemcpyHostToDevice, 
                              stream));
    
    // 等待传输完成
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end_transfer = std::chrono::high_resolution_clock::now();
    
    
    // 清理
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(pinned_host_ptr));
    
    // 计算时间
    auto copy_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_copy - start_copy).count();
    auto transfer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_transfer - start_transfer).count();
    
    double bandwidth = (size * sizeof(float)) / (transfer_duration * 1e-6) / (1024.0 * 1024.0 * 1024.0); // GB/s
    
    std::cout << "固定内存复制时间: " << copy_duration << " ms\n";
    std::cout << "CPU->GPU 传输时间: " << transfer_duration << " ms\n";
    std::cout << "传输带宽: " << bandwidth << " GB/s\n";
}

// GPU TO CPU
void transferToCPUWithPinnedMemory(const float* host_data, float* device_ptr, size_t size) {
    CUDA_CHECK(cudaSetDevice(1));
    // 分配固定内存 (pinned memory)
    float* pinned_host_ptr;
    CUDA_CHECK(cudaMallocHost((void**)&pinned_host_ptr, size * sizeof(float)));
    
    // 复制数据到固定内存 (比普通内存快)
    auto start_copy = std::chrono::high_resolution_clock::now();
    memcpy(pinned_host_ptr, host_data, size * sizeof(float));
    auto end_copy = std::chrono::high_resolution_clock::now();
    
    // 异步传输到 GPU
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    auto start_transfer = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(device_ptr, pinned_host_ptr, 
                              size * sizeof(float), 
                              cudaMemcpyDeviceToHost, 
                              stream));
    
    // 等待传输完成
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end_transfer = std::chrono::high_resolution_clock::now();
    
    
    // 清理
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(pinned_host_ptr));
    
    // 计算时间
    auto copy_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_copy - start_copy).count();
    auto transfer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_transfer - start_transfer).count();
    
    double bandwidth = (size * sizeof(float)) / (transfer_duration * 1e-6) / (1024.0 * 1024.0 * 1024.0); // GB/s
    
    std::cout << "固定内存复制时间: " << copy_duration << " ms\n";
    std::cout << "GPU->CPU 传输时间: " << transfer_duration << " ms\n";
    std::cout << "传输带宽: " << bandwidth << " GB/s\n";
}

// 直接传输数据到 GPU (无优化)
void transferToGPUDirect(const float* host_data, float* device_ptr, size_t size) {
    CUDA_CHECK(cudaSetDevice(1));

    auto start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(device_ptr, host_data, 
                         size * sizeof(float), 
                         cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double bandwidth = (size * sizeof(float)) / (duration * 1e-6) / (1024.0 * 1024.0 * 1024.0); // GB/s
    
    std::cout << "直接传输时间: " << duration << " ms\n";
    std::cout << "传输带宽: " << bandwidth << " GB/s\n";
}


int main() {
    const size_t data_size = 512 * 4 * 2048 * 3  / 2; // 100 MB 数据 (25 million floats)
    std::vector<float> host_data(data_size);
    
    // 初始化主机数据
    for (size_t i = 0; i < data_size; ++i) {
        host_data[i] = static_cast<float>(i);
    }
    
    // 在 GPU 上分配内存
    float* device_ptr;
    // 指定使用设备 0，可根据需求修改设备编号
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaMalloc(&device_ptr, data_size * sizeof(float)));
    
    transferToGPUDirect(host_data.data(), device_ptr, data_size);
    transferToGPUWithPinnedMemory(host_data.data(), device_ptr, data_size);
    
    std::cout << "\n=== 直接传输测试 ===\n";
    transferToGPUDirect(host_data.data(), device_ptr, data_size);
    
    std::cout << "\n=== 固定内存优化传输测试 ===\n";
    transferToGPUWithPinnedMemory(host_data.data(), device_ptr, data_size);
    
    std::cout << "\n=== 固定CPU TO GPU优化传输测试 ===\n";
    transferToCPUWithPinnedMemory(host_data.data(), device_ptr, data_size);

    // 验证传输 - 在 GPU 上执行简单操作
    const int block_size = 256;
    const int grid_size = (data_size + block_size - 1) / block_size;
    // validateTransfer<<<grid_size, block_size>>>(device_ptr, data_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 将结果复制回 CPU
    std::vector<float> result(data_size);
    CUDA_CHECK(cudaMemcpy(result.data(), device_ptr, 
                         data_size * sizeof(float), 
                         cudaMemcpyDeviceToHost));
    
    
    // 清理
    CUDA_CHECK(cudaFree(device_ptr));
    
    return 0;
}