// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------

#include "cuda_memcpy.h"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA memory copy functions
int CudaMemcpyHostToDevice(void* dst, const void* src, size_t count, cudaStream_t stream) {
    cudaError_t err;
    if (stream != nullptr) {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
    } else {
        err = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
    }
    
    if (err != cudaSuccess) {
        return -1;  // Error
    }
    return 0;  // Success
}

int CudaMemcpyDeviceToHost(void* dst, const void* src, size_t count, cudaStream_t stream) {
    cudaError_t err;
    if (stream != nullptr) {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
    } else {
        err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
    }
    
    if (err != cudaSuccess) {
        return -1;  // Error
    }
    return 0;  // Success
}

int CudaMemcpyDeviceToDevice(void* dst, const void* src, size_t count, cudaStream_t stream) {
    cudaError_t err;
    if (stream != nullptr) {
        err = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
    } else {
        err = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
    }
    
    if (err != cudaSuccess) {
        return -1;  // Error
    }
    return 0;  // Success
}

// Tensor-based CUDA memory copy functions
int CudaMemcpyTensorHostToDevice(torch::Tensor& dst_tensor, const torch::Tensor& src_tensor, bool non_blocking) {
    // Ensure tensors are contiguous
    auto src_contiguous = src_tensor.contiguous();
    auto dst_contiguous = dst_tensor.contiguous();
    
    // Check size compatibility
    if (src_contiguous.numel() != dst_contiguous.numel()) {
        return -2;  // Size mismatch
    }
    
    // Get data pointers
    void* dst_ptr = dst_contiguous.data_ptr();
    const void* src_ptr = src_contiguous.data_ptr();
    size_t size = src_contiguous.numel() * src_contiguous.element_size();
    
    // Get current CUDA stream if non_blocking
    cudaStream_t stream = nullptr;
    if (non_blocking && dst_tensor.is_cuda()) {
        stream = at::cuda::getCurrentCUDAStream(dst_tensor.device().index());
    }
    
    return CudaMemcpyHostToDevice(dst_ptr, src_ptr, size, stream);
}

int CudaMemcpyTensorDeviceToHost(torch::Tensor& dst_tensor, const torch::Tensor& src_tensor, bool non_blocking) {
    // Ensure tensors are contiguous
    auto src_contiguous = src_tensor.contiguous();
    auto dst_contiguous = dst_tensor.contiguous();
    
    // Check size compatibility
    if (src_contiguous.numel() != dst_contiguous.numel()) {
        return -2;  // Size mismatch
    }
    
    // Get data pointers
    void* dst_ptr = dst_contiguous.data_ptr();
    const void* src_ptr = src_contiguous.data_ptr();
    size_t size = src_contiguous.numel() * src_contiguous.element_size();
    
    // Get current CUDA stream if non_blocking
    cudaStream_t stream = nullptr;
    if (non_blocking && src_tensor.is_cuda()) {
        stream = at::cuda::getCurrentCUDAStream(src_tensor.device().index());
    }
    
    return CudaMemcpyDeviceToHost(dst_ptr, src_ptr, size, stream);
}

int CudaMemcpyTensorDeviceToDevice(torch::Tensor& dst_tensor, const torch::Tensor& src_tensor, bool non_blocking) {
    // Ensure tensors are contiguous
    auto src_contiguous = src_tensor.contiguous();
    auto dst_contiguous = dst_tensor.contiguous();
    
    // Check size compatibility
    if (src_contiguous.numel() != dst_contiguous.numel()) {
        return -2;  // Size mismatch
    }
    
    // Get data pointers
    void* dst_ptr = dst_contiguous.data_ptr();
    const void* src_ptr = src_contiguous.data_ptr();
    size_t size = src_contiguous.numel() * src_contiguous.element_size();
    
    // Get current CUDA stream if non_blocking
    cudaStream_t stream = nullptr;
    if (non_blocking) {
        // Use source device stream
        if (src_tensor.is_cuda()) {
            stream = at::cuda::getCurrentCUDAStream(src_tensor.device().index());
        } else if (dst_tensor.is_cuda()) {
            stream = at::cuda::getCurrentCUDAStream(dst_tensor.device().index());
        }
    }
    
    return CudaMemcpyDeviceToDevice(dst_ptr, src_ptr, size, stream);
}

// Smart tensor copy that automatically determines copy direction
int CudaMemcpyTensorSmart(torch::Tensor& dst_tensor, const torch::Tensor& src_tensor, bool non_blocking) {
    bool src_is_cuda = src_tensor.is_cuda();
    bool dst_is_cuda = dst_tensor.is_cuda();
    
    if (!src_is_cuda && dst_is_cuda) {
        // Host to Device
        return CudaMemcpyTensorHostToDevice(dst_tensor, src_tensor, non_blocking);
    } else if (src_is_cuda && !dst_is_cuda) {
        // Device to Host
        return CudaMemcpyTensorDeviceToHost(dst_tensor, src_tensor, non_blocking);
    } else if (src_is_cuda && dst_is_cuda) {
        // Device to Device
        return CudaMemcpyTensorDeviceToDevice(dst_tensor, src_tensor, non_blocking);
    } else {
        // Host to Host - use regular tensor copy
        dst_tensor.copy_(src_tensor);
        return 0;
    }
}
