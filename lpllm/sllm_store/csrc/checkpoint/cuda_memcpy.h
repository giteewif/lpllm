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

#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

// Basic CUDA memory copy functions
int CudaMemcpyHostToDevice(void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);
int CudaMemcpyDeviceToHost(void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);
int CudaMemcpyDeviceToDevice(void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);

// Tensor-based CUDA memory copy functions
int CudaMemcpyTensorHostToDevice(torch::Tensor& dst_tensor, const torch::Tensor& src_tensor, bool non_blocking = false);
int CudaMemcpyTensorDeviceToHost(torch::Tensor& dst_tensor, const torch::Tensor& src_tensor, bool non_blocking = false);
int CudaMemcpyTensorDeviceToDevice(torch::Tensor& dst_tensor, const torch::Tensor& src_tensor, bool non_blocking = false);

// Smart tensor copy that automatically determines copy direction
int CudaMemcpyTensorSmart(torch::Tensor& dst_tensor, const torch::Tensor& src_tensor, bool non_blocking = false);
