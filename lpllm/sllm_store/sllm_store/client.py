# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

import grpc
import sllm_store.proto.storage_pb2 as storage_pb2
import sllm_store.proto.storage_pb2_grpc as storage_pb2_grpc
from sllm_store.logger import init_logger

logger = init_logger(__name__)


# This is a singleton class that manages the checkpoint
class SllmStoreClient:
    def __init__(self, server_address="127.0.0.1:8073"):
        self.server_address = server_address
        self.channel = grpc.insecure_channel(server_address)
        self.stub = storage_pb2_grpc.StorageStub(self.channel)
        self.checkpoints_in_gpu = {}

    def __del__(self):
        # TODO: cleanup
        pass

    def load_into_cpu(self, model_path):
        request = storage_pb2.LoadModelRequest(
            model_path=model_path,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_CPU,
        )
        try:
            response = self.stub.LoadModelAsync(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.error(f"Model not loaded {e}")
                return False
            else:
                logger.error(f"Error: {e}")
                return False
        else:
            return response

    def unload_from_cpu(self, model_path):
        request = storage_pb2.UnloadModelRequest(
            model_path=model_path,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_CPU,
        )
        try:
            response = self.stub.UnloadModel(request)
        except grpc.RpcError as e:
            logger.error(f"Error: {e}")
            return False
        else:
            return response

    def load_into_gpu(
        self, model_path, replica_uuid, tensor_copy_chunks, cuda_memory_handles
    ):
        logger.debug(f"load_into_gpu: {model_path}, {replica_uuid}")

        gpu_chunk_map = {}
        for device_uuid, chunks in tensor_copy_chunks.items():
            gpu_chunk_map[device_uuid] = storage_pb2.MemCopyChunkList(
                chunks=[
                    storage_pb2.MemCopyChunk(
                        src_offset=chunk[0],
                        size=chunk[1],
                        dst_offset=chunk[2],
                        handle_idx=chunk[3],
                    )
                    for chunk in chunks
                ]
            )
        cuda_handle_map = {}
        for device_uuid, handles in cuda_memory_handles.items():
            cuda_handle_map[device_uuid] = storage_pb2.MemCopyHandleList(
                handles=[
                    storage_pb2.MemCopyHandle(
                        cuda_ipc_handle=handle_str,
                    )
                    for handle_str in handles
                ]
            )
        request = storage_pb2.LoadModelRequest(
            model_path=model_path,                
            replica_uuid=replica_uuid,
            chunks=gpu_chunk_map,
            handles=cuda_handle_map,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        try:
            response = self.stub.LoadModelAsync(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.error(f"Model not loaded {e}")
            else:
                logger.error(f"Error: {e}")
            return False
        else:
            logger.info(f"Model loaded: {model_path}, {replica_uuid}")
            return response

    def confirm_model_loaded(self, model_path, replica_uuid):
        logger.info(f"confirm_model_loaded: {model_path}, {replica_uuid}")
        request = storage_pb2.ConfirmModelRequest(
            model_path=model_path,
            replica_uuid=replica_uuid,
            target_device_type=storage_pb2.DeviceType.DEVICE_TYPE_GPU,
        )
        try:
            _ = self.stub.ConfirmModel(request)
            logger.info("Model loaded")
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.error("Model not loaded")
                return False
            else:
                logger.error(f"Error: {e}")
                return False

    def register_model(self, model_path) -> int:
        logger.info(f"register_model: {model_path}")
        request = storage_pb2.RegisterModelRequest(model_path=model_path)
        try:
            response = self.stub.RegisterModel(request)
        except grpc.RpcError as e:
            logger.error(f"Error: {e}")
            return -1
        else:
            logger.info("Model registered")
            return response.model_size

    def get_server_config(self):
        request = storage_pb2.GetServerConfigRequest()
        try:
            response = self.stub.GetServerConfig(request)
        except grpc.RpcError as e:
            logger.error(f"Error: {e}")
            return None
        else:
            return {
                "chunk_size": response.chunk_size,
                "mem_pool_size": response.mem_pool_size,
            }

    def allocate_pool_memory(self, client_id: str, device_id: int, size_mb: int = None):
        """Allocate pool memory from server"""
        request = storage_pb2.AllocatePoolMemoryRequest(
            client_id=client_id,
            device_id=device_id,
            size_mb=size_mb or 0  # Server will use default if 0
        )
        try:
            response = self.stub.AllocatePoolMemory(request)
            logger.info(f"Allocated pool memory for client {client_id}: {response.size_mb}MB")
            return response
        except grpc.RpcError as e:
            logger.error(f"Failed to allocate pool memory: {e}")
            return None

    def free_pool_memory(self, client_id: str):
        """Free pool memory for client"""
        request = storage_pb2.FreePoolMemoryRequest(client_id=client_id)
        try:
            response = self.stub.FreePoolMemory(request)
            if response.success:
                logger.info(f"Freed pool memory for client {client_id}")
            else:
                logger.warning(f"Failed to free pool memory for client {client_id}")
            return response.success
        except grpc.RpcError as e:
            logger.error(f"Error freeing pool memory: {e}")
            return False

    def allocate_from_pool(self, client_id: str, size_kb: int):
        """Allocate a memory block from client's pool using chunk-based allocation"""
        request = storage_pb2.AllocateFromPoolRequest(
            client_id=client_id,
            size_kb=size_kb
        )
        try:
            response = self.stub.AllocateFromPool(request)
            if response.success:
                return {
                    'allocation_id': response.allocation_id,
                    'chunk_indices': list(response.chunk_indices),
                    'chunk_size': response.chunk_size
                }
            else:
                logger.warning(f"Failed to allocate {size_kb}KB from pool for client {client_id}: {response.error_message}")
                return None
        except grpc.RpcError as e:
            logger.error(f"Error allocating from pool: {e}")
            return None

    def free_from_pool(self, client_id: str, allocation_id: int):
        """Free a memory block back to client's pool using allocation ID"""
        request = storage_pb2.FreeFromPoolRequest(
            client_id=client_id,
            allocation_id=allocation_id
        )
        try:
            response = self.stub.FreeFromPool(request)
            if not response.success:
                logger.warning(f"Failed to free allocation {allocation_id} for client {client_id}: {response.error_message}")
            return response.success
        except grpc.RpcError as e:
            logger.error(f"Error freeing from pool: {e}")
            return False