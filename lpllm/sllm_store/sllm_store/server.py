import asyncio
import grpc
import sllm_store
from sllm_store.proto import storage_pb2, storage_pb2_grpc
from sllm_store.logger import init_logger

# this is necessary to avoid libtorch.so not found error
import torch  # noqa: F401

import ctypes
import os

ctypes.CDLL(os.path.join(sllm_store.__path__[0], "libglog.so"))

from sllm_store._checkpoint_store import (  # noqa: E402
    CheckpointStore,
    MemCopyChunk,
)
import torch
import threading

logger = init_logger(__name__)


class StorageServicer(storage_pb2_grpc.StorageServicer):
    def __init__(
        self,
        storage_path,
        mem_pool_size,
        num_thread,
        chunk_size,
        registration_required,
        pool_memory_size=4*1024*1024*1024,  # Default 4GB for pool memory
    ):
        if not storage_path:
            logger.error("storage_path is empty")
            raise ValueError("storage_path is empty")

        if mem_pool_size <= 0:
            logger.error("mem_pool_size must be greater than 0")
            raise ValueError("Invalid mem_pool_size")

        logger.info(
            f"StorageServicer: storage_path={storage_path}, "
            f"mem_pool_size={mem_pool_size}, num_thread={num_thread}, "
            f"chunk_size={chunk_size}, "
            f"registration_required={registration_required}, "
            f"pool_memory_size={pool_memory_size}"
        )

        self.storage = CheckpointStore(
            storage_path, mem_pool_size, num_thread, chunk_size
        )
        self.registration_required = registration_required
        
        # Initialize pre-allocated large contiguous pinned memory
        self.pool_memory_size = pool_memory_size  # in MB
        self.pre_allocated_memory = None  # Pre-allocated large pinned memory block
        self.memory_lock = threading.RLock()  # Thread safety for memory operations
        
        # Client memory allocation tracking
        self.client_allocations = {}  # client_id -> allocation info
        self.free_blocks = []  # List of (start_offset, size) tuples for free memory
        self.allocated_blocks = {}  # Maps allocation_id -> (start_offset, size)
        self.next_allocation_id = 0
        
        # Pre-allocate large contiguous pinned memory
        self._pre_allocate_memory()

    def _pre_allocate_memory(self):
        """Pre-allocate a large contiguous shared pinned memory block on server startup"""
        try:
            # Calculate total size in bytes
            total_bytes = self.pool_memory_size
            
            # Create shared pinned memory tensor on CPU
            self.pre_allocated_memory = torch.empty(
                total_bytes,
                dtype=torch.uint8,  # Use uint8 for raw bytes
                device="cpu",  # Explicitly allocate on CPU,
                pin_memory=True
            )
           
            # Verify it's on CPU
            if self.pre_allocated_memory.device.type != 'cpu':
                raise RuntimeError(f"Memory allocated on {self.pre_allocated_memory.device}, expected CPU")
            
            # Get the shared memory handle
            try:
                # Try new method first
                self.shared_memory_handle = self.pre_allocated_memory.untyped_storage()._share_filename_()
            except AttributeError:
                try:
                    # Fallback to old method
                    self.shared_memory_handle = self.pre_allocated_memory.storage()._share_filename_()
                except AttributeError:
                    # Fallback to CPU method
                    self.shared_memory_handle = self.pre_allocated_memory.storage()._share_filename_cpu_()
            
            # Initialize free blocks list with the entire block
            self.free_blocks = [(0, total_bytes)]
            
            logger.info(f"Pre-allocated {self.pool_memory_size}MB shared pinned memory block "
                       f"({total_bytes} bytes) with handle: {self.shared_memory_handle}")
            
        except Exception as e:
            logger.error(f"Failed to pre-allocate shared pinned memory: {e}")
            raise RuntimeError(f"Failed to pre-allocate shared pinned memory: {e}")

    def _allocate_from_pre_allocated(self, size_bytes: int):
        """Allocate memory from pre-allocated block using first-fit algorithm"""
        with self.memory_lock:
            # Find a suitable free block using first-fit algorithm
            suitable_block_idx = None
            for i, (start_offset, block_size) in enumerate(self.free_blocks):
                if block_size >= size_bytes:
                    suitable_block_idx = i
                    break
            
            if suitable_block_idx is None:
                return None  # No suitable block found
            
            # Get the suitable block
            start_offset, block_size = self.free_blocks[suitable_block_idx]
            
            # Generate allocation ID
            allocation_id = self.next_allocation_id
            self.next_allocation_id += 1
            
            # Record allocation
            self.allocated_blocks[allocation_id] = (start_offset, size_bytes)
            
            # Update free blocks
            if block_size == size_bytes:
                # Exact fit - remove the block
                del self.free_blocks[suitable_block_idx]
            else:
                # Partial fit - update the block
                self.free_blocks[suitable_block_idx] = (start_offset + size_bytes, block_size - size_bytes)
            
            logger.debug(f"Allocated {size_bytes} bytes at offset {start_offset}, "
                        f"allocation_id={allocation_id}, {len(self.free_blocks)} free blocks remaining")
            
            return {
                'allocation_id': allocation_id,
                'start_offset': start_offset,
                'size_bytes': size_bytes
            }

    def _free_from_pre_allocated(self, allocation_id: int):
        """Free memory back to pre-allocated block"""
        with self.memory_lock:
            if allocation_id not in self.allocated_blocks:
                return False
            
            start_offset, size_bytes = self.allocated_blocks[allocation_id]
            
            # Add the block back to free blocks
            self.free_blocks.append((start_offset, size_bytes))
            
            # Sort free blocks by start offset for easier merging
            self.free_blocks.sort(key=lambda x: x[0])
            
            # Merge adjacent free blocks
            self._merge_adjacent_free_blocks()
            
            # Remove from allocated blocks
            del self.allocated_blocks[allocation_id]
            
            logger.debug(f"Freed {size_bytes} bytes at offset {start_offset}, "
                        f"allocation_id={allocation_id}, {len(self.free_blocks)} free blocks")
            
            return True

    def _merge_adjacent_free_blocks(self):
        """Merge adjacent free blocks to reduce fragmentation"""
        if len(self.free_blocks) <= 1:
            return
        
        merged_blocks = []
        current_start, current_size = self.free_blocks[0]
        
        for i in range(1, len(self.free_blocks)):
            next_start, next_size = self.free_blocks[i]
            
            # Check if blocks are adjacent
            if current_start + current_size == next_start:
                # Merge blocks
                current_size += next_size
            else:
                # Not adjacent, add current block and start new one
                merged_blocks.append((current_start, current_size))
                current_start, current_size = next_start, next_size
        
        # Add the last block
        merged_blocks.append((current_start, current_size))
        
        self.free_blocks = merged_blocks

    async def LoadModelAsync(self, request, context):
        model_path = request.model_path
        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.LoadModelResponse()

        if not self.registration_required:
            model_size = self.storage.register_model_info(model_path)
            if model_size < 0:
                logger.error("RegisterModel failed")
                context.set_code(grpc.StatusCode.INTERNAL)
                return storage_pb2.LoadModelResponse()

        device_type = request.target_device_type
        if device_type == storage_pb2.DEVICE_TYPE_CPU:
            ret = self.storage.load_model_from_disk_async(model_path)
        elif device_type == storage_pb2.DEVICE_TYPE_GPU:
            replica_uuid = request.replica_uuid
            if not replica_uuid:
                logger.error("replica_uuid is empty")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return storage_pb2.LoadModelResponse()

            gpu_memory_handles = {
                device_uuid: [
                    handle.cuda_ipc_handle for handle in handle_list.handles
                ]
                for device_uuid, handle_list in request.handles.items()
            }

            def create_mem_copy_chunk(chunk):
                mem_copy_chunk = MemCopyChunk()
                mem_copy_chunk.src_offset = chunk.src_offset
                mem_copy_chunk.size = chunk.size
                mem_copy_chunk.dst_offset = chunk.dst_offset
                mem_copy_chunk.handle_idx = chunk.handle_idx
                return mem_copy_chunk

            mem_copy_chunks = {
                device_uuid: [
                    create_mem_copy_chunk(chunk) for chunk in chunk_list.chunks
                ]
                for device_uuid, chunk_list in request.chunks.items()
            }
            # logger.debug(
            #     f"LoadModelAsync: {model_path}, {replica_uuid}, "
            #     f"{gpu_memory_handles}, {mem_copy_chunks}"
            # )
            ret = self.storage.load_model_from_mem_async(
                model_path, replica_uuid, gpu_memory_handles, mem_copy_chunks
            )
        else:
            logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.LoadModelResponse()

        if ret != 0:
            logger.error("LoadModel failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            return storage_pb2.LoadModelResponse()

        logger.info(
            f"LoadModel: success {model_path} with target {device_type}"
        )
        return storage_pb2.LoadModelResponse(model_path=model_path)

    async def ConfirmModel(self, request, context):
        model_path = request.model_path
        replica_uuid = request.replica_uuid
        device_type = request.target_device_type

        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.ConfirmModelResponse()

        if device_type != storage_pb2.DEVICE_TYPE_GPU:
            logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.ConfirmModelResponse()

        for i in range(5):
            ret = self.storage.wait_model_in_gpu(model_path, replica_uuid)
            if ret == 0:
                logger.info(
                    f"Confirm model {model_path} replica {replica_uuid} success"
                )
                return storage_pb2.ConfirmModelResponse(model_path=model_path)
            logger.info(f"Confirm model failed, retry {i + 1}")

            await asyncio.sleep(0.05)

        logger.error(
            f"Confirm model {model_path} replica {replica_uuid} failed"
        )
        context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.ConfirmModelResponse()

    async def UnloadModel(self, request, context):
        model_path = request.model_path
        device_type = request.target_device_type

        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.UnloadModelResponse()

        if device_type != storage_pb2.DEVICE_TYPE_CPU:
            logger.error(f"Unsupported device type: {device_type}")
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return storage_pb2.UnloadModelResponse()

        for i in range(5):
            ret = self.storage.unload_model_from_host(model_path)
            if ret == 0:
                logger.info(f"UnloadModel: success {model_path}")
                return storage_pb2.UnloadModelResponse(model_path=model_path)
            logger.info(f"UnloadModel failed, retry {i + 1}")

            await asyncio.sleep(0.01)

        logger.error(f"UnloadModel failed for model {model_path}")
        context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.UnloadModelResponse()

    async def ClearMem(self, request, context):
        ret = self.storage.clear_mem()
        if ret != 0:
            logger.error("ClearMem failed")
            context.set_code(grpc.StatusCode.INTERNAL)
        return storage_pb2.ClearMemResponse()

    async def RegisterModel(self, request, context):
        model_path = request.model_path
        if not model_path:
            logger.error("model_path is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.RegisterModelResponse()

        model_size = self.storage.register_model_info(model_path)
        if model_size < 0:
            logger.error("RegisterModel failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            return storage_pb2.RegisterModelResponse()

        return storage_pb2.RegisterModelResponse(
            model_path=model_path, model_size=model_size
        )

    async def GetServerConfig(self, request, context):
        return storage_pb2.GetServerConfigResponse(
            mem_pool_size=self.storage.get_mem_pool_size(),
            chunk_size=self.storage.get_chunk_size(),
        )

    async def AllocatePoolMemory(self, request, context):
        """Allocate memory from pre-allocated block for client"""
        client_id = request.client_id
        device_id = request.device_id  # Not used for pinned memory, but kept for compatibility
        size_mb = request.size_mb or (self.pool_memory_size // 4)  # Default to 1/4 of total
        size_bytes = size_mb*1024*1024
        
        if not client_id:
            logger.error("client_id is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.AllocatePoolMemoryResponse()
        
        with self.memory_lock:
            if client_id in self.client_allocations:
                logger.warning(f"Client {client_id} already has allocated memory")
                allocation_info = self.client_allocations[client_id]
                return storage_pb2.AllocatePoolMemoryResponse(
                    client_id=client_id,
                    memory_handles=[],  # No handles needed for pinned memory
                    size_mb=allocation_info['size_mb']
                )
            
            try:
                # Calculate size in bytes
                logger.info(f"Allocating {size_mb}MB from pre-allocated memory for client {client_id}")
                # Allocate from pre-allocated memory
                allocation_info = self._allocate_from_pre_allocated(size_bytes)
                
                if allocation_info is None:
                    logger.warning(f"Not enough free memory for client {client_id}, "
                                 f"requested {size_mb}MB")
                    context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                    return storage_pb2.AllocatePoolMemoryResponse()
                
                # Store client allocation info
                self.client_allocations[client_id] = {
                    'allocation_id': allocation_info['allocation_id'],
                    'start_offset': allocation_info['start_offset'],
                    'size_bytes': allocation_info['size_bytes'],
                    'size_mb': size_mb
                }
                
                logger.info(f"Allocated {size_mb}MB from pre-allocated memory for client {client_id} "
                           f"at offset {allocation_info['start_offset']}")
                
                # Return shared memory handle and offset
                # Convert tuple handle to bytes for gRPC
                if isinstance(self.shared_memory_handle, tuple):
                    # Handle is a tuple (socket_path, name, size), convert to bytes
                    handle_bytes = str(self.shared_memory_handle).encode('utf-8')
                else:
                    handle_bytes = self.shared_memory_handle
                
                return storage_pb2.AllocatePoolMemoryResponse(
                    client_id=client_id,
                    memory_handles=[handle_bytes],  # Shared memory handle as bytes
                    size_mb=size_mb,
                    memory_ptr=0,  # Not used for shared memory
                    start_offset=allocation_info['start_offset'],
                    size_bytes=allocation_info['size_bytes']
                )
            except Exception as e:
                logger.error(f"Failed to allocate memory for client {client_id}: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                return storage_pb2.AllocatePoolMemoryResponse()

    async def FreePoolMemory(self, request, context):
        """Free memory back to pre-allocated block for client"""
        client_id = request.client_id
        
        if not client_id:
            logger.error("client_id is empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return storage_pb2.FreePoolMemoryResponse()
        
        with self.memory_lock:
            if client_id not in self.client_allocations:
                logger.warning(f"Client {client_id} has no allocated memory")
                return storage_pb2.FreePoolMemoryResponse(success=False)
            
            try:
                # Get allocation info
                allocation_info = self.client_allocations[client_id]
                allocation_id = allocation_info['allocation_id']
                
                # Free from pre-allocated memory
                success = self._free_from_pre_allocated(allocation_id)
                
                if success:
                    # Remove client allocation
                    del self.client_allocations[client_id]
                    logger.info(f"Freed memory for client {client_id}")
                    return storage_pb2.FreePoolMemoryResponse(success=True)
                else:
                    logger.error(f"Failed to free memory for client {client_id}")
                    return storage_pb2.FreePoolMemoryResponse(success=False)
            except Exception as e:
                logger.error(f"Failed to free memory for client {client_id}: {e}")
                context.set_code(grpc.StatusCode.INTERNAL)
                return storage_pb2.FreePoolMemoryResponse(success=False)

async def serve(
    host,
    port,
    storage_path,
    num_thread,
    chunk_size,
    mem_pool_size,
    registration_required,
    pool_memory_size=2048,  # Default 2GB for pool memory
):
    server = grpc.aio.server()
    storage_pb2_grpc.add_StorageServicer_to_server(
        StorageServicer(
            storage_path,
            mem_pool_size,
            num_thread,
            chunk_size,
            registration_required,
            pool_memory_size,
        ),
        server,
    )
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()

    try:
        await server.wait_for_termination()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down gRPC server")
        await server.stop(5)
