import torch
import threading
import uuid
import ctypes
from typing import Dict, List, Optional, Tuple, Any
from sllm_store.client import SllmStoreClient
from lpllm.logger import init_logger
from lpllm.cuda_memcpy_utils import cuda_copy_, safe_copy_

logger = init_logger(__name__)


class ServerPinnedMemoryPool:
    """
    A memory pool that allocates a large contiguous pinned memory block from server
    Client manages local allocation within this large block
    Compatible with sllm-store's PinnedMemoryPool approach but implemented in Python
    Maintains the same interface as PinnedMemoryPool
    """
    
    def __init__(self,
                 dtype: torch.dtype = torch.bfloat16,
                 pool_size: int = 2,
                 device: str = "cuda:0",
                 client: Optional[SllmStoreClient] = None):
        self.dtype = dtype
        self.pool_size_bytes = pool_size*1024*1024*1024  # GB - size of the large block to allocate
        self.device = device
        
        # Extract device index
        if device.startswith("cuda"):
            self.device_index = int(device.split(":")[1])
        else:
            raise ValueError(f"Unsupported device {device}")
        
        # Initialize client
        if client is None:
            client = SllmStoreClient("127.0.0.1:8073")
        self.client = client
        
        # Generate unique client ID for tracking allocations
        self.client_id = str(uuid.uuid4())
        
        # Calculate memory parameters
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        
        # Memory management data structures
        self.lock = threading.RLock()
        self.allocations = {}  # Maps local block ID -> allocation info
        self.next_block_id = 0
        
        # Server pool configuration - will be set from server config
        self.chunk_size = None
        
        # Server memory allocation info
        self.server_allocation_info = None  # Server allocation info
        self.server_memory_size = 0  # Total size in bytes from server
        self.free_blocks = []  # List of (start_offset, size) tuples
        self.allocated_blocks = {}  # Maps block_id -> (start_offset, size)
        
        # Initialize connection and allocate memory from server

        self._allocate_memory_from_server()
        

    def _allocate_memory_from_server(self):
        """Allocate memory from server's pre-allocated block"""
        try:
            # Request memory allocation from server
            response = self.client.allocate_pool_memory(
                client_id=self.client_id,
                device_id=self.device_index,
                size_mb=self.pool_size_bytes//(1024*1024)
            )
            
            if not response:
                raise RuntimeError("Failed to allocate memory from server")
            
            # Store server allocation info with shared memory handle
            self.server_memory_size = response.size_bytes
            self.server_memory_handle = response.memory_handles[0] if response.memory_handles else None
            self.server_allocation_info = {
                'allocation_id': getattr(response, 'allocation_id', 0),
                'size_mb': response.size_mb,
                'size_bytes': response.size_bytes,
                'memory_handle': self.server_memory_handle,
                'start_offset': response.start_offset
            }
            
            # Create shared memory tensor from handle
            if self.server_memory_handle:
                try:
                    # Convert bytes back to tuple if needed
                    if isinstance(self.server_memory_handle, bytes):
                        handle_str = self.server_memory_handle.decode('utf-8')
                        # Parse the tuple string back to tuple
                        import ast
                        handle_tuple = ast.literal_eval(handle_str)
                    else:
                        handle_tuple = self.server_memory_handle
                    
                    # Create shared memory tensor from tuple handle
                    # handle_tuple[1] is bytes, need to decode to string
                    memory_name = handle_tuple[1].decode('utf-8') if isinstance(handle_tuple[1], bytes) else handle_tuple[1]
                    
                    # Create storage from file with the correct size
                    storage = torch.UntypedStorage.from_file(
                        memory_name,  # Use the name from tuple
                        shared=True,
                        nbytes=self.server_memory_size
                    )
                    # Create tensor from storage as uint8 (raw bytes)
                    self.shared_memory_tensor = torch.tensor(storage, dtype=torch.uint8)
                    logger.info(f"Successfully created shared memory tensor from handle: {handle_tuple}")
                    logger.info(f"Shared memory tensor shape: {self.shared_memory_tensor.shape}, "
                              f"size: {self.shared_memory_tensor.numel()} bytes")
                except Exception as e:
                    logger.warning(f"Failed to create shared memory tensor: {e}")
                    # Fallback to local pinned memory
                    self.shared_memory_tensor = torch.empty(
                        self.server_memory_size,
                        dtype=torch.uint8,
                        pin_memory=True
                    )
            else:
                raise RuntimeError("No shared memory handle received from server")
            
            # Initialize free blocks list with the entire allocated block
            self.free_blocks = [(0, self.server_memory_size)]
            
            logger.info(f"Successfully allocated {response.size_mb}MB from server's pre-allocated memory "
                       f"for client {self.client_id}, memory_ptr={response.memory_ptr}")
            
        except Exception as e:
            logger.error(f"Failed to allocate memory from server: {e}")
            raise RuntimeError(f"Failed to allocate memory from server: {e}")
    
    def _create_tensor_from_shared_memory(self, start_offset: int, num_elements: int, shape: tuple):
        """Create tensor from shared memory"""
        try:
            # Calculate the slice in shared memory
            element_size = self.element_size
            total_bytes = num_elements * element_size
            end_offset = start_offset + total_bytes
            
            logger.debug(f"Creating tensor from shared memory: offset={start_offset}, "
                        f"end_offset={end_offset}, total_bytes={total_bytes}, "
                        f"shared_memory_size={self.shared_memory_tensor.numel()}")
            
            # Check bounds
            if end_offset > self.shared_memory_tensor.numel():
                raise ValueError(f"End offset {end_offset} exceeds shared memory size {self.shared_memory_tensor.numel()}")
            
            # Create tensor from shared memory slice
            shared_slice = self.shared_memory_tensor[start_offset:end_offset]
            logger.debug(f"Shared slice shape: {shared_slice.shape}, size: {shared_slice.numel()}")
            
            # Convert from uint8 bytes to target dtype
            if self.dtype == torch.uint8:
                tensor = shared_slice.view(shape)
            else:
                # Convert bytes to target dtype
                tensor = shared_slice.view(self.dtype).view(shape)
            
            logger.debug(f"Created tensor from shared memory at offset={start_offset}, "
                        f"elements={num_elements}, shape={shape}, dtype={self.dtype}")
            
            return tensor
        except Exception as e:
            logger.error(f"Failed to create tensor from shared memory: {e}")
            # Fallback to local allocation
            return torch.empty(shape, dtype=self.dtype, pin_memory=True)
    
    def alloc_same_pin_tensor(self, tensor: torch.Tensor):
        """Allocate memory block matching given tensor shape and type, return tensor directly"""
        # Direct allocation by element count to avoid KB alignment issues
        required_elements = tensor.numel()
        required_bytes = required_elements * self.element_size
        
        with self.lock:
            # Find a suitable free block using first-fit algorithm
            suitable_block_idx = None
            for i, (start_offset, block_size) in enumerate(self.free_blocks):
                if block_size >= required_bytes:
                    suitable_block_idx = i
                    break
            
            if suitable_block_idx is None:
                raise MemoryError(f"No free block large enough for {required_elements} elements allocation")
            
            # Get the suitable block
            start_offset, block_size = self.free_blocks[suitable_block_idx]
            
            # Create allocation info
            block_id = self.next_block_id
            self.next_block_id += 1
            
            # Update free blocks
            if block_size == required_bytes:
                # Perfect match, remove this free block
                self.free_blocks.pop(suitable_block_idx)
            else:
                # Partial use, update free block start and size
                self.free_blocks[suitable_block_idx] = (start_offset + required_bytes, block_size - required_bytes)
            
            # Record allocated block
            self.allocated_blocks[block_id] = (start_offset, required_bytes)
            
            # Create tensor from shared memory
            try:
                # Create tensor from shared memory
                allocated_tensor = self._create_tensor_from_shared_memory(
                    start_offset, required_elements, tensor.shape
                )
                
                # Store tensor info for potential cleanup
                allocated_tensor._server_pool_info = {
                    'block_id': block_id,
                    'start_offset': start_offset,
                    'size_bytes': required_bytes,
                    'parent_pool': self,
                    'server_allocation_id': self.server_allocation_info['allocation_id']
                }
                
                logger.debug(f"Allocated tensor from shared memory "
                           f"offset={start_offset}, elements={required_elements}")
                
                return allocated_tensor
                
            except Exception as e:
                logger.error(f"Failed to create tensor from shared memory: {e}")
                # Fallback to local pinned memory
                return torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    
   
    
    def free(self, memory_block):
        """Free previously allocated memory block"""
        if memory_block is None:
            return
        
        with self.lock:
            # Handle different types of memory blocks
            if hasattr(memory_block, '_server_pool_info'):
                # Direct tensor with server pool info
                block_id = memory_block._server_pool_info['block_id']
                start_offset = memory_block._server_pool_info['start_offset']
                size_bytes = memory_block._server_pool_info['size_bytes']
            else:
                # For compatibility, ignore non-server blocks
                logger.warning("Attempting to free non-server memory block, ignoring")
                return
            
            if block_id not in self.allocated_blocks:
                logger.warning(f"Block {block_id} not found in allocated blocks")
                return
            
            # Add the block back to free blocks
            self.free_blocks.append((start_offset, size_bytes))
            
            # Sort free blocks by start offset for easier merging
            self.free_blocks.sort(key=lambda x: x[0])
            
            # Merge adjacent free blocks
            self._merge_adjacent_free_blocks()
            
            # Remove from tracking
            del self.allocated_blocks[block_id]
            
            logger.debug(f"Freed memory block {block_id} at offset {start_offset}, "
                        f"size {size_bytes} bytes, {len(self.free_blocks)} free blocks")
    
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
    
    def reshape(self, memory_block, new_shape):
        """Reshape memory block to new shape, no memory copy"""
        return memory_block.view(*new_shape)
    
    def get_usage_info(self):
        """Get memory pool usage information"""
        with self.lock:
            num_allocations = len(self.allocations)
            total_allocated_bytes = sum(
                info['size_bytes'] for info in self.allocations.values()
            )
            total_free_bytes = sum(size for _, size in self.free_blocks)
            
            return {
                "total_mb": self.pool_size_bytes//(1024*1024),
                "active_allocations": num_allocations,
                "allocated_bytes": total_allocated_bytes,
                "free_bytes": total_free_bytes,
                "utilization": total_allocated_bytes / self.server_memory_size if self.server_memory_size > 0 else 0,
                "client_id": self.client_id
            }
    
    def __del__(self):
        """Cleanup resources when object is destroyed"""
        try:
            # Clear local allocations (no need to free individually)
            if hasattr(self, 'lock'):
                with self.lock:
                    if hasattr(self, 'allocations'):
                        self.allocations.clear()
                    if hasattr(self, 'allocated_blocks'):
                        self.allocated_blocks.clear()
                    if hasattr(self, 'free_blocks'):
                        self.free_blocks.clear()
            
            # Free server memory allocation
            if hasattr(self, 'client') and hasattr(self, 'client_id'):
                try:
                    self.client.free_pool_memory(self.client_id)
                    logger.info(f"Freed server memory allocation for client {self.client_id}")
                except Exception as e:
                    logger.warning(f"Failed to free server memory for client {self.client_id}: {e}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
