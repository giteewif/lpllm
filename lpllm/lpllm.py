from ast import mod
from filecmp import clear_cache
import logging
from lpllm.lpmodel import LPModuleWrapper
from lpllm.logger import init_logger
from lpllm.tutils import calculate_layer_memory, calculate_device_offset, calculate_model_memory
from lpllm.pinpool import PinnedMemoryPool
from lpllm.server_pool import ServerPinnedMemoryPool
from lpllm.cuda_memcpy_utils import cuda_copy_, safe_copy_
from transformers.cache_utils import Cache, DynamicCache
import json
import importlib
import concurrent
import os
import torch
from typing import Optional, Union
import uuid
from transformers import AutoConfig
from queue import Queue, Empty
import time
import threading
import signal
import atexit
import sys
# from collections import deque as Queue
from dataclasses import dataclass
from lpllm.tutils import get_thread_id
from sllm_store.client import SllmStoreClient
from sllm_store._C import (
    allocate_cuda_memory,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors,
    restore_tensors2,
    free_cuda_memory,
)
from sllm_store.utils import (
    send_module_buffers_to_device,
    set_module_buffer_to_device
)


from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

logger = init_logger(__name__)

def log_cuda_memory_usage(device, step_name="", step_num=None):
    """Log CUDA memory usage for debugging"""
    return 0, 0, 0
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        used_mem = (torch.cuda.memory_allocated(device) + torch.cuda.memory_reserved(device)) / 1024**3  # GB
        step_info = f" (step {step_num})" if step_num is not None else ""
        logger.debug(f"Memory {step_name}{step_info}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB, Total={total_mem:.2f}GB, Used={used_mem:.2f}GB")
        return allocated, reserved, max_allocated  

def log_error_with_stack(error_msg, exception, context=None, device=None):
    """记录错误和完整的调用栈信息"""
    import traceback
    
    logger.error(f"{error_msg}: {exception}")
    logger.error(f"Exception type: {type(exception).__name__}")
    
    # 记录完整调用栈
    logger.error("Full traceback:")
    for line in traceback.format_exc().split('\n'):
        logger.error(f"  {line}")
    
    # 记录内存状态
    if device and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        logger.error(f"Memory state: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")
    
    # 记录上下文信息
    if context:
        for key, value in context.items():
            logger.error(f"{key}: {value}")

def process_logits_efficiently(logits, temperature=1.0, top_p=0.9, do_sample=True, device=None):
    """
    Efficiently process logits for token generation with proper memory management.
    
    Args:
        logits: Tensor of shape (batch_size, vocab_size)
        temperature: Temperature for scaling logits
        top_p: Top-p (nucleus) sampling parameter
        do_sample: Whether to use sampling or greedy decoding
        device: Device for memory monitoring
    
    Returns:
        next_tokens: Tensor of shape (batch_size,)
    """
    if device is not None:
        log_cuda_memory_usage(device, "process_logits_start")
    
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    

    
    if do_sample:
        # Apply top-p filtering if needed
        if top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            
            # Compute softmax probabilities
            probs = torch.softmax(sorted_logits, dim=-1)
            
            # Compute cumulative probabilities
            cumulative_probs = torch.cumsum(probs, dim=-1)
            
            # Find cutoff point
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Apply mask
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Clean up
            del sorted_logits, sorted_indices, probs, cumulative_probs, sorted_indices_to_remove, indices_to_remove
        
        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        
        # Ensure valid probabilities
        if (probs < 0).any():
            probs = torch.clamp(probs, min=0.0)
        
        # Renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Sample
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        # Clean up
        del probs
    else:
        # Greedy decoding
        next_tokens = torch.argmax(logits, dim=-1)
    
    return next_tokens
def cuda_hook(name):
    torch.cuda.nvtx.range_push(name)
def cuda_hook_end(name):
    torch.cuda.nvtx.range_pop()

class model:
    def __init__(self, model_path: str, device: str, embed_tokens: torch.Tensor, norm: torch.Tensor, lm_head: torch.Tensor):
        self.embed_tokens = embed_tokens
        self.norm = norm
        self.lm_head = lm_head

def _get_uuid():
    return str(uuid.uuid4())

def load_into_gpu_async(
        model_path: Optional[Union[str, os.PathLike]],
        tensor_copy_chunks,
        cuda_memory_handles,
    ):
        client = SllmStoreClient("127.0.0.1:8073")
        device_uuid_map = get_device_uuid_map()
        replica_uuid = _get_uuid()
        ret = client.load_into_gpu(
            model_path,
            replica_uuid,
            {
                device_uuid_map[device_id]: v
                for device_id, v in tensor_copy_chunks.items()
            },
            {
                device_uuid_map[device_id]: [v]
                for device_id, v in cuda_memory_handles.items()
            },
        )
        return ret, replica_uuid
        
class LPLLM():
    def __init__(
            self, 
            model_name: str, 
            device: str ="cuda:0", 
            storage_path: Optional[str] = None,
            pool_size: int = 2,
            chunk_size_mb: int = 1*1024
        ):
        self.model_path = model_name
        self._shutdown = False
        self._threads = []
        
        # 注册信号处理器和退出清理函数
        self._setup_signal_handlers()
        atexit.register(self._cleanup_on_exit)
        
        if not storage_path:
            storage_path = os.getenv("STORAGE_PATH", "./models")

        lpmodel_path = f"lpllm.models.{model_name}.lpmodule"
        lpmodule = importlib.import_module(lpmodel_path)

        lpmodule_class: LPModuleWrapper = getattr(lpmodule, "LPModule")
        self.lpmodule_class = lpmodule_class        

        config = AutoConfig.from_pretrained(
            f"{os.path.join(storage_path, model_name)}", trust_remote_code=True
        )
        self.lpmodule_class_instance = lpmodule_class.get_model(config)
        self.lpmodule_class_instance.to(config.torch_dtype)


        self.config = config
        self.device = device

        client = SllmStoreClient("127.0.0.1:8073")
        ret = client.load_into_cpu(model_name)
        if not ret:
            raise ValueError(f"Failed to load model {model_name} into CPU")
    
        with init_empty_weights():
            # should not be  None
            layer_dense, layer_moe = lpmodule_class.get_layer(self.lpmodule_class_instance)
            layer_dense.to(config.torch_dtype)
            layer_moe.to(config.torch_dtype)
        # decoderlayer has no attribute tie_weights
        # layer_dense.tie_weights()
        self.layer_dense = layer_dense
        # layer_moe.tie_weights()
        self.layer_moe   = layer_moe

        layer_attr_name, layer_loc_index, layer_attn_name, layer_mlp_name = lpmodule_class.get_layer_attr_info()

        # get tensor info
        with open(
            os.path.join(storage_path, model_name, "tensor_index.json"), "r"
        ) as f:
            tensor_index = json.load(f)
            
        # 针对统一文件内容的 offset
        layers_tensor_meta_index = {}
        layers_tensor_data_index = {}
        other_tensor_meta_index = {}
        other_tensor_data_index = {}
        layers_attention_meta_index = {}
        layers_attention_data_index = {}
        layers_mlp_meta_index = {}
        layers_mlp_data_index = {}
        for name, (offset, size, shape, stride, dtype) in tensor_index.items():
            # for layer
            if name.startswith(layer_attr_name):
                ksplits = name.split(".")
                layer_index = int(ksplits[layer_loc_index])
                if layer_index not in layers_tensor_data_index:
                    layers_tensor_data_index[layer_index] = {}
                    layers_tensor_meta_index[layer_index] = {}
                layers_tensor_meta_index[layer_index][name] = (shape, stride, dtype)
                layers_tensor_data_index[layer_index][name] = (offset, size)
            else:
                other_tensor_meta_index[name] = (shape, stride, dtype)
                other_tensor_data_index[name] = (offset, size)

        logger.debug(f"other_tensor_data_index: {other_tensor_data_index}")

        self.layers_tensor_meta_index = layers_tensor_meta_index
        self.layers_tensor_data_index = layers_tensor_data_index

        all_qkv_memory, layer_memory_size = calculate_layer_memory(
            layers_tensor_data_index, layer_attr_name, layer_mlp_name
        )
        logger.debug(f"cuda memory needed for all qkv related {all_qkv_memory}, for single layer {layer_memory_size}")
        if device.startswith("cuda"):
            device_index = int(device.split(":")[1])
        else:
            raise ValueError(f"not support device {device}")
        
        all_help_model_memory = calculate_model_memory(other_tensor_data_index)
        logger.debug(f"cuda memory needed for all model {all_help_model_memory}")

        model_tensor_device_offset = {}
        model_tensor_copy_chunks = {}
        model_tensor_device_offset, model_tensor_copy_chunks, device_offset = calculate_device_offset(
            tensor_index=other_tensor_data_index, device_idx=device_index
        )
        self.model_tensor_device_offset = model_tensor_device_offset
        self.model_tensor_copy_chunks = model_tensor_copy_chunks
        self.model_tensor_meta_index = other_tensor_meta_index
        self.min_device_memory_need = device_offset

        if device_offset > all_help_model_memory:
            raise ValueError(f"max_device_memory {device_offset} > all_model_memory {all_help_model_memory}")
        

        layers_tensor_device_offset = {}
        layers_tensor_copy_chunks = {}
        min_device_memory_need = 0
        for layer_index, layer_tensor_data in layers_tensor_data_index.items():
            layers_tensor_device_offset[layer_index], layers_tensor_copy_chunks[layer_index], device_offset = calculate_device_offset(
                tensor_index=layer_tensor_data, device_idx=device_index
            )      
            if device_offset > min_device_memory_need:
                min_device_memory_need = device_offset
        self.layers_tensor_device_offset=layers_tensor_device_offset
        self.layers_tensor_copy_chunks=layers_tensor_copy_chunks
        self.min_device_memory_need = min_device_memory_need

        if min_device_memory_need > layer_memory_size:
            raise ValueError(f"max_device_memory {min_device_memory_need} > layer_memory_size {layer_memory_size}")

        self.cuda_memory_view = CudaMemoryView(
            memory_num=3, 
            memory_size=layer_memory_size,
            model_memory_size=all_help_model_memory,
            device_index=device_index, 
            client=client
        )

        replica_uuid = self.cuda_memory_view.load_help_model_gpu(model_path=self.model_path, 
            model_tensor_copy_chunks=self.model_tensor_copy_chunks, 
            model_tensor_meta_index=self.model_tensor_meta_index, 
            model_tensor_device_offsets=self.model_tensor_device_offset
        )
        help_model_state_dict = self.cuda_memory_view.get_help_state_dict()
        self.cuda_memory_view.restore_help_model(self.lpmodule_class_instance, help_model_state_dict)

        
        self.cuda_memory_view.wait_help_model_gpu_loading(
            model_path=self.model_path,
            replica_uuid=replica_uuid
        )
        
        # logger.debug(self.lpmodule_class_instance.model.embed_tokens)

        self.attn_manager = AttnManager(lpmodule_class=lpmodule_class, device=device, config=config, 
            pool_size=pool_size, use_server_pool=False, chunk_size_mb=chunk_size_mb
        )
        self.attn_manager.start()
        
        # 记录线程引用
        self._threads.extend([
            self.attn_manager.gpu2cpu_thread.thread,
            self.attn_manager.cpu_compute_thread.thread,
            self.attn_manager.cpu2gpu_thread.thread
        ])
    def load_attention_weight(self):
        pass

    def _setup_signal_handlers(self):
        """设置信号处理器，确保进程异常退出时能清理资源"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating shutdown...")
            self._shutdown = True
            self.stop()
            sys.exit(0)
        
        # 注册常见信号
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
        signal.signal(signal.SIGHUP, signal_handler)   # 挂起信号
        
        # 在Windows上注册额外信号
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _cleanup_on_exit(self):
        """进程退出时的清理函数"""
        if not self._shutdown:
            logger.info("Process exiting, cleaning up resources...")
            self.stop()
    
    def _force_stop_threads(self):
        """强制停止所有线程"""
        logger.warning("Force stopping all threads...")
        
        # 停止注意力管理器
        if hasattr(self, 'attn_manager') and self.attn_manager:
            try:
                self.attn_manager.stop()
            except Exception as e:
                logger.error(f"Error stopping attn_manager: {e}")
        
        # 强制停止所有线程
        for thread in self._threads:
            if thread and thread.is_alive():
                try:
                    # 设置线程为守护线程，这样主进程退出时线程也会退出
                    thread.daemon = True
                    logger.debug(f"Set thread {thread.name} as daemon")
                except Exception as e:
                    logger.error(f"Error setting thread as daemon: {e}")
        
        # 等待线程结束（最多等待5秒）
        for thread in self._threads:
            if thread and thread.is_alive():
                try:
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        logger.warning(f"Thread {thread.name} did not stop gracefully")
                except Exception as e:
                    logger.error(f"Error joining thread {thread.name}: {e}")
    
    def is_shutdown(self):
        """检查是否正在关闭"""
        return self._shutdown
    def prepare_layer(self, layer_idx):
        time_start_prepare = time.time()
        layerc = self.get_model_layer(layer_idx=layer_idx)
        layer_tensor_copy_chunks = self.layers_tensor_copy_chunks[layer_idx]
        layer_tensor_device_offset = self.layers_tensor_device_offset[layer_idx]
        layer_tensor_meta_index = self.layers_tensor_meta_index[layer_idx]
        self.cuda_memory_view.load_layer_gpu(layer_idx=layer_idx, model_path=self.model_path, 
            layer_tensor_copy_chunks=layer_tensor_copy_chunks, layer_tensor_meta_index=layer_tensor_meta_index,
            layer_tensor_device_offsets=layer_tensor_device_offset
        )
        layer_attr_name, layer_loc_index, layer_attn_name, layer_mlp_name = self.lpmodule_class.get_layer_attr_info()
        main_state_dict = self.cuda_memory_view.get_state_dict(layer_idx)

        self.cuda_memory_view.restore_layer(layerc, main_state_dict, layer_loc_index+1)
        self.cuda_memory_view.wait_layer_gpu_loading(layer_idx=layer_idx, model_path=self.model_path)
        logger.debug(f"prepare layer cost {time.time()-time_start_prepare} s")
        return layerc
    def test_other_tensors(self, input_ids):
        input_embeds = self.lpmodule_class_instance.model.embed_tokens(input_ids)
        return input_embeds
    def async_get_decoder_attn(self):
        attn_output: AttnOut = self.attn_manager.wait_task()
        return attn_output.attn_output
    def start_load_into_gpu(self, layer_idx):
        self.cuda_memory_view.load_layer_gpu(layer_idx=layer_idx, model_path=self.model_path, 
            layer_tensor_copy_chunks=self.layers_tensor_copy_chunks[layer_idx], layer_tensor_meta_index=self.layers_tensor_meta_index[layer_idx],
            layer_tensor_device_offsets=self.layers_tensor_device_offset[layer_idx]
        )
    def get_model_layer(self, layer_idx):
        return self.lpmodule_class.get_model_layer(self.lpmodule_class_instance, layer_idx)
    def wait_load_into_gpu(self, layer_idx):
        self.cuda_memory_view.wait_layer_gpu_loading(layer_idx=layer_idx, model_path=self.model_path)

    @torch.no_grad()
    def decoder_qkv(
        self,
        layer,
        hidden_states,
        past_key_value,
        attention_mask,
        position_ids,
        if_batch=False,
    ):
        if if_batch:
            (query_states, key_states, value_states, sin, cos) = self.lpmodule_class.decoder_qkv_batch(
                layer=layer,
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            (query_states, key_states, value_states, sin, cos) = self.lpmodule_class.decoder_qkv(
                layer=layer,
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return (query_states, key_states, value_states, sin, cos)
    @torch.no_grad()
    def decoder_mlp(
        self,
        mlp_layer,
        mlp_hidden_states,
        mlp_o_hidden_states,
        attn_func,
    ):
        mlp_output = self.lpmodule_class.decoder_mlp(
            layer=mlp_layer,
            mlp_hidden_states=mlp_hidden_states,
            mlp_o_hidden_states=mlp_o_hidden_states,
            attn_func=attn_func,
        )
        return mlp_output

    def reset_next_layer_need(
        self,
        layer_attn_idx,
        layer_mlp_idx,
        j_loc,
    ):
        layer_attn = self.get_model_layer(layer_idx=layer_attn_idx)
        layer_mlp = self.get_model_layer(layer_idx=layer_mlp_idx)
        layer_attr_name, layer_loc_index, layer_attn_name, layer_mlp_name = self.lpmodule_class.get_layer_attr_info()
        # the last one in queue
        layer_attr_loc = layer_loc_index + 1
        # all related layer should be ready
        time_start_restore = time.time()
        if j_loc % 2 == 1:
            # change attn, mlp not change, could get from class
            logger.debug(f"reset update attn")
            update_str_list = [f"{layer_attn_name}", "post_attention_layernorm", "input_layernorm"]
            update_state_dict = self.cuda_memory_view.updateState(layer_idx=layer_attn_idx, update_name_list=update_str_list, if_mlp=False)
            self.cuda_memory_view.restore_layer(layer_attn, update_state_dict, layer_attr_loc=layer_attr_loc)
        else:
            logger.debug(f"reset update experts")
            if layer_mlp_idx == 0 :
                return layer_attn, layer_mlp
            # change mlp, attn not change, could get from class
            update_str_list = [f"{layer_mlp_name}"]
            update_state_dict = self.cuda_memory_view.updateState(layer_idx=layer_mlp_idx, update_name_list=update_str_list, if_mlp=True)

            self.cuda_memory_view.restore_layer(layer_mlp, update_state_dict, layer_attr_loc=layer_attr_loc)
        logger.debug(f"restore layer cost {time.time()-time_start_restore} s")
        logger.debug(f"reset_next_layer_need: layer_attn_idx={layer_attn_idx}, layer_mlp_idx={layer_mlp_idx}, j_loc={j_loc}")
        return layer_attn, layer_mlp

        

    def sync(self):
        torch.cuda.synchronize(device=self.device)
        pass
    def decoder_attn_call(self, layer, layer_idx, hidden_states, past_key_value, attention_mask, position_ids,
        query_states, key_states, value_states, sin, cos, if_batch
    ):
        bsz, q_len = hidden_states.shape[:2]
        
        # 决定并行度：可以根据batch size调整
        num_parallel = min(bsz, 10)  # 最多4个并行任务，避免过度拆分
        
        if num_parallel > 1 and bsz % num_parallel == 0 and False:
            # 并行拆分处理
            self.attn_manager.submit_parallel_tasks(
                bsz, q_len, layer, layer_idx, query_states, 
                key_states, value_states, sin, cos, attention_mask, 
                past_key_value, position_ids, num_parallel
            )
        else:
            # 单任务处理（原有逻辑）
            self.attn_manager.submit_task(
                bsz, q_len, layer, layer_idx, query_states, 
                key_states, value_states, sin, cos, attention_mask, 
                past_key_value, position_ids=position_ids, if_batch=if_batch
            )
        return
    
    @torch.no_grad()
    def generate_batch(self, input_ids, past_key_values, encoder_batch_size=360,gpu_batch_size=2, max_new_tokens=32, temperature=1.0, top_p=0.9, 
                 do_sample=False, pad_token_id=None, eos_token_id=None, ):
        try:
            from transformers.modeling_attn_mask_utils import (
                _prepare_4d_causal_attention_mask,
                _prepare_4d_causal_attention_mask_for_sdpa,
            )
            # INSERT_YOUR_CODE
            # 将hidden_states分为gpu_batch_size份，避免拷贝
            # hidden_states: [batch, seq, hidden]
            step = 0
            input_batch_size, input_seq_length = input_ids.shape
            device = input_ids.device

            assert input_batch_size % encoder_batch_size == 0, "batch_size must be divisible by max_batch"
            # INSERT_YOUR_CODE
            # 将input_ids分成平均的max_batch_size
            decoders_batch = encoder_batch_size

            # 再将 decoders_batch 分为两份，每份有gpu_batch_size 小份, 每份的 batch_size 是 chunk_batch_size        
            assert decoders_batch % (gpu_batch_size*2) == 0, "batch_size must be divisible by gpu_batch_size*2"
            chunk_batch_size = decoders_batch // (gpu_batch_size*2)
            
            # position_ids 和 attention_mask 统一使用
            position_ids = None
            attention_mask = None

            embed_tokens, _, _  = self.lpmodule_class.get_embed_tokens_norm_lm_head(self.lpmodule_class_instance)

            #change here, not embeds all
            # inputs_embeds=embed_tokens(input_ids)

            # hidden_states=inputs_embeds
            hidden_states_ids = input_ids
            # hidden_states_decoders_chunks = hidden_states.split(decoders_batch, dim=0)
            hidden_states_ids_decoders_chunks = input_ids.split(decoders_batch, dim=0)

            # orig_shape = hidden_states_decoders_chunks[0].shape
            orig_shape = hidden_states_ids_decoders_chunks[0].shape

            hidden_states_ids_decoders_chunks_list = []
            for i in range(len(hidden_states_ids_decoders_chunks)):
                hidden_states_ids_chunks = hidden_states_ids_decoders_chunks[i].split(chunk_batch_size, dim=0)
                hidden_states_ids_chunks1 = hidden_states_ids_chunks[:gpu_batch_size]
                hidden_states_ids_chunks2 = hidden_states_ids_chunks[gpu_batch_size:]
                
                hidden_states_ids_decoders_chunks_list.append((hidden_states_ids_chunks1, hidden_states_ids_chunks2))

            hidden_states_chunk_ids_example = hidden_states_ids_decoders_chunks_list[0][0][0]
            hidden_states_chunk_example = embed_tokens(hidden_states_chunk_ids_example)
            # 所有batch保持一致
            attention_mask = torch.ones((chunk_batch_size, input_seq_length), dtype=torch.long, device="cpu")
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            _use_sdpa=True
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (chunk_batch_size, input_seq_length),
                hidden_states_chunk_example,
                0,
            )
            logger.debug(f"input_ids shape {input_ids.shape} orig_shape for decoders one chunk: {orig_shape}")
            logger.debug(f"chunk_batch_size: {chunk_batch_size} hidden_states_chunk_example shape and value: {hidden_states_chunk_example.shape}")
            logger.debug(f"attention_mask shape and value: {attention_mask}")
            del hidden_states_chunk_example
            # 定义get_next_token函数，避免重复定义
            norm = self.lpmodule_class_instance.model.norm
            lm_head = self.lpmodule_class_instance.lm_head
            @torch.no_grad()
            def get_next_token(hidden_states):
                """获取下一个token的ID和对应的embedding"""
                max_batch_size = 60
                batch_size_actual = hidden_states.size(0)

                if batch_size_actual <= max_batch_size:
                    # normed_hidden_states = norm(hidden_states)
                    normed_last_hidden_states = norm(hidden_states)
                    # get last 
                    normed_last_hidden_states = normed_last_hidden_states[:, -1:, :]
                else:
                    normed_chunks = []
                    for i in range(0, batch_size_actual, max_batch_size):
                        end_idx = min(i + max_batch_size, batch_size_actual)
                        chunk = hidden_states[i:end_idx]
                        normed_chunk = norm(chunk)
                        norm_last_hidden_states = normed_chunk[:, -1:, :]
                        normed_chunks.append(norm_last_hidden_states)
                        # normed_chunks.append(normed_chunk)
                    # norm_hidden_states = torch.cat(normed_chunks, dim=0)
                    normed_last_hidden_states = torch.cat(normed_chunks, dim=0)
                # hidden_states = normed_hidden_states[:, -1:, :]

                # last_hidden_states = hidden_states[:, -1:, :]  # Shape: (batch_size, 1, hidden_size)
                last_hidden_states = normed_last_hidden_states

                batch_size_actual = last_hidden_states.size(0)
                logger.debug(f"last_hidden_states shape {last_hidden_states.shape}")
                # not need in decoder batch
                max_batch_size = 560
                if batch_size_actual <= max_batch_size:
                    # Small batch, process all at once
                    next_token_logits = lm_head(last_hidden_states).squeeze(1)  # Shape: (batch_size, vocab_size)
                else:
                    # Large batch, process in chunks
                    next_token_logits_list = []
                    for i in range(0, batch_size_actual, max_batch_size):
                        end_idx = min(i + max_batch_size, batch_size_actual)
                        chunk = last_hidden_states[i:end_idx]
                        chunk_logits = lm_head(chunk).squeeze(1)  # Shape: (chunk_size, vocab_size)
                        next_token_logits_list.append(chunk_logits)
                    
                    # Concatenate results
                    next_token_logits = torch.cat(next_token_logits_list, dim=0)
                next_token_logits = next_token_logits.float()
            
                logger.debug(f"next_token_logits shape {next_token_logits.shape}")
                # Use efficient logits processing function
                next_token_ids = process_logits_efficiently(
                    logits=next_token_logits,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    device=device
                )
                
                # 将token IDs转换为embeddings
                next_token_ids = next_token_ids.unsqueeze(1)  # Shape: (batch_size, 1)
                return next_token_ids

            # 初始化存储生成的tokens
            generated_tokens1 = []
            generated_tokens2 = []
            
            # encoder first - 处理每个chunk并立即生成token
            for i in range(len(hidden_states_ids_decoders_chunks_list)):
                hidden_states_ids_chunks1, hidden_states_ids_chunks2 = hidden_states_ids_decoders_chunks_list[i]

                hidden_states_chunks1 = []
                hidden_states_chunks2 = []
                for j in range(len(hidden_states_ids_chunks1)):
                    hidden_states_chunks1_element = embed_tokens(hidden_states_ids_chunks1[j])
                    hidden_states_chunks2_element = embed_tokens(hidden_states_ids_chunks2[j])
                    hidden_states_chunks1.append(hidden_states_chunks1_element)
                    hidden_states_chunks2.append(hidden_states_chunks2_element)

                time_start_decoders_batch = time.time()
                layer_output_chunk = self.decoders_batch(
                    hidden_states_chunks1=hidden_states_chunks1,
                    hidden_states_chunks2=hidden_states_chunks2,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                logger.debug(f"decoders batch cost for {i} cost {time.time() - time_start_decoders_batch} s")

                log_cuda_memory_usage(device=input_ids.device, step_name=f"decoders batch for {i}")
                assert layer_output_chunk.shape[:2] == orig_shape, f"layer_output_chunk shape {layer_output_chunk.shape} does not match input hidden shape in any {orig_shape}"
                
                # 立即处理layer_output_chunk，而不是累积
                layer_output_chunk_batch_size, seq_length = layer_output_chunk.shape[:2]
                assert layer_output_chunk_batch_size % 2 == 0, f"layer_output_chunk_batch_size {layer_output_chunk_batch_size} must be even"
                half = layer_output_chunk_batch_size // 2
                layer_output_chunk1 = layer_output_chunk[:half]
                layer_output_chunk2 = layer_output_chunk[half:]
                assert layer_output_chunk1.shape == layer_output_chunk2.shape, f"layer_output_chunk1 shape {layer_output_chunk1.shape} does not match layer_output_chunk2 shape {layer_output_chunk2.shape}"

                # 立即生成token
                next_token_ids1_chunk = get_next_token(layer_output_chunk1)
                next_token_ids2_chunk = get_next_token(layer_output_chunk2)
                
                # 存储生成的tokens
                generated_tokens1.append(next_token_ids1_chunk)
                generated_tokens2.append(next_token_ids2_chunk)
                
                # 显式删除不再需要的对象以释放显存
                del layer_output_chunk, layer_output_chunk1, layer_output_chunk2
                del hidden_states_chunks1, hidden_states_chunks2
                torch.cuda.empty_cache()
            
            # 整合所有生成的tokens
            next_token_ids1 = torch.cat(generated_tokens1, dim=0)
            next_token_ids2 = torch.cat(generated_tokens2, dim=0)
            logger.debug(f"next_token_ids shape {next_token_ids1.shape}")
            
            # 清空临时存储
            generated_tokens1.clear()
            generated_tokens2.clear()
            
            # 重新初始化用于后续步骤
            generated_tokens1 = [next_token_ids1]
            generated_tokens2 = [next_token_ids2]

            # decoder next
            self.sync()
            torch.cuda.empty_cache()
            log_cuda_memory_usage(device=input_ids.device, step_name="step 0")
            # generated_new_tokens = torch.cat([next_token_ids1, next_token_ids2], dim=0)
            # generate_tokens = torch.cat([input_ids, generated_new_tokens], dim=1)
            # return generate_tokens, generated_new_tokens

            for i in range(max_new_tokens-1):
                # step start from 1
                step = i+1
                logger.debug(f"\n\nstep {step} decoders")

                time_start_pre = time.time()
                current_input1 = next_token_ids1  # 使用embeddings而不是token IDs
                current_input2 = next_token_ids2  # 使用embeddings而不是token IDs

                batch_size, _ = current_input1.shape

                #step start from 1
                generate_seq_len = input_seq_length + step
                attention_mask = torch.ones((batch_size, generate_seq_len), dtype=torch.long, device="cpu")
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

                attention_mask1 = attention_mask
                attention_mask2 = attention_mask
                position_ids1 = position_ids
                position_ids2 = position_ids

                past_key_values1 = past_key_values
                past_key_values2 = past_key_values

                if past_key_values1:
                    position_ids1 = position_ids1[:, -current_input1.shape[1] :]
                if past_key_values2:
                    position_ids2 = position_ids2[:, -current_input2.shape[1] :]
            
                (input_embeds1, attention_mask1,
                past_key_values1, position_ids1) \
                =self.forward_prepare(
                    input_ids=current_input1,
                    inputs_embeds=None,
                    position_ids=position_ids1,
                    past_key_values=past_key_values1,
                    output_attentions=False,
                    use_cache=True,
                    attention_mask=attention_mask1,
                )

                (input_embeds2, attention_mask2,
                past_key_values2, position_ids2) \
                =self.forward_prepare(
                    input_ids=current_input2,
                    inputs_embeds=None,
                    position_ids=position_ids2,
                    past_key_values=past_key_values2,
                    output_attentions=False,
                    use_cache=True,
                    attention_mask=attention_mask2,
                )
                logger.debug(f"decoders non batch pre cost {time.time()-time_start_pre} s")

                time_start_decoders = time.time()
                layer_output1, layer_output2 = self.decoders(
                    hidden_states1=input_embeds1,
                    hidden_states2=input_embeds2,
                    past_key_value1=past_key_values1,
                    past_key_value2=past_key_values2,
                    attention_mask1=attention_mask1,
                    attention_mask2=attention_mask2,
                    position_ids1=position_ids1,
                    position_ids2=position_ids2,
                    step=step,
                    if_batch=True
                )
                logger.debug(f"decoders non batch cost {time.time() - time_start_decoders} s")

                time_start_get_next_token=time.time()
                next_token_ids1 = get_next_token(layer_output1)
                logger.debug(f"get_next_token1 non batch cost {time.time()-time_start_get_next_token} s")
                next_token_ids2 = get_next_token(layer_output2)
                logger.debug(f"get_next_token2 non batch cost {time.time()-time_start_get_next_token} s")

                generated_tokens1.append(next_token_ids1)
                generated_tokens2.append(next_token_ids2)
                
                # 立即清理不需要的tensor以释放显存
                del layer_output1, layer_output2
                del input_embeds1, input_embeds2
                torch.cuda.empty_cache()
            
            # 整合所有生成的tokens
            generated_new_tokens = torch.tensor([], dtype=torch.long, device=input_ids.device)
            for next_token_ids1, next_token_ids2 in zip(generated_tokens1, generated_tokens2):
                generated_new_tokens_step = torch.cat([next_token_ids1, next_token_ids2], dim=0)
                generated_new_tokens = torch.cat([generated_new_tokens, generated_new_tokens_step], dim=1)
                logger.debug(f"input_ids shape {input_ids.shape} generted_new_tokens.shape {generated_new_tokens.shape}")
            generated_tokens = torch.cat([input_ids, generated_new_tokens], dim=1)
            # 清理临时存储
            generated_tokens1.clear()
            generated_tokens2.clear()
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during generation at step {step + 1}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            
            # 添加详细的调用栈信息
            import traceback
            logger.error("Full traceback:")
            for line in traceback.format_exc().split('\n'):
                logger.error(f"  {line}")
            
            # 添加内存状态信息
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
                logger.error(f"Memory state on error: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")
            
            # 添加当前状态信息
            logger.error(f"Current step: {step}")
            logger.error(f"Generated tokens shape: {generated_tokens.shape}")
            logger.error(f"Input shape: {input_ids.shape}")
            logger.error(f"Batch size: {batch_size}")
            raise e

        # Final cleanup
        torch.cuda.empty_cache()
        logger.info(f"Generation completed. Final shape: {generated_tokens.shape} generated_new_tokens {generated_new_tokens.shape}")
        return generated_tokens, generated_new_tokens

    @torch.no_grad()
    def decoders_batch(
        self,
        hidden_states_chunks1,
        hidden_states_chunks2,
        past_key_values,
        attention_mask,
        position_ids,
    ):
        
        num_chunks = len(hidden_states_chunks1)

        logger.debug(f"hidden_states_chunks1 shape: {hidden_states_chunks1[0].shape}, hidden_states_chunks1 length: {len(hidden_states_chunks1)}")
        logger.debug(f"hidden_states_chunks2 shape: {hidden_states_chunks2[0].shape}, hidden_states_chunks2 length: {len(hidden_states_chunks2)}")

        layer_num = self.config.num_hidden_layers

        self.prepare_layer(layer_idx=0)

        cur_layer_idx=0
        layerc = self.get_model_layer(layer_idx=cur_layer_idx)

        time_start_load_gpu = time.time()
        # prepare next layer
        self.start_load_into_gpu(cur_layer_idx+1)

        layer_output = None

        hidden_states_attn_chunks = hidden_states_chunks1

        # use the same attention_mask and position_ids
        for i in range(num_chunks):
            hidden_states_chunk = hidden_states_attn_chunks[i]
            attention_mask_chunk = attention_mask
            position_ids_chunk = position_ids
            
            (query_states, key_states, value_states, sin, cos) = self.decoder_qkv(
                layer=layerc,
                hidden_states=hidden_states_chunk,
                past_key_value=past_key_values,
                attention_mask=attention_mask_chunk,
                position_ids=position_ids_chunk,
                if_batch=True
            )

            attn_func = lambda: self.decoder_attn_call(
                layer=layerc,
                layer_idx=cur_layer_idx,
                hidden_states=hidden_states_chunk,
                past_key_value=past_key_values,
                attention_mask=attention_mask_chunk,
                position_ids=position_ids_chunk,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                sin=sin,
                cos=cos,
                if_batch=True
            )
            attn_func()

        last_attn_input_chunk = hidden_states_attn_chunks
        last_mlp_output_chunk = hidden_states_chunks2

        layer_attn = layerc
        layer_mlp  = layerc
        
        layer_attn_idx = cur_layer_idx
        layer_mlp_idx = cur_layer_idx-1

        j = 0
        while True:
            j+=1
            load_next_layer=False
            need_wait_layer=False

            if j%2==0:
                cur_layer_idx += 1
                layer_attn_idx += 1
                # not True in cur_layer_idx == 0, start load layer_idx=1 at the begining
                if cur_layer_idx < layer_num - 1 and cur_layer_idx >= 1:
                    load_next_layer=True
                need_wait_layer=False
            else: 
                load_next_layer=False
                if cur_layer_idx < layer_num - 1 and cur_layer_idx >= 0:
                    need_wait_layer=True
                layer_mlp_idx += 1
            logger.debug(f"\ndecoder loop j: {j} cur_layer_idx: {cur_layer_idx} layer_attn_idx: {layer_attn_idx} layer_mlp_idx: {layer_mlp_idx}")
        
            # break at the last mlp
            if cur_layer_idx == layer_num and j == 2*layer_num :
                break

            # here layer
            if load_next_layer:
                time_start_load_gpu =time.time()
                logger.debug(f"start load next layer cur_layer_idx: {cur_layer_idx+1}")
                self.start_load_into_gpu(cur_layer_idx+1)
        
            hidden_states_attn_chunk = last_mlp_output_chunk

            qkv_chunk = []
            cuda_hook("decoder_qkv and attn chunk")
            for chunk_idx in range(num_chunks):
                hidden_states_chunk = hidden_states_attn_chunk[chunk_idx]
                (query_states, key_states, value_states, sin, cos) = self.decoder_qkv(
                    layer=layer_attn,
                    hidden_states=hidden_states_chunk,
                    past_key_value=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    if_batch=True
                )
                qkv_chunk.append((query_states, key_states, value_states, sin, cos))
            cuda_hook_end("decoder_qkv and attn chunk")

            last_attn_output_chunk = []
            cuda_hook("async_get_decoder_attn chunk")
            for chunk_idx in range(num_chunks):
                last_attn_output = self.async_get_decoder_attn()
                last_attn_output_chunk.append(last_attn_output)
            cuda_hook_end("async_get_decoder_attn chunk")

            hidden_states_mlp_chunk = last_attn_input_chunk
            hidden_states_mlp_o_chunk = last_attn_output_chunk

            cuda_hook("decoder_attn_call_chunk")
            for chunk_idx in range(num_chunks):
                query_states, key_states, value_states, sin, cos = qkv_chunk[chunk_idx]
                
                hidden_states_attn = hidden_states_attn_chunk[chunk_idx]
                # if_batch flag, control the past_key_value update
                attn_func = lambda: self.decoder_attn_call(
                    layer=layer_attn,
                    layer_idx=layer_attn_idx,
                    hidden_states=hidden_states_attn,
                    past_key_value=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    sin=sin,
                    cos=cos,
                    if_batch=True
                )
                attn_func()
            cuda_hook_end("decoder_attn_call_chunk")

            # check 
            mlp_output_chunk = []
            cuda_hook("decoder_mlp_chunk")
            for chunk_idx in range(num_chunks):
                hidden_states_mlp = hidden_states_mlp_chunk[chunk_idx]
                hidden_states_mlp_o = hidden_states_mlp_o_chunk[chunk_idx]

                if hidden_states_mlp is None:
                    logger.warning(f"hidden_states_mlp at chunk {chunk_idx} is None (空值)")
                if hidden_states_mlp_o is None:
                    logger.warning(f"hidden_states_mlp_o at chunk {chunk_idx} is None (空值)")
                if isinstance(hidden_states_mlp, torch.Tensor):
                    if torch.isnan(hidden_states_mlp).any():
                        logger.warning(f"NaN detected in hidden_states_mlp at chunk {chunk_idx}")
                    if torch.isinf(hidden_states_mlp).any():
                        logger.warning(f"Inf detected in hidden_states_mlp at chunk {chunk_idx}")
                if isinstance(hidden_states_mlp_o, torch.Tensor):
                    if torch.isnan(hidden_states_mlp_o).any():
                        logger.warning(f"NaN detected in hidden_states_mlp_o at chunk {chunk_idx}")
                    if torch.isinf(hidden_states_mlp_o).any():
                        logger.warning(f"Inf detected in hidden_states_mlp_o at chunk {chunk_idx}")

                mlp_output=self.decoder_mlp(
                    mlp_layer=layer_mlp,
                    mlp_hidden_states=hidden_states_mlp,
                    mlp_o_hidden_states=hidden_states_mlp_o,
                    attn_func=None
                )

                if torch.isnan(mlp_output).any():
                        logger.warning(f"NaN detected in hidden_states_mlp at chunk {chunk_idx}")
                if torch.isinf(mlp_output).any():
                    logger.warning(f"Inf detected in hidden_states_mlp at chunk {chunk_idx}")

                mlp_output_chunk.append(mlp_output)
                # self.sync()
            cuda_hook_end("decoder_mlp_chunk")

            last_attn_input_chunk=hidden_states_attn_chunk
            last_mlp_output_chunk=mlp_output_chunk

            # always reset
            if j < 2*layer_num - 1 :
                time_start = time.time()
                if j % 2 == 1:
                    next_layer_attn_idx = layer_attn_idx + 1
                    next_layer_mlp_idx = layer_mlp_idx
                else:
                    next_layer_attn_idx = layer_attn_idx
                    next_layer_mlp_idx = layer_mlp_idx + 1
                layer_attn, layer_mlp = self.reset_next_layer_need(layer_attn_idx=next_layer_attn_idx, layer_mlp_idx=next_layer_mlp_idx, j_loc=j)
                logger.debug(f"reset layer cost {time.time()-time_start} s")
                logger.debug(f"have reset next layer layer_attn {next_layer_attn_idx} layer_mlp {next_layer_mlp_idx} ")

            cuda_hook("wait_load_into_gpu")
            # here layer
            if need_wait_layer:
                # waiting here
                logger.debug(f"j: {j} waiting the layer with layer_idx {cur_layer_idx+1} before wait time {time.time()-time_start_load_gpu} s")
                time_start_waiting = time.time()
                self.wait_load_into_gpu(cur_layer_idx+1)
                self.sync()
                logger.debug(f"j: load cost {time.time()-time_start_load_gpu} s waiting cost {time.time()-time_start_waiting} s")
            cuda_hook_end("wait_load_into_gpu")

            # here layer
            if load_next_layer:
                cuda_hook("free_layer_gpu")
                self.cuda_memory_view.free_layer_gpu(cur_layer_idx-1)
                cuda_hook_end("free_layer_gpu")
    
            # self.sync()
        last_attn_output_chunk = []
        cuda_hook("async_get_decoder_attn_chunk")
        for chunk_idx in range(num_chunks):
            last_attn_output = self.async_get_decoder_attn()
            last_attn_output_chunk.append(last_attn_output)
        cuda_hook_end("async_get_decoder_attn_chunk")

        hidden_states_mlp_chunk = last_attn_input_chunk
        hidden_states_mlp_o_chunk = last_attn_output_chunk
        

        mlp_output_chunk = []
        cuda_hook("decoder_mlp_chunk last layer")
        for chunk_idx in range(num_chunks):
            hidden_states_mlp = hidden_states_mlp_chunk[chunk_idx]
            hidden_states_mlp_o = hidden_states_mlp_o_chunk[chunk_idx]
            mlp_output = self.decoder_mlp(
                mlp_layer=layer_mlp,
                mlp_hidden_states=hidden_states_mlp,
                mlp_o_hidden_states=hidden_states_mlp_o,
                attn_func=None
            )
            mlp_output_chunk.append(mlp_output)
        cuda_hook_end("decoder_mlp_chunk last layer")
        # self.sync()
        self.cuda_memory_view.free_layer_gpu_all()
        
        logger.debug(f"last_mlp_output_chunk shape: {last_mlp_output_chunk[0].shape}, mlp_output_chunk shape: {mlp_output_chunk[0].shape}")
        logger.debug(f"last_mlp_output_chunk len {len(last_mlp_output_chunk)}, mlp_output_chunk len {len(mlp_output_chunk)}")
        layer_output = torch.cat(last_mlp_output_chunk + mlp_output_chunk, dim=0)
        return layer_output

    @torch.no_grad()
    def decoders(
        self,
        hidden_states1,
        hidden_states2,
        past_key_value1,
        past_key_value2,
        attention_mask1,
        attention_mask2,
        position_ids1,
        position_ids2,
        step=0,
        if_batch=False,
    ):
        layer_num = self.config.num_hidden_layers
        
        
        # load first layer
        self.prepare_layer(layer_idx=0)
        # self.prepare_layer(layer_idx=1)
        
        
        # here layer
        cur_layer_idx = 0
        layerc = self.get_model_layer(layer_idx=cur_layer_idx)

        time_start_load_gpu = time.time()
        # prepare next layer
        self.start_load_into_gpu(cur_layer_idx+1)
        
        
        layer_output1 = None
        layer_output2 = None
        
        hidden_states_attn=hidden_states1
        
        past_key_value=past_key_value1
        attention_mask=attention_mask1
        position_ids=position_ids1
        

        

        logger.debug(f"start first layer")

        
        # not need batch, decoder_qkv_batch only for control the kv_seq_length
        (query_states, key_states, value_states, sin, cos) = self.decoder_qkv(
            layer=layerc,
            hidden_states=hidden_states_attn,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # if_batch flag only for control the past_key_value update operation
        _ = self.decoder_attn_call(
            layer=layerc,
            layer_idx=cur_layer_idx,
            hidden_states=hidden_states_attn,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            sin=sin,
            cos=cos,
            if_batch=if_batch,
        )
        
    
        last_attn_input = hidden_states_attn
        
        last_mlp_output = hidden_states2

        layer_attn = layerc
        layer_mlp  = layerc
        layer_attn_idx = cur_layer_idx
        layer_mlp_idx = cur_layer_idx-1
        
        
        j=0
        while True:
            
            j+=1
            
            load_next_layer=False   
            need_wait_layer=False 
            if j%2==0:
                
                past_key_value=past_key_value1
                attention_mask=attention_mask1
                position_ids=position_ids1
                
                cur_layer_idx += 1
                layer_attn_idx += 1
                if cur_layer_idx == layer_num:
                    break
                # not True in cur_layer_idx == 0, start load layer_idx=1 at the begining
                elif cur_layer_idx < layer_num - 1 and cur_layer_idx >= 1:
                    load_next_layer=True
                need_wait_layer=False
            else: 
                load_next_layer=False
                if cur_layer_idx < layer_num - 1 and cur_layer_idx >= 0:
                    need_wait_layer=True
                
                layer_mlp_idx += 1
                past_key_value=past_key_value2
                attention_mask=attention_mask2
                position_ids=position_ids2
            
            logger.debug(f"\none decoder loop j: {j} cur_layer_idx: {cur_layer_idx}")
            
            # here layer
            if load_next_layer:
                time_start_load_gpu =time.time()
                logger.debug(f"start load next layer cur_layer_idx: {cur_layer_idx+1}")
                self.start_load_into_gpu(cur_layer_idx+1)
            logger.debug(f"start decoder qkv layer_attn {layer_attn_idx} layer_mlp {layer_mlp_idx}")

            hidden_states_attn = last_mlp_output

            cuda_hook("decoder_qkv")
            # not need batch, decoder_qkv_batch only for control the kv_seq_length
            (query_states, key_states, value_states, sin, cos) = self.decoder_qkv(
                layer=layer_attn,
                hidden_states=hidden_states_attn,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            cuda_hook_end("decoder_qkv")
            
            cuda_hook("async_get_decoder_attn")
            # get last attn_output in here
            last_attn_output = self.async_get_decoder_attn()
            cuda_hook_end("async_get_decoder_attn")


            

            hidden_states_mlp = last_attn_input
            hidden_states_mlp_o = last_attn_output

            use_dep = True
            if layer_mlp_idx >= 1 and use_dep:
                (hidden_states_inter, tokens_per_expert, topk_weight, token_idxs, idxs, identity), residual = self.lpmodule_class.mlp_prepare(layer_mlp, hidden_states_mlp, hidden_states_mlp_o)

            attn_func = lambda: self.decoder_attn_call(
                layer=layer_attn,
                layer_idx=layer_attn_idx,
                hidden_states=hidden_states_attn,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                position_ids=position_ids,
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                sin=sin,
                cos=cos,
                if_batch=if_batch
            )
            cuda_hook("decoder_attn_call")
            attn_func()
            cuda_hook_end("decoder_attn_call")

            cuda_hook("decoder_mlp")
            if layer_mlp_idx >= 1 and use_dep:
                mlp_output=self.lpmodule_class.decoder_mlp_post(layer_mlp, hidden_states_inter, tokens_per_expert, topk_weight, token_idxs, idxs, identity, residual)
            else:
                mlp_output=self.decoder_mlp(
                    mlp_layer=layer_mlp,
                    mlp_hidden_states=hidden_states_mlp,
                    mlp_o_hidden_states=hidden_states_mlp_o,
                    attn_func=None
                )
            cuda_hook_end("decoder_mlp")            

            last_attn_input=hidden_states_attn
            last_mlp_output=mlp_output

            # here layer
            if not (cur_layer_idx == layer_num - 1 and need_wait_layer == False):
                time_start = time.time()
                if j % 2 == 1:
                    next_layer_attn_idx = layer_attn_idx + 1
                    next_layer_mlp_idx = layer_mlp_idx
                else:
                    next_layer_attn_idx = layer_attn_idx
                    next_layer_mlp_idx = layer_mlp_idx + 1
                layer_attn, layer_mlp = self.reset_next_layer_need(layer_attn_idx=next_layer_attn_idx, layer_mlp_idx=next_layer_mlp_idx, j_loc=j)
                logger.debug(f"reset layer cost {time.time()-time_start} s")
            logger.debug(f"have reset next layer layer_attn {layer_attn_idx} layer_mlp {layer_mlp_idx} ")

            cuda_hook("wait_load_into_gpu")
            # here layer
            if need_wait_layer:
                # waiting here
                logger.debug(f"j: {j} waiting the layer with layer_idx {cur_layer_idx+1} before wait time {time.time()-time_start_load_gpu} s")
                time_start_waiting = time.time()
                self.wait_load_into_gpu(cur_layer_idx+1)
                # self.sync()
                logger.debug(f"j: load cost {time.time()-time_start_load_gpu} s waiting cost {time.time()-time_start_waiting} s")
            cuda_hook_end("wait_load_into_gpu")

            # here layer
            if load_next_layer:
                cuda_hook("free_layer_gpu")
                self.cuda_memory_view.free_layer_gpu(cur_layer_idx-1)
                cuda_hook_end("free_layer_gpu")

            # self.sync()
        last_attn_output = self.async_get_decoder_attn()
        hidden_states_mlp = last_attn_input
        hidden_states_mlp_o = last_attn_output

        logger.debug(f"start decoder mlp last layer")
        mlp_output=self.decoder_mlp(
            mlp_layer=layerc,
            mlp_hidden_states=hidden_states_mlp,
            mlp_o_hidden_states=hidden_states_mlp_o,
            attn_func=None
        )    
      
        self.sync()
        self.cuda_memory_view.free_layer_gpu_all()
        layer_output1=last_mlp_output
        layer_output2=mlp_output
        
        return layer_output1, layer_output2
        
    def forward_prepare(self, input_ids, inputs_embeds, position_ids, past_key_values, output_attentions, use_cache, attention_mask):
        return self.lpmodule_class.forward_prepare(
            model=self.lpmodule_class_instance, input_ids=input_ids, inputs_embeds=inputs_embeds,
            position_ids=position_ids, past_key_values=past_key_values, 
            output_attentions=output_attentions, use_cache=use_cache, attention_mask=attention_mask
        )

    @torch.no_grad
    def split_decoders(self, input_ids, step, past_key_values1, past_key_values2, attention_mask=None, position_ids=None):
        logger.debug(f" split_decoders input_ids shape: {input_ids.shape}")


        # split inputs
        time_start = time.time()
        batch_size = input_ids.size(0)
        if batch_size == 1:
            # For single batch, duplicate the input to maintain the split structure
            raise ValueError(f"batch_size == 1 is not supported")
        
        split_input_ids = input_ids.split(batch_size // 2, dim = 0)
        input_ids1 = split_input_ids[0]
        input_ids2 = split_input_ids[1]

        # split attention_mask
        if attention_mask is not None:
            split_attention_mask = attention_mask.split(batch_size // 2, dim = 0)
            logger.debug(f"Step {step}: split_attention_mask shape: {split_attention_mask[0].shape}, {split_attention_mask[1].shape}")
            attention_mask1 = split_attention_mask[0]
            attention_mask2 = split_attention_mask[1]
        else:
            attention_mask1 = None
            attention_mask2 = None
        if position_ids is not None:
            split_position_ids = position_ids.split(batch_size // 2, dim = 0)
            logger.debug(f"Step {step}: split_position_ids shape: {split_position_ids[0].shape}, {split_position_ids[1].shape}")
            position_ids1 = split_position_ids[0]
            position_ids2 = split_position_ids[1]
        else:
            position_ids1 = None
            position_ids2 = None

        # First step: process the entire input sequence
        (input_embeds1, attention_mask1,
        past_key_values1, position_ids1) \
        =self.forward_prepare(
            input_ids=input_ids1,
            inputs_embeds=None,
            position_ids=position_ids1,
            past_key_values=past_key_values1,
            output_attentions=False,
            use_cache=True,
            attention_mask=attention_mask1,
        )

        (input_embeds2, attention_mask2,
        past_key_values2, position_ids2) \
        =self.forward_prepare(
            input_ids=input_ids2,
            inputs_embeds=None,
            position_ids=position_ids2,
            past_key_values=past_key_values2,
            output_attentions=False,
            use_cache=True,
            attention_mask=attention_mask2,
        )

        return input_embeds1, input_embeds2, past_key_values1, past_key_values2, attention_mask1, attention_mask2, position_ids1, position_ids2
        
        
    
    @torch.no_grad
    def generate(self, input_ids, past_key_values1, past_key_values2, max_new_tokens=32, temperature=1.0, top_p=0.9, 
                 do_sample=False, pad_token_id=None, eos_token_id=None, ):
        """
        Generate text using the LPLLM model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            Generated token IDs of shape (batch_size, original_seq_len + max_new_tokens)
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if batch_size == 1:
            # For single batch, duplicate the input to maintain the split structure
            input_ids = input_ids.repeat(2, 1)
            batch_size = 2

        # Set default token IDs if not provided
        if pad_token_id is None:
            pad_token_id = getattr(self.config, 'pad_token_id', None)
        if eos_token_id is None:
            eos_token_id = getattr(self.config, 'eos_token_id', None)
        
        # Initialize generation state
        if hasattr(input_ids, "input_ids"):
            generated_tokens = input_ids.input_ids
            attention_mask = input_ids.attention_mask
        generated_tokens = input_ids.clone()

        generated_tokens1 = []
        generated_tokens2 = []
        

        
        logger.info(f"Starting generation with {batch_size} sequences, max_new_tokens={max_new_tokens}")
        
        for step in range(max_new_tokens):
            # Check if all sequences are finished
                
            logger.info(f"step: {step}")
            

            # Forward pass through the model
            try:

                if step == 0:
                    # Create attention_mask: 1 for valid tokens, 0 for padding
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device="cpu")
                    
                    # Create position_ids: sequential positions for each token
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)

                    # logger.debug(f"attention_mask  {attention_mask} position_ids {position_ids}")
                    
                    # First step: process the entire input sequence
                    current_input = generated_tokens

                    time_start_pre = time.time()
                    input_embeds1, input_embeds2, \
                    past_key_values1_split, past_key_values2_split, \
                    attention_mask1, attention_mask2, position_ids1, position_ids2 = self.split_decoders(
                        current_input, step=0, past_key_values1=past_key_values1, past_key_values2=past_key_values2, attention_mask=attention_mask, position_ids=position_ids
                    )
                    logger.debug(f"split decoders init cost {time.time()-time_start_pre} s")

                    time_start_decoders = time.time()
                    layer_output1, layer_output2 = self.decoders(
                        hidden_states1=input_embeds1,
                        hidden_states2=input_embeds2,
                        past_key_value1=past_key_values1,
                        past_key_value2=past_key_values2,
                        attention_mask1=attention_mask1,
                        attention_mask2=attention_mask2,
                        position_ids1=position_ids1,
                        position_ids2=position_ids2,
                        step=step
                    )
                    logger.debug(f"split_decoders decoders cost {time.time()-time_start_decoders} s")

                else:

                    time_start_pre = time.time()
                    # Subsequent steps: only process the new token with cached states
                    current_input1 = next_token_ids1  # 使用embeddings而不是token IDs
                    current_input2 = next_token_ids2  # 使用embeddings而不是token IDs

                    batch_size, _ = current_input1.shape
                    generated_seq_len = generated_tokens.shape[1] + step

                    # should be [batch, generated_tokens+step], [8,256] [8,257]
                    attention_mask1 = torch.ones((batch_size, generated_seq_len), dtype=torch.long, device="cpu")
                    attention_mask2 = torch.ones((batch_size, generated_seq_len), dtype=torch.long, device="cpu")

                    position_ids1 = attention_mask1.long().cumsum(-1) - 1
                    position_ids1.masked_fill_(attention_mask1 == 0, 1)
                    if past_key_values1:
                        position_ids1 = position_ids1[:, -current_input1.shape[1] :]

                    position_ids2 = attention_mask2.long().cumsum(-1) - 1
                    position_ids2.masked_fill_(attention_mask2 == 0, 1)
                    if past_key_values2:
                        position_ids2 = position_ids2[:, -current_input2.shape[1] :]
                        
                    (input_embeds1, attention_mask1,
                    past_key_values1, position_ids1) \
                    =self.forward_prepare(
                        input_ids=current_input1,
                        inputs_embeds=None,
                        position_ids=position_ids1,
                        past_key_values=past_key_values1,
                        output_attentions=False,
                        use_cache=True,
                        attention_mask=attention_mask1,
                    )

                    (input_embeds2, attention_mask2,
                    past_key_values2, position_ids2) \
                    =self.forward_prepare(
                        input_ids=current_input2,
                        inputs_embeds=None,
                        position_ids=position_ids2,
                        past_key_values=past_key_values2,
                        output_attentions=False,
                        use_cache=True,
                        attention_mask=attention_mask2,
                    )
                    logger.debug(f"split decoders pre cost {time.time()-time_start_pre} s")


                    time_start_decoders = time.time()
                    layer_output1, layer_output2 = self.decoders(
                        hidden_states1=input_embeds1,
                        hidden_states2=input_embeds2,
                        past_key_value1=past_key_values1,
                        past_key_value2=past_key_values2,
                        attention_mask1=attention_mask1,
                        attention_mask2=attention_mask2,
                        position_ids1=position_ids1,
                        position_ids2=position_ids2,
                        step=step
                    )
                    logger.debug(f"split_decoders decoders cost {time.time()-time_start_decoders} s")

                # Apply norm and lm_head to get logits
                # Get the model components
                norm = self.lpmodule_class_instance.model.norm
                lm_head = self.lpmodule_class_instance.lm_head

                
                def get_next_token(hidden_states):
                    """获取下一个token的ID和对应的embedding"""
                    # 先分块norm，再获取last_hidden_states，再拼接
                    max_batch_size = 180
                    batch_size_actual = hidden_states.size(0)
                    if batch_size_actual <= max_batch_size:
                        normed_hidden_states = norm(hidden_states)
                    else:
                        normed_chunks = []
                        for i in range(0, batch_size_actual, max_batch_size):
                            end_idx = min(i + max_batch_size, batch_size_actual)
                            chunk = hidden_states[i:end_idx]
                            normed_chunk = norm(chunk)
                            normed_chunks.append(normed_chunk)
                        normed_hidden_states = torch.cat(normed_chunks, dim=0)
                    hidden_states = normed_hidden_states[:, -1:, :]

                    last_hidden_states = hidden_states[:, -1:, :]  # Shape: (batch_size, 1, hidden_size)

                    batch_size_actual = last_hidden_states.size(0)
            
                    max_batch_size = 180
                    if batch_size_actual <= max_batch_size:
                        # Small batch, process all at once
                        next_token_logits = lm_head(last_hidden_states).squeeze(1)  # Shape: (batch_size, vocab_size)
                    else:
                        # Large batch, process in chunks
                        next_token_logits_list = []
                        for i in range(0, batch_size_actual, max_batch_size):
                            end_idx = min(i + max_batch_size, batch_size_actual)
                            chunk = last_hidden_states[i:end_idx]
                            chunk_logits = lm_head(chunk).squeeze(1)  # Shape: (chunk_size, vocab_size)
                            next_token_logits_list.append(chunk_logits)
                        
                        # Concatenate results
                        next_token_logits = torch.cat(next_token_logits_list, dim=0)
                    next_token_logits = next_token_logits.float()
                
                    # Use efficient logits processing function
                    next_token_ids = process_logits_efficiently(
                        logits=next_token_logits,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        device=device
                    )
                    
                    # 将token IDs转换为embeddings
                    next_token_ids = next_token_ids.unsqueeze(1)  # Shape: (batch_size, 1)
                    return next_token_ids

                time_start_get_next_token=time.time()
                next_token_ids1 = get_next_token(layer_output1)
                logger.debug(f"get_next_token1 cost {time.time()-time_start_get_next_token} s")
                next_token_ids2 = get_next_token(layer_output2)
                logger.debug(f"get_next_token2 cost {time.time()-time_start_get_next_token} s")
                
                past_key_values1 = past_key_values1
                past_key_values2 = past_key_values2
                

                generated_tokens1.append(next_token_ids1)
                generated_tokens2.append(next_token_ids2)
                
            except Exception as e:
                logger.error(f"Error during generation at step {step + 1}: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                
                # 添加详细的调用栈信息
                import traceback
                logger.error("Full traceback:")
                for line in traceback.format_exc().split('\n'):
                    logger.error(f"  {line}")
                
                # 添加内存状态信息
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    reserved = torch.cuda.memory_reserved(device) / 1024**3
                    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
                    logger.error(f"Memory state on error: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")
                
                # 添加当前状态信息
                logger.error(f"Current step: {step}")
                logger.error(f"Generated tokens shape: {generated_tokens.shape}")
                logger.error(f"Input shape: {input_ids.shape}")
                logger.error(f"Batch size: {batch_size}")
                
                break
        
        for next_token_ids1, next_token_ids2 in zip(generated_tokens1, generated_tokens2):
            generated_new_tokens = torch.cat([next_token_ids1, next_token_ids2], dim=0)
            generated_tokens = torch.cat([generated_tokens, generated_new_tokens], dim=1)
        # Final cleanup
        torch.cuda.empty_cache()
        
        logger.info(f"Generation completed. Final shape: {generated_tokens.shape}")
        return generated_tokens
    
    def stop(self):
        """停止所有线程和清理资源"""
        if self._shutdown:
            return
            
        self._shutdown = True
        logger.info("Stopping LPLLM...")
        
        try:
            # 停止注意力管理器
            if hasattr(self, 'attn_manager') and self.attn_manager:
                self.attn_manager.stop()
                logger.info("AttnManager stopped")
        except Exception as e:
            logger.error(f"Error stopping attn_manager: {e}")
        
        # 强制停止所有线程
        self._force_stop_threads()
        
        # 清理CUDA内存
        try:
            if hasattr(self, 'cuda_memory_view') and self.cuda_memory_view:
                self.cuda_memory_view.free_layer_gpu_all()
                logger.info("CUDA memory freed")
        except Exception as e:
            logger.error(f"Error freeing CUDA memory: {e}")
        
        logger.info("LPLLM stopped successfully")
    
    def __del__(self):
        """析构函数，确保资源被清理"""
        if not self._shutdown:
            try:
                self.stop()
            except Exception as e:
                logger.error(f"Error in __del__: {e}")
@dataclass
class StateView:
    memory_ptr: dict
    state_dict: dict
    index: int=0
# for single device
class CudaMemoryView:
    def __init__(self, memory_num: int, memory_size: int, model_memory_size: int, device_index: int, client: SllmStoreClient):
        self.memory_num = memory_num
        self.device_index = device_index
        self.model_memory_size = model_memory_size
        self.memory_queue_free: Queue[StateView] = Queue()
        self.memory_queue_allocated: Queue[StateView] = Queue()
        self.client = client

        self.replica_uuid_list: Queue[(str,int)] = Queue()

        device_memory = {
            device_index: memory_size
        }
        for i in range(memory_num):
            # for single device
            cuda_memory_ptr = allocate_cuda_memory(device_memory)
            sv = StateView(memory_ptr=cuda_memory_ptr, state_dict={})

            self.memory_queue_free.put(sv)
        help_model_device_memory = {
            device_index: model_memory_size
        }
        self.help_model_memory_ptrs = allocate_cuda_memory(help_model_device_memory)
    def get_state_loc_mlp(self):
        return 2
    def get_state_loc_attn(self):
        return 1
    def restore_help_model(self, model, model_state_dict):
        for name, param in model_state_dict.items():
            set_module_tensor_to_device(model, name, param.device, param, clear_cache=True)
        # buffer_names = [name for name, _ in model.named_buffers()] 
        # logger.debug(f"{buffer_names}")
        # for name, param in model_state_dict.items():
        #     names = name.split(".")
        #     name = ".".join(names[:-2])
        #     logger.debug(f"{name}")
        #     set_module_buffer_to_device(model, name, param.device)
        
        model.eval()
    def restore_layer(self, layer, state_dict, layer_attr_loc):
        time_start_restore_func =  time.time()
        for name, param in state_dict.items():
            relative_layer_name = ".".join(name.split(".")[layer_attr_loc:])

            set_module_tensor_to_device(layer, relative_layer_name, param.device, param, clear_cache=False)
        layer.eval()

        logger.debug(f"restore layer func cost {time.time()-time_start_restore_func} s")

    def get_state_dict(self, layer_idx: int):
        for state_view in self.memory_queue_allocated.queue:
            if state_view.index == layer_idx:
                state_dict = state_view.state_dict
                return state_dict
        raise ValueError(f"state dict should not be empty or layer_idx not match")
    def updateState(self, layer_idx: int, update_name_list: list[str], if_mlp: bool):
        # the loc in queue

        time_start_updatestate = time.time()

        if if_mlp:
            update_state_loc = self.get_state_loc_mlp()
        else:
            update_state_loc = self.get_state_loc_attn()
        for i in range(len(self.memory_queue_allocated.queue)):
            if self.memory_queue_allocated.queue[i].index == layer_idx:
                update_state_loc = i
                break
        if update_state_loc == -1:
            raise ValueError(f"update with {layer_idx} but get {update_state_loc} not match")
        update_state_dict = self.memory_queue_allocated.queue[update_state_loc].state_dict

        # logger.debug(f"update_state_dict {update_state_dict.keys()}")
        ustate_dict = {}
        for update_name in update_name_list:
            for src_name, tensor in update_state_dict.items():
                if update_name in src_name:
                    ustate_dict[src_name] = tensor
        logger.debug(f"update state cost {time.time()-time_start_updatestate} s")
        return ustate_dict

    def load_help_model_gpu(self, model_path, model_tensor_copy_chunks, model_tensor_meta_index, model_tensor_device_offsets):
        cuda_memory_ptrs = self.help_model_memory_ptrs

        cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)
        # logger.debug(f"cuda memory handles: {cuda_memory_ptrs} {cuda_memory_handles}")
        ret1, replica_uuid1 = load_into_gpu_async(
            model_path=model_path,
            tensor_copy_chunks=model_tensor_copy_chunks,
            cuda_memory_handles=cuda_memory_handles
        )
        if not ret1:
            raise ValueError(f"Failed to load model {model_path} into GPU")
        help_model_state_dict = restore_tensors2(
            model_tensor_meta_index, cuda_memory_ptrs, model_tensor_device_offsets
        )
        self.help_model_state_dict = help_model_state_dict
        return replica_uuid1
    def get_help_state_dict(self):
        return self.help_model_state_dict
    def wait_help_model_gpu_loading(self, model_path: str, replica_uuid: str):
        self.client.confirm_model_loaded(model_path, replica_uuid)
        return
    def load_layer_gpu(self, layer_idx, model_path, layer_tensor_copy_chunks, layer_tensor_meta_index, layer_tensor_device_offsets):

        sv = self.memory_queue_free.get()
        cuda_memory_ptrs = sv.memory_ptr
        cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)
        logger.debug(f"cuda memory handles: {cuda_memory_ptrs} {cuda_memory_handles}")
        ret1, replica_uuid1 = load_into_gpu_async(
            model_path=model_path,
            tensor_copy_chunks=layer_tensor_copy_chunks,
            cuda_memory_handles=cuda_memory_handles
        )
        if not ret1:
            raise ValueError(f"Failed to load model {model_path} {layer_idx} into GPU")
            
        # 分配torch
        layer_state_dict = restore_tensors2(
            layer_tensor_meta_index, cuda_memory_ptrs, layer_tensor_device_offsets
        )
        sv.state_dict = layer_state_dict
        sv.index = layer_idx
        self.memory_queue_allocated.put(sv)
        
        self.replica_uuid_list.put((replica_uuid1, layer_idx))

    def wait_layer_gpu_loading(self, layer_idx: int, model_path: str):
        replica_uuid, idx = self.replica_uuid_list.get(block=False)
        if idx != layer_idx:
            raise ValueError(f"Waiting {layer_idx}, but get {idx}")
        self.client.confirm_model_loaded(model_path, replica_uuid)
        return
    def free_layer_gpu_all(self):
        while self.memory_queue_allocated.qsize() > 0:
            sv_need_free: StateView = self.memory_queue_allocated.get()
            sv_need_free.index = 0
            self.memory_queue_free.put(sv_need_free)
        return


    def free_layer_gpu(self, layer_free_idx):
        sv_need_free: StateView = self.memory_queue_allocated.get()
        if sv_need_free.index != layer_free_idx:
            raise ValueError(f"Going to Free {layer_free_idx}, but the real layer is {sv_need_free.index}")
        sv_need_free.index = 0
        self.memory_queue_free.put(sv_need_free)

    def __del__(self):
        def free_queue(fqueue):
            while not fqueue.empty():
                element: StateView = fqueue.get()
                memory_ptr = element.memory_ptr
                if memory_ptr != None:
                    free_cuda_memory(memory_ptr)
        free_queue(self.memory_queue_free)
        free_queue(self.memory_queue_allocated)

@dataclass
class AttnInputs:
    attn_input: dict
    query_states: torch.Tensor
    key_states: torch.Tensor
    value_states: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor
    if_batch: bool
@dataclass
class AttnOutputs:
    attn_output: torch.Tensor
@dataclass
class AttnTask:
    type: int
    if_batch: bool=False
    attn_inputs: AttnInputs | None = None
    task_group_id: int | None = None  # 并行任务组ID
    subtask_idx: int | None = None    # 子任务索引
@dataclass
class AttnOut:
    attn_output: torch.Tensor
    task_group_id: int | None = None  # 并行任务组ID
    subtask_idx: int | None = None    # 子任务索引
    updated_past_key_value: Cache | None = None  # 更新后的cache
@dataclass
class IOTask:
    type: int
    attn_inputs: AttnInputs | None = None
    attn_output: AttnOut | None = None
    task_group_id: int | None = None  # 并行任务组ID
    subtask_idx: int | None = None    # 子任务索引
    updated_past_key_value: Cache | None = None  # 更新后的cache
class AttnManager:
    def __init__(self, 
        lpmodule_class: LPModuleWrapper, device: str, config: AutoConfig, 
        pool_size: int=2, use_server_pool: bool = True, chunk_size_mb = 1*1024
    ):
        
        self.lpmodule_class = lpmodule_class
        # Use server pool memory if enabled, otherwise use local pool
        if use_server_pool:
            self.pool_memory = ServerPinnedMemoryPool(
                dtype=config.torch_dtype,
                pool_size=pool_size,
                device=device
            )
        else:
            self.pool_memory = PinnedMemoryPool(config.torch_dtype, pool_size, chunk_size_mb = chunk_size_mb)
        # 分离的队列系统实现真正并行
        self.out_queue: Queue[AttnOut] = Queue()
        self.gpu2cpu_queue: Queue[IOTask] = Queue()  # GPU→CPU移动队列
        self.cpu_compute_queue: Queue[AttnTask] = Queue()  # CPU计算队列  
        self.cpu2gpu_queue: Queue[IOTask] = Queue()  # CPU→GPU移动队列
        
        # 三个独立线程实现并行处理
        self.gpu2cpu_thread = GPU2CPUThread(
            device=device, lpmodule_class=self.lpmodule_class,
            pool_memory=self.pool_memory, 
            # change output_queue to out_queue for test, origin cpu_compute_queue
            input_queue=self.gpu2cpu_queue, output_queue=self.out_queue)
        self.cpu_compute_thread = CPUComputeThread(
            device=device,
            lpmodule_class=lpmodule_class, 
            # change output_queue to out_queue for test, origin cpu2gpu_queue
            input_queue=self.cpu_compute_queue, output_queue=self.out_queue, pool_memory=self.pool_memory)
        self.cpu2gpu_thread = CPU2GPUThread(
            device=device, pool_memory=self.pool_memory,
            input_queue=self.cpu2gpu_queue, output_queue=self.out_queue)
        
        # 用于跟踪并行任务
        self.parallel_tasks = {}
        self.task_lock = threading.Lock()
    def start(self):
        self.gpu2cpu_thread.start()
        self.cpu_compute_thread.start()
        self.cpu2gpu_thread.start()
    def stop(self):
        logger.info("Stopping AttnManager...")
        try:
            self.gpu2cpu_thread.stop()
            self.cpu_compute_thread.stop()
            self.cpu2gpu_thread.stop()
            
            # 等待线程结束
            if hasattr(self, 'gpu2cpu_thread') and self.gpu2cpu_thread.thread.is_alive():
                self.gpu2cpu_thread.thread.join(timeout=2.0)
            if hasattr(self, 'cpu_compute_thread') and self.cpu_compute_thread.thread.is_alive():
                self.cpu_compute_thread.thread.join(timeout=2.0)
            if hasattr(self, 'cpu2gpu_thread') and self.cpu2gpu_thread.thread.is_alive():
                self.cpu2gpu_thread.thread.join(timeout=2.0)
                
            logger.info("AttnManager stopped")
        except Exception as e:
            logger.error(f"Error stopping AttnManager: {e}")
    def submit_task(self, bsz, q_len, layer, layer_idx, query_states, key_states, value_states, sin, cos, attention_mask, past_key_value, position_ids, if_batch=False):
        attn_input = {
            "bsz": bsz,
            "q_len": q_len,
            "layer": layer,
            "layer_idx": layer_idx,
            "query_states": query_states,
            "key_states": key_states,
            "value_states": value_states,
            "sin": sin,
            "cos": cos,
            "attention_mask": attention_mask,
            "past_key_value": past_key_value,
            "position_ids": position_ids,
        }
        attn_inputs = AttnInputs(
            attn_input=attn_input,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            sin=sin,
            cos=cos,
            if_batch=if_batch,
        )
        iotask = IOTask(
            type=0,
            attn_inputs=attn_inputs,
        )
        self.gpu2cpu_thread.submit_task(iotask)
    
    def submit_parallel_tasks(self, bsz, q_len, layer, layer_idx, query_states, key_states, value_states, sin, cos, attention_mask, past_key_value, position_ids, num_parallel):
        """
        将注意力计算任务拆分为多个并行子任务
        """
        chunk_size = bsz // num_parallel
        logger.debug(f"Splitting attention into {num_parallel} parallel tasks, chunk_size={chunk_size}")
    
        # 记录这是一个并行任务组
        task_group_id = id((query_states, key_states, value_states))  # 使用tensor id作为组标识
        self.parallel_tasks[task_group_id] = {
            'num_tasks': num_parallel,
            'completed': 0,
            'results': [None] * num_parallel,
            'updated_caches': [None] * num_parallel,  # 存储更新后的cache
            'original_past_key_value': past_key_value,  # 保存原始cache引用
            'layer_idx': layer_idx  # 保存当前处理的层索引
        }
        
        # 拆分并提交子任务
        for i in range(num_parallel):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            
            # 拆分各个tensor的batch维度
            chunk_query = query_states[start_idx:end_idx]
            chunk_key = key_states[start_idx:end_idx] 
            chunk_value = value_states[start_idx:end_idx]
            chunk_sin = sin[start_idx:end_idx] if sin is not None else None
            chunk_cos = cos[start_idx:end_idx] if cos is not None else None
            
            # 处理attention_mask和position_ids
            chunk_attention_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
            chunk_position_ids = position_ids[start_idx:end_idx] if position_ids is not None else None
            
            # 处理past_key_value - Cache对象需要特殊处理
            chunk_past_key_value = None
            if past_key_value is not None:
                if isinstance(past_key_value, Cache):
                    # 正确拆分cache，保留历史信息，但避免None值问题
                    chunk_past_key_value = DynamicCache()
                    if hasattr(past_key_value, 'key_cache') and past_key_value.key_cache:
                        chunk_past_key_value.key_cache = []
                        chunk_past_key_value.value_cache = []
                        
                        for layer_idx in range(len(past_key_value.key_cache)):
                            if past_key_value.key_cache[layer_idx] is not None:
                                # 拆分非空层
                                chunk_key = past_key_value.key_cache[layer_idx][start_idx:end_idx]
                                chunk_value = past_key_value.value_cache[layer_idx][start_idx:end_idx]
                                chunk_past_key_value.key_cache.append(chunk_key)
                                chunk_past_key_value.value_cache.append(chunk_value)
                            else:
                                # 对于空层，不添加到cache中
                                # DynamicCache.update()会在需要时自动扩展
                                break  # 遇到第一个None就停止，因为后续层肯定都是None
                    # 如果没有key_cache或者为空，保持空的DynamicCache
                elif hasattr(past_key_value, '__getitem__') and hasattr(past_key_value, '__len__'):
                    # 处理tuple/list形式的past_key_value
                    chunk_past_key_value = []
                    for layer_past in past_key_value:
                        if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
                            # (key, value) pair
                            chunk_key = layer_past[0][start_idx:end_idx] if layer_past[0] is not None else None
                            chunk_value = layer_past[1][start_idx:end_idx] if layer_past[1] is not None else None
                            chunk_past_key_value.append((chunk_key, chunk_value))
                        else:
                            chunk_past_key_value.append(layer_past)
                else:
                    # 无法拆分的情况，每个子任务使用相同的past_key_value
                    # 注意：这可能导致问题，因为不同子任务会更新同一个cache
                    logger.warning("past_key_value cannot be split, sharing among subtasks - this may cause issues")
                    chunk_past_key_value = past_key_value
            
            attn_input = {
                "bsz": chunk_size,
                "q_len": q_len,
                "layer": layer,
                "layer_idx": layer_idx,
                "query_states": chunk_query,
                "key_states": chunk_key,
                "value_states": chunk_value,
                "sin": chunk_sin,
                "cos": chunk_cos,
                "attention_mask": chunk_attention_mask,
                "past_key_value": chunk_past_key_value,
                "position_ids": chunk_position_ids,
            }
            
            attn_inputs = AttnInputs(
                attn_input=attn_input,
                query_states=chunk_query,
                key_states=chunk_key,
                value_states=chunk_value,
                sin=chunk_sin,
                cos=chunk_cos,
            )
            
            iotask = IOTask(
                type=0,
                attn_inputs=attn_inputs,
                task_group_id=task_group_id,  # 添加组标识
                subtask_idx=i,  # 添加子任务索引
            )
            self.gpu2cpu_thread.submit_task(iotask)
    def wait_task(self):
        logger.debug("waiting out_queue")
        attn_output: AttnOut = self.out_queue.get()
        
        # 检查是否是并行任务的结果
        if hasattr(attn_output, 'task_group_id') and attn_output.task_group_id is not None:
            return self._handle_parallel_result(attn_output)
        else:
            return attn_output
    
    def _handle_parallel_result(self, attn_output: AttnOut):
        """处理并行任务的结果合并"""
        task_group_id = attn_output.task_group_id
        subtask_idx = attn_output.subtask_idx
        
        # 检查并更新任务状态
        need_wait = False
        with self.task_lock:
            if task_group_id not in self.parallel_tasks:
                logger.error(f"Unknown task group {task_group_id}")
                return attn_output
            
            task_group = self.parallel_tasks[task_group_id]
            task_group['results'][subtask_idx] = attn_output.attn_output
            task_group['updated_caches'][subtask_idx] = attn_output.updated_past_key_value
            task_group['completed'] += 1
            
            # 检查是否所有子任务都完成了
            if task_group['completed'] < task_group['num_tasks']:
                # 还有子任务未完成，需要继续等待
                need_wait = True
                logger.debug(f"Parallel task {subtask_idx}/{task_group['num_tasks']} completed, waiting for more")
            else:
                # 所有子任务完成，合并结果
                logger.debug(f"All {task_group['num_tasks']} parallel tasks completed, merging results")
                merged_result = torch.cat(task_group['results'], dim=0)  # 沿batch维度合并
                
                # 合并更新后的cache
                merged_cache = self._merge_caches(
                    task_group['updated_caches'], 
                    task_group['original_past_key_value'],
                    task_group['layer_idx']
                )
                
                # 清理任务组记录
                del self.parallel_tasks[task_group_id]
                
                # 返回合并后的结果
                return AttnOut(
                    attn_output=merged_result,
                    updated_past_key_value=merged_cache
                )
        
        # 在锁外面递归调用，避免死锁
        if need_wait:
            return self.wait_task()  # 递归等待下一个结果
    
    def _merge_caches(self, updated_caches, original_past_key_value, layer_idx):
        """直接在原有cache对象上合并指定层的内容 - 避免创建新对象"""
        if not any(cache is not None for cache in updated_caches):
            return original_past_key_value
        
        if original_past_key_value is None:
            # 如果没有原始cache，创建新的
            original_past_key_value = DynamicCache()
            original_past_key_value.key_cache = [None] * (layer_idx + 1)
            original_past_key_value.value_cache = [None] * (layer_idx + 1)
        
        if isinstance(original_past_key_value, Cache):
            # 确保cache有足够的层数
            if hasattr(original_past_key_value, 'key_cache'):
                while len(original_past_key_value.key_cache) <= layer_idx:
                    original_past_key_value.key_cache.append(None)
                    original_past_key_value.value_cache.append(None)
            else:
                original_past_key_value.key_cache = [None] * (layer_idx + 1)
                original_past_key_value.value_cache = [None] * (layer_idx + 1)
            
            # 收集要合并的层数据
            layer_keys = []
            layer_values = []
            
            for cache in updated_caches:
                if (cache is not None and 
                    hasattr(cache, 'key_cache') and 
                    len(cache.key_cache) > layer_idx and
                    cache.key_cache[layer_idx] is not None):
                    layer_keys.append(cache.key_cache[layer_idx])
                    layer_values.append(cache.value_cache[layer_idx])
            
            if layer_keys:
                # 直接在原始cache上更新指定层
                original_past_key_value.key_cache[layer_idx] = torch.cat(layer_keys, dim=0)
                original_past_key_value.value_cache[layer_idx] = torch.cat(layer_values, dim=0)
            
            # seen_tokens 通常由 DynamicCache 内部管理，不需要手动设置
            # 如果需要的话，DynamicCache.update() 方法会自动更新 seen_tokens
            
            return original_past_key_value
        
        else:
            # 处理其他类型的cache（如tuple/list）
            logger.warning("Non-Cache type past_key_value merging not fully implemented")
            return original_past_key_value

# 新的三线程并行架构
class GPU2CPUThread:
    """专门处理GPU到CPU的数据移动"""
    def __init__(self, device: str, lpmodule_class: LPModuleWrapper, pool_memory: PinnedMemoryPool, input_queue: Queue[IOTask], output_queue: Queue[AttnTask]):
        self.input_queue = input_queue
        self.lpmodule_class = lpmodule_class
        self.output_queue = output_queue
        self.io_stream = torch.cuda.Stream(device=device)
        self.thread = threading.Thread(target=self.run, daemon=True)  # 设置为守护线程
        self.device = device
        self.pool = pool_memory
        self._stop_event = threading.Event()
        self._shutdown = False
        
    def start(self):
        self.thread.start()
        
    def run(self):
        logger.debug(f"GPU2CPU thread started, id {get_thread_id()}")
        try:
            with torch.cuda.stream(self.io_stream):
                while not self._stop_event.is_set() and not self._shutdown:
                    try:
                        io_task: IOTask = self.input_queue.get(timeout=1.0)
                        if io_task.type < 0:  # 停止信号
                            break
                        self.process_gpu2cpu(io_task)
                    except Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error in GPU2CPU thread: {e}")
                        if self._shutdown:
                            break
        except Exception as e:
            logger.error(f"Fatal error in GPU2CPU thread: {e}")
            import traceback
            logger.error("GPU2CPU thread traceback:")
            for line in traceback.format_exc().split('\n'):
                logger.error(f"  {line}")
        finally:
            logger.debug("GPU2CPU thread exiting")
                    
    def process_gpu2cpu(self, io_task: IOTask):
        """处理GPU到CPU的数据移动"""
        if io_task.attn_inputs is None:
            return
            
        attn_inputs: AttnInputs = io_task.attn_inputs
        q_data = attn_inputs.query_states
        k_data = attn_inputs.key_states
        v_data = attn_inputs.value_states
        s_data = attn_inputs.sin
        c_data = attn_inputs.cos
        
        time_start = time.time()
        
        # 分配固定内存并异步拷贝到CPU
        q_c = self.pool.alloc_same_pin_tensor(q_data)
        k_c = self.pool.alloc_same_pin_tensor(k_data)
        v_c = self.pool.alloc_same_pin_tensor(v_data)
        s_c = self.pool.alloc_same_pin_tensor(s_data)
        c_c = self.pool.alloc_same_pin_tensor(c_data)

        # q_c =q_data.pin_memory()
        # k_c =k_data.pin_memory()
        # v_c =v_data.pin_memory()
        # s_c =s_data.pin_memory()
        # c_c =c_data.pin_memory()
        
        if_nonblcok  = True

        cuda_copy_(q_c, q_data, non_blocking=if_nonblcok)
        cuda_copy_(k_c, k_data, non_blocking=if_nonblcok) 
        cuda_copy_(v_c, v_data, non_blocking=if_nonblcok)
        cuda_copy_(s_c, s_data, non_blocking=if_nonblcok)
        cuda_copy_(c_c, c_data, non_blocking=if_nonblcok)
        
        logger.debug(f"GPU2CPU move cost {time.time()-time_start:.6f} seconds")
        
        # 直接计算 attention
        attn_inputs.attn_input.update({
            "query_states": q_c,
            "key_states": k_c,
            "value_states": v_c,
            "sin": s_c,
            "cos": c_c
        })
        attn_input = attn_inputs.attn_input
        if_batch = attn_inputs.if_batch

        time_start_attn = time.time()
        if if_batch:
            attn_result = self.lpmodule_class.decoder_attn_batch(**attn_input, pool_memory=self.pool)
        else:
            attn_result = self.lpmodule_class.decoder_attn(**attn_input, pool_memory=self.pool)
        logger.debug(f"CPU attn cost {time.time() - time_start_attn:.6f} seconds if batch {if_batch}")

        time_start_deal = time.time()
        # 处理返回值
        if isinstance(attn_result, tuple) and len(attn_result) == 2:
            attn_output, updated_past_key_value = attn_result
        else:
            attn_output = attn_result
            updated_past_key_value = attn_input.get('past_key_value', None)
        logger.debug(f"deal attn result cost {time.time() - time_start_deal:.6f} seconds")

        logger.debug(f"CPU compute cost {time.time() - time_start:.6f} seconds")
        
        attn_out = AttnOut(
            attn_output=attn_output,
            task_group_id=getattr(io_task, 'task_group_id', None),
            subtask_idx=getattr(io_task, 'subtask_idx', None),
            updated_past_key_value=updated_past_key_value
        )

        self.output_queue.put(attn_out)
        # 清理内存
        # self.pool.free(attn_cpu)
        time_start = time.time()
        self.pool.free(q_c)
        self.pool.free(k_c)
        self.pool.free(v_c)
        self.pool.free(s_c)
        self.pool.free(c_c)
        logger.debug(f"free cost {time.time() - time_start:.6f} seconds")

        # old
        # # 更新attn_inputs中的数据
        # attn_inputs.query_states = q_c
        # attn_inputs.key_states = k_c
        # attn_inputs.value_states = v_c
        # attn_inputs.sin = s_c
        # attn_inputs.cos = c_c
        
        # # 提交到CPU计算队列
        # attn_task = AttnTask(
        #     type=0,
        #     if_batch=attn_inputs.if_batch,
        #     attn_inputs=attn_inputs,
        #     task_group_id=getattr(io_task, 'task_group_id', None),
        #     subtask_idx=getattr(io_task, 'subtask_idx', None)
        # )
        # self.output_queue.put(attn_task)

        # del q_data, k_data, v_data, s_data, c_data
        
    def submit_task(self, task: IOTask):
        self.input_queue.put(task)
        
    def stop(self):
        self._shutdown = True
        self._stop_event.set()
        try:
            self.input_queue.put(IOTask(type=-1))
        except Exception as e:
            logger.error(f"Error sending stop signal to GPU2CPU thread: {e}")

class CPUComputeThread:
    """专门处理CPU上的注意力计算"""
    def __init__(self, device: str, lpmodule_class: LPModuleWrapper, input_queue: Queue[AttnTask], output_queue: Queue[IOTask], pool_memory: PinnedMemoryPool):
        self.lpmodule_class = lpmodule_class
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.pool = pool_memory
        self.thread = threading.Thread(target=self.run, daemon=True)  # 设置为守护线程
        self._stop_event = threading.Event()
        self._shutdown = False
        self.io_stream = torch.cuda.Stream(device=device)
        torch.set_num_threads(128)  # Set intra-op threads
        
        
    def start(self):
        self.thread.start()
        
    def run(self):
        logger.info(f"CPU compute thread started, id {get_thread_id()}")
        try:
            with torch.cuda.stream(self.io_stream):
                while not self._stop_event.is_set() and not self._shutdown:
                    try:
                        attn_task: AttnTask = self.input_queue.get(timeout=1.0)
                        if attn_task.type < 0:  # 停止信号
                            break
                        self.process_cpu_compute(attn_task)
                    except Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error in CPU compute thread: {e}")

                        import traceback
                        logger.error("Full traceback:")
                        for line in traceback.format_exc().split('\n'):
                            logger.error(f"  {line}")
                    

                        if self._shutdown:
                            break
        except Exception as e:
            log_error_with_stack("Fatal error in CPU compute thread", e)
        finally:
            logger.debug("CPU compute thread exiting")
                
    def process_cpu_compute(self, attn_task: AttnTask):
        """处理CPU注意力计算"""
        attn_inputs = attn_task.attn_inputs
        time_start = time.time()
        
        # 更新attn_input字典
        attn_inputs.attn_input.update({
            "query_states": attn_inputs.query_states,
            "key_states": attn_inputs.key_states,
            "value_states": attn_inputs.value_states,
            "sin": attn_inputs.sin,
            "cos": attn_inputs.cos
        })
        
        attn_input = attn_inputs.attn_input
        if_batch = attn_inputs.if_batch

        time_start_attn = time.time()
        if if_batch:
            attn_result = self.lpmodule_class.decoder_attn_batch(**attn_input, pool_memory=self.pool)
        else:
            attn_result = self.lpmodule_class.decoder_attn(**attn_input, pool_memory=self.pool)
        logger.debug(f"CPU attn cost {time.time() - time_start_attn:.6f} seconds if batch {if_batch}")

        time_start_deal = time.time()
        # 处理返回值
        if isinstance(attn_result, tuple) and len(attn_result) == 2:
            attn_output, updated_past_key_value = attn_result
        else:
            attn_output = attn_result
            updated_past_key_value = attn_input.get('past_key_value', None)
        logger.debug(f"deal attn result cost {time.time() - time_start_deal:.6f} seconds")

        logger.debug(f"CPU compute cost {time.time() - time_start:.6f} seconds")
        
        # 提交到CPU2GPU队列
        # io_task = IOTask(
        #     type=1,
        #     attn_output=attn_output,
        #     attn_inputs=attn_inputs,
        #     task_group_id=getattr(attn_task, 'task_group_id', None),
        #     subtask_idx=getattr(attn_task, 'subtask_idx', None),
        #     updated_past_key_value=updated_past_key_value
        # )
        # self.output_queue.put(io_task)
        
        # 更改attn_output 不提交到cpu 2 gpu队列

        # 创建最终结果
        attn_out = AttnOut(
            attn_output=attn_output,
            task_group_id=getattr(attn_task, 'task_group_id', None),
            subtask_idx=getattr(attn_task, 'subtask_idx', None),
            updated_past_key_value=updated_past_key_value
        )

        self.output_queue.put(attn_out)
        # 清理内存
        # self.pool.free(attn_cpu)
        query_states = attn_inputs.query_states
        key_states = attn_inputs.key_states
        value_states = attn_inputs.value_states
        sin = attn_inputs.sin
        cos = attn_inputs.cos
        self.pool.free(query_states)
        self.pool.free(key_states)
        self.pool.free(value_states)
        self.pool.free(sin)
        self.pool.free(cos)
    def stop(self):
        self._shutdown = True
        self._stop_event.set()
        try:
            self.input_queue.put(AttnTask(type=-1))
        except Exception as e:
            logger.error(f"Error sending stop signal to CPU compute thread: {e}")

class CPU2GPUThread:
    """专门处理CPU到GPU的数据移动"""
    def __init__(self, device: str, pool_memory: PinnedMemoryPool, input_queue: Queue[IOTask], output_queue: Queue[AttnOut]):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.io_stream = torch.cuda.Stream(device=device)
        self.thread = threading.Thread(target=self.run, daemon=True)  # 设置为守护线程
        self.device = device
        self.pool = pool_memory
        self._stop_event = threading.Event()
        self._shutdown = False
        
    def start(self):
        self.thread.start()
        
    def run(self):
        logger.debug(f"CPU2GPU thread started, id {get_thread_id()}")
        try:
            with torch.cuda.stream(self.io_stream):
                while not self._stop_event.is_set() and not self._shutdown:
                    try:
                        io_task: IOTask = self.input_queue.get(timeout=1.0)
                        if io_task.type < 0:  # 停止信号
                            break
                        self.process_cpu2gpu(io_task)
                    except Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error in CPU2GPU thread: {e}")
                        if self._shutdown:
                            break
        except Exception as e:
            log_error_with_stack("Fatal error in CPU2GPU thread", e)
        finally:
            logger.debug("CPU2GPU thread exiting")
                    
    def process_cpu2gpu(self, io_task: IOTask):
        """处理CPU到GPU的数据移动"""
        if io_task.attn_inputs is None:
            return
            
        move_attn_output = io_task.attn_output
        attn_inputs = io_task.attn_inputs
        
        time_start = time.time()
        
        if_nonblcok = True
        # 移动注意力输出到GPU - 优化CPU到CPU拷贝
        # attn_cpu = self.pool.alloc_same_pin_tensor(move_attn_output)
        
        # 优化策略：使用最优拷贝方法（基于性能测试结果）
        # if move_attn_output.is_contiguous():
        #     # 已经是连续内存，直接使用 non-blocking copy（最快）
        #     # attn_cpu.copy_(move_attn_output, non_blocking=if_nonblcok)
        #     pass
        # else:
        #     # 非连续内存，先转换为连续再拷贝
        #     move_attn_output = move_attn_output.contiguous()
            # attn_cpu.copy_(move_attn_output, non_blocking=if_nonblcok)

        time_start_pin = time.time()
        attn_cpu = move_attn_output.pin_memory()
        logger.debug(f"Pin memory cost {time.time()-time_start_pin:.6f} seconds")
        attn_output = attn_cpu.to(device=self.device, non_blocking=if_nonblcok)
        
        # transpose and reshape here, move to mlp before
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(
        #     attn_inputs.query_states.shape[0], #bsz
        #     attn_inputs.query_states.shape[2], #q_len
        #     attn_inputs.query_states.shape[1] * attn_inputs.query_states.shape[3] #hidden_size
        # )

        logger.debug(f"CPU2GPU move cost {time.time()-time_start:.6f} seconds")
        
        # 创建最终结果
        attn_out = AttnOut(
            attn_output=attn_output,
            task_group_id=getattr(io_task, 'task_group_id', None),
            subtask_idx=getattr(io_task, 'subtask_idx', None),
            updated_past_key_value=getattr(io_task, 'updated_past_key_value', None)
        )
        
        self.output_queue.put(attn_out)
        
        # 清理内存

        # self.pool.free(attn_cpu)
        query_states = attn_inputs.query_states
        key_states = attn_inputs.key_states
        value_states = attn_inputs.value_states
        sin = attn_inputs.sin
        cos = attn_inputs.cos
        self.pool.free(query_states)
        self.pool.free(key_states)
        self.pool.free(value_states)
        self.pool.free(sin)
        self.pool.free(cos)
        
    def stop(self):
        self._shutdown = True
        self._stop_event.set()
        try:
            self.input_queue.put(IOTask(type=-1))
        except Exception as e:
            logger.error(f"Error sending stop signal to CPU2GPU thread: {e}")

class QThread:
    def __init__(self, device: str, pool_memory: PinnedMemoryPool, out_queue: Queue[AttnOut], cqueue: Queue[AttnTask]):
        self.io_queue = Queue()
        self.io_stream = torch.cuda.Stream(device=device)
        self.thread = threading.Thread(target=self.run)
        self.device = device
        self.pool = pool_memory
        self.out_queue = out_queue
        self.cqueue = cqueue
    def start(self):
        self.thread.start()
    def run(self):
        
        logger.debug(f"setup io func")
        logger.debug(f"io_func id {get_thread_id()}")
        with torch.cuda.stream(self.io_stream):
            while True:
                io_task: IOTask = self.io_queue.get()
                if io_task.type < 0:
                    return
                self.process(io_task)
    def submit_task(self, task: IOTask):
        self.io_queue.put(task)
        return
    def stop(self):
        self.io_queue.put(IOTask(type=-1))
    def process(self, io_task: IOTask):
        io_type=io_task.type
        # gpu2cpu
        if io_type == 0: 
            if io_task.attn_inputs is None:
                return
            attn_inputs = io_task.attn_inputs
            q_data = attn_inputs.query_states
            k_data = attn_inputs.key_states
            v_data = attn_inputs.value_states
            s_data = attn_inputs.sin
            c_data = attn_inputs.cos
            
            time_start = time.time()
            # logger.info(f"move qkv map")
            # torch.cuda.synchronize(device=self.device)
            # 将数据从GPU移动到CPU
            
            # with same shape
            time_start_alloc = time.time()
            q_c = self.pool.alloc_same_pin_tensor(q_data)
            k_c = self.pool.alloc_same_pin_tensor(k_data)
            v_c = self.pool.alloc_same_pin_tensor(v_data)
            
            s_c = self.pool.alloc_same_pin_tensor(s_data)
            c_c = self.pool.alloc_same_pin_tensor(c_data)
            # logger.info(f"alloc pin memory cost {time.time()-time_start_alloc:.6f} seconds")
            
        
            if_block=True
            cuda_copy_(q_c, q_data, non_blocking=if_block)
            cuda_copy_(k_c, k_data, non_blocking=if_block)
            cuda_copy_(v_c, v_data, non_blocking=if_block)
            
            cuda_copy_(s_c, s_data, non_blocking=if_block)
            cuda_copy_(c_c, c_data, non_blocking=if_block)

                
            logger.debug(f"move qkv finish {time.time()} cost {time.time()-time_start}")
            # logger.info(type(attn_inputs))
            attn_inputs.query_states = q_c
            attn_inputs.key_states = k_c
            attn_inputs.value_states = v_c
            attn_inputs.sin = s_c
            attn_inputs.cos = c_c
            # logger.info(type(attn_inputs))

            attn_task = AttnTask(
                attn_inputs=attn_inputs,
                type=0,
                task_group_id=getattr(io_task, 'task_group_id', None),
                subtask_idx=getattr(io_task, 'subtask_idx', None)
            )
            self.cqueue.put(attn_task)
            
        # cpu2gpu

        elif io_type == 1:
            # logger.info(f"move attn_output")

            time_start=time.time()
            d_device = self.device

            move_attn_output = io_task.attn_output
            if io_task.attn_inputs is None:
                return
            attn_inputs = io_task.attn_inputs
            
            # move_attn_output not in pin cpu
            attn_cpu = self.pool.alloc_same_pin_tensor(move_attn_output)
            cuda_copy_(attn_cpu, move_attn_output, non_blocking=True)       
            attn_output = attn_cpu.to(device=d_device, non_blocking=True)
            
            attn_out = AttnOut(
                attn_output=attn_output,
                task_group_id=getattr(io_task, 'task_group_id', None),
                subtask_idx=getattr(io_task, 'subtask_idx', None),
                updated_past_key_value=getattr(io_task, 'updated_past_key_value', None)
            )

            self.out_queue.put(attn_out)

                # 记录GPU拷贝完成事件
            gpu_copy_event = torch.cuda.Event(blocking=True)
            gpu_copy_event.record(stream=self.io_stream)
            gpu_copy_event.wait(stream=self.io_stream)
            self.pool.free(attn_cpu)

            # 保留KV数据
            query_states = attn_inputs.query_states
            sin = attn_inputs.sin
            cos = attn_inputs.cos
            self.pool.free(query_states)
            self.pool.free(sin)
            self.pool.free(cos)
class CThread:
    def __init__(self, lpmodule_class: LPModuleWrapper, iothread: QThread, cqueue: Queue[AttnTask]):
        self.lpmodule_class = lpmodule_class
        self.iothread = iothread
        self.cqueue = cqueue
        self.thread = threading.Thread(target=self.run)
    def start(self):
        self.thread.start()
    def run(self):
        logger.info(f"cpu_attn id {get_thread_id()}")
        while True:
            attn_task: AttnTask = self.cqueue.get()
            if attn_task.type < 0:
                return
            self.process(attn_task)
    def submit_task(self, task: AttnTask):
        self.cqueue.put(task)
    def stop(self):
        self.cqueue.put(AttnTask(type=-1))
        return
    def process(self, attn_task: AttnTask):
        
        attn_inputs = attn_task.attn_inputs
        # logger.info(type(attn_inputs))
        # on cpu
        time_start =time.time()

        attn_inputs.attn_input.update({
            "query_states": attn_inputs.query_states,
            "key_states": attn_inputs.key_states,
            "value_states": attn_inputs.value_states,
            "sin": attn_inputs.sin,
            "cos": attn_inputs.cos
        })
        
        attn_input = attn_inputs.attn_input
        attn_result = self.lpmodule_class.decoder_attn(**attn_input)    
        
        # 处理返回值 - 可能是 (attn_output, past_key_value) 或只是 attn_output
        if isinstance(attn_result, tuple) and len(attn_result) == 2:
            attn_output, updated_past_key_value = attn_result
        else:
            attn_output = attn_result
            updated_past_key_value = attn_input.get('past_key_value', None)
        
        logger.debug(f"cpu attn cost {time.time() - time_start}")

        io_type=1
        io_task = IOTask(
            type=io_type,
            attn_output=attn_output,
            attn_inputs=attn_inputs,
            task_group_id=getattr(attn_task, 'task_group_id', None),
            subtask_idx=getattr(attn_task, 'subtask_idx', None),
            updated_past_key_value=updated_past_key_value  # 添加更新后的cache
        )
        self.iothread.submit_task(io_task)