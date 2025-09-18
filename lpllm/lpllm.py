from ast import mod
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
            model_path: str, 
            device: str ="cuda:0", 
            storage_path: Optional[str] = None,
        ):
        self.model_path = model_path
        
        if not storage_path:
            storage_path = os.getenv("STORAGE_PATH", "./models")

        lpmodel_path = f"lpllm.models.{model_path}.lpmodule"
        lpmodule = importlib.import_module(lpmodel_path)

        lpmodule_class: LPModuleWrapper = getattr(lpmodule, "LPModule")
        self.lpmodule_class = lpmodule_class        

        config = AutoConfig.from_pretrained(
            f"{os.path.join(storage_path, model_path)}", trust_remote_code=True
        )
        self.lpmodule_class_instance = lpmodule_class.get_model(config)
        self.lpmodule_class_instance.to(config.torch_dtype)


        self.config = config
        self.device = device

        client = SllmStoreClient("127.0.0.1:8073")
        ret = client.load_into_cpu(model_path)
        if not ret:
            raise ValueError(f"Failed to load model {model_path} into CPU")
        
        
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
            os.path.join(storage_path, model_path, "tensor_index.json"), "r"
        ) as f:
            tensor_index = json.load(f)
            
        # 针对统一文件内容的 offset
        layers_tensor_meta_index = {}
        layers_tensor_data_index = {}
        other_tensor_meta_index = {}
        other_tensor_data_index = {}
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
        
        logger.debug(self.lpmodule_class_instance.model.embed_tokens)

        # return
        layerc = self.get_layer(0)

        layer0_tensor_copy_chunks = self.layers_tensor_copy_chunks[0]
        layer0_tensor_device_offset = self.layers_tensor_device_offset[0]
        layer0_tensor_meta_index = self.layers_tensor_meta_index[0]

        time_start_load = time.time()
        self.cuda_memory_view.load_layer_gpu(layer_idx=0, model_path=self.model_path, 
            layer_tensor_copy_chunks=layer0_tensor_copy_chunks, layer_tensor_meta_index=layer0_tensor_meta_index,
            layer_tensor_device_offsets=layer0_tensor_device_offset
        )
        logger.debug(f"copy chunks: {layer0_tensor_copy_chunks}, device_offset: {layer0_tensor_device_offset}, meta_index: {layer0_tensor_meta_index}")
        main_state_dict = self.cuda_memory_view.get_state_dict(0)
        self.cuda_memory_view.restore_layer(layerc, main_state_dict, layer_loc_index+1)
        self.cuda_memory_view.wait_layer_gpu_loading(layer_idx=0, model_path=self.model_path)
        logger.debug(f"load first layer cost {time.time()-time_start_load}  s")

        layerc1 = self.get_layer(1)

        layer1_tensor_copy_chunks = self.layers_tensor_copy_chunks[1]
        layer1_tensor_device_offset = self.layers_tensor_device_offset[1]
        layer1_tensor_meta_index = self.layers_tensor_meta_index[1]
        time_start_load = time.time()
        self.cuda_memory_view.load_layer_gpu(layer_idx=1, model_path=self.model_path, 
            layer_tensor_copy_chunks=layer1_tensor_copy_chunks, layer_tensor_meta_index=layer1_tensor_meta_index,
            layer_tensor_device_offsets=layer1_tensor_device_offset
        )
        
        main_state_dict = self.cuda_memory_view.get_state_dict(1)
        # time_start_restore = time.time()
        self.cuda_memory_view.restore_layer(layerc1, main_state_dict, layer_loc_index+1)
        # logger.debug(f"restore second layer cost {time.time()-time_start_restore}  s")
        self.cuda_memory_view.wait_layer_gpu_loading(layer_idx=1, model_path=self.model_path)
        self.layerc1 = layerc1

        logger.debug(f"load second layer cost {time.time()-time_start_load}  s")

        self.attn_manager = AttnManager(lpmodule_class=lpmodule_class, device=device, config=config, 
            pool_size=45, use_server_pool=False
        )
        self.attn_manager.start()
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
    def get_layer(self, layer_idx):
        if self.lpmodule_class.check_layer_is_dense(self.config, layer_idx):
            return self.layer_dense
        else:
            return self.layer_moe
        
    def wait_load_into_gpu(self, layer_idx):
        self.cuda_memory_view.wait_layer_gpu_loading(layer_idx=layer_idx, model_path=self.model_path)
    def decoder_qkv(
        self,
        layer,
        hidden_states,
        past_key_value,
        attention_mask,
        position_ids,
    ):
        (query_states, key_states, value_states, sin, cos) = self.lpmodule_class.decoder_qkv(
            layer=layer,
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return (query_states, key_states, value_states, sin, cos)
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
        layer_attn = self.get_layer(layer_idx=layer_attn_idx)
        layer_mlp = self.get_layer(layer_idx=layer_mlp_idx)
        layer_attr_name, layer_loc_index, layer_attn_name, layer_mlp_name = self.lpmodule_class.get_layer_attr_info()
        # the last one in queue
        layer_attr_loc = layer_loc_index + 1
        # all related layer should be ready
        if j_loc % 2 == 1:
            # change attn, mlp not change, could get from class
            update_str = "self_attn"
            update_state_dict = self.cuda_memory_view.updateState(layer_idx=layer_attn_idx, update_name=update_str, if_mlp=False)
            self.cuda_memory_view.restore_layer(layer_attn, update_state_dict, layer_attr_loc=layer_attr_loc)
            layer_attn.layer_idx = layer_mlp_idx
        else:
            # change mlp, attn not change, could get from class
            update_str = "experts"
            update_state_dict = self.cuda_memory_view.updateState(layer_idx=layer_mlp_idx, update_name=update_str, if_mlp=True)
            self.cuda_memory_view.restore_layer(layer_mlp, update_state_dict, layer_attr_loc=layer_attr_loc)
            layer_mlp.layer_idx = layer_mlp_idx
        return layer_attn, layer_mlp

        

    def sync(self):
        torch.cuda.synchronize(device=self.device)
        pass
    def decoder_attn_call(self, layer, layer_idx, hidden_states, past_key_value, attention_mask, position_ids,
        query_states, key_states, value_states, sin, cos
    ):
        bsz, q_len = hidden_states.shape[:2]
        
        # 决定并行度：可以根据batch size调整
        num_parallel = min(bsz, 4)  # 最多4个并行任务，避免过度拆分
        
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
                past_key_value, position_ids=position_ids
            )
        return
    @torch.no_grad
    def decoders(
        self,
        hidden_states1,
        hidden_states2,
        past_key_value1,
        past_key_value2,
        attention_mask1,
        attention_mask2,
        position_ids1,
        position_ids2
    ):
        layer_num = self.config.num_hidden_layers
        
        
        
        # here layer
        cur_layer_idx = 1
        layerc = self.get_layer(layer_idx=cur_layer_idx)
        
        layer_output1 = None
        layer_output2 = None
        
        hidden_states_attn=hidden_states1
        
        past_key_value=past_key_value1
        attention_mask=attention_mask1
        position_ids=position_ids1
        
        # here layer
        # self.start_load_into_gpu(cur_layer_idx+1)
        

        logger.debug(f"start first layer")

        (query_states, key_states, value_states, sin, cos) = self.decoder_qkv(
            layer=layerc,
            hidden_states=hidden_states_attn,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

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
        )
        
        # wait the attention finish
        # last_attn_output = self.async_get_decoder_attn()
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
                elif cur_layer_idx < layer_num - 1 and cur_layer_idx >= 1:
                    load_next_layer=True
                need_wait_layer=False
            else: 
                load_next_layer=False
                if cur_layer_idx < layer_num - 1 and cur_layer_idx >= 1:
                    need_wait_layer=True
                
                layer_mlp_idx += 1
                past_key_value=past_key_value2
                attention_mask=attention_mask2
                position_ids=position_ids2
            
            logger.debug(f"\none decoder loop j: {j} cur_layer_idx: {cur_layer_idx}")

            # change to mlp call loc
            # hidden_states_mlp = last_attn_input
            # hidden_states_mlp_o = last_attn_output
            
            hidden_states_attn = last_mlp_output
            
            # here layer
            if load_next_layer:
                logger.debug(f"start load next layer cur_layer_idx: {cur_layer_idx+1}")
                self.start_load_into_gpu(cur_layer_idx+1)
                
            logger.debug(f"start decoder qkv layer_attn {layer_attn_idx} layer_mlp {layer_mlp_idx}")

            (query_states, key_states, value_states, sin, cos) = self.decoder_qkv(
                layer=layer_attn,
                hidden_states=hidden_states_attn,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            
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
            )

            # get last attn_output in here
            last_attn_output = self.async_get_decoder_attn()
            hidden_states_mlp = last_attn_input
            hidden_states_mlp_o = last_attn_output
            # attn_func()
            mlp_output=self.decoder_mlp(
                mlp_layer=layer_mlp,
                mlp_hidden_states=hidden_states_mlp,
                mlp_o_hidden_states=hidden_states_mlp_o,
                attn_func=attn_func
            )    
            
            
            # here layer
            if not (cur_layer_idx == layer_num - 1 and need_wait_layer == False):
                layer_attn, layer_mlp = self.reset_next_layer_need(layer_attn_idx=layer_attn_idx, layer_mlp_idx=layer_mlp_idx, j_loc=j)
            logger.debug(f"start reset next layer layer_attn {layer_attn_idx} layer_mlp {layer_mlp_idx}")

            logger.debug(f"start async get decoder attn")
            last_attn_input=hidden_states_attn
            # last_attn_output = self.async_get_decoder_attn()
            last_mlp_output=mlp_output


            # here layer
            if need_wait_layer:
                # waiting here
                time_start_waiting = time.time()
                self.wait_load_into_gpu(cur_layer_idx+1)
                logger.debug(f"j: {j} waiting the layer with layer_idx {cur_layer_idx+1} load cost {time.time()-time_start_waiting} s")

            # here layer
            if load_next_layer:
                self.cuda_memory_view.free_layer_gpu(cur_layer_idx-1)

            # self.sync()
        logger.debug(f"start decoder mlp last layer")
        mlp_output=self.decoder_mlp(
            mlp_layer=layerc,
            mlp_hidden_states=hidden_states_mlp,
            mlp_o_hidden_states=hidden_states_mlp_o,
            attn_func=None
        )    
        
        self.sync()
        layer_output1=last_mlp_output
        layer_output2=mlp_output
        
        return layer_output1, layer_output2
        
    def generate(**inputs):
        pass
    def forward_prepare(self, input_ids, position_ids, past_key_values, output_attentions, use_cache, attention_mask):
        return self.lpmodule_class.forward_prepare(
            model=self.lpmodule_class_instance, input_ids=input_ids, 
            position_ids=position_ids, past_key_values=past_key_values, 
            output_attentions=output_attentions, use_cache=use_cache, attention_mask=attention_mask
        )
    @torch.no_grad
    def split_decoders(self, input_ids):
        split_input_ids = input_ids.split(input_ids.size(0) // 2, dim = 0)
        input_ids1 = split_input_ids[0]
        input_ids2 = split_input_ids[1]

        (input_embeds1, attention_mask1,
        past_key_values1, position_ids1) \
        =self.forward_prepare(
            input_ids=input_ids1,
            position_ids=None,
            past_key_values=None,
            output_attentions=None,
            use_cache=True,
            attention_mask=None,
        )

        (input_embeds2, attention_mask2,
        past_key_values2, position_ids2) \
        =self.forward_prepare(
            input_ids=input_ids2,
            position_ids=None,
            past_key_values=None,
            output_attentions=None,
            use_cache=True,
            attention_mask=None,
        )
        
        logger.debug(f"input_embeds shapes: {input_embeds1.shape}, {input_embeds2.shape}")
        # logger.debug(f"attention_mask shapes: {attention_mask1.shape}, {attention_mask2.shape}")
        # logger.debug(f"position_ids shapes: {position_ids1.shape}, {position_ids2.shape}")

        layer_output1, layer_output2 = self.decoders(
            hidden_states1=input_embeds1,
            hidden_states2=input_embeds2,
            past_key_value1=past_key_values1,
            past_key_value2=past_key_values2,
            attention_mask1=attention_mask1,
            attention_mask2=attention_mask2,
            position_ids1=position_ids1,
            position_ids2=position_ids2
        )
        
        layer_output = torch.cat([layer_output1, layer_output2], dim=0)
        # if attention_mask1 is not None:
        #     attention_mask = torch.cat([attention_mask1, attention_mask2], dim=0)
        #     logger.debug(f"attention_mask {attention_mask1.shape} {attention_mask2.shape}")
        # else:
        #     attention_mask = None
        # if isinstance(past_key_values1, Cache) and isinstance(past_key_values2, Cache):
        #     # Handle Cache objects by concatenating their key_cache and value_cache separately
        #     past_key_values = DynamicCache()
            
        #     # Debug shapes before concatenation
        #     logger.debug(f"key_cache lengths: {len(past_key_values1.key_cache)} vs {len(past_key_values2.key_cache)}")
        #     for i, (k1, k2) in enumerate(zip(past_key_values1.key_cache, past_key_values2.key_cache)):
        #         logger.debug(f"Layer {i} key shapes: {k1.shape} vs {k2.shape}")
            
        #     try:
        #         past_key_values.key_cache = [torch.cat([k1, k2], dim=0) for k1, k2 in zip(past_key_values1.key_cache, past_key_values2.key_cache)]
        #         past_key_values.value_cache = [torch.cat([v1, v2], dim=0) for v1, v2 in zip(past_key_values1.value_cache, past_key_values2.value_cache)]
        #     except RuntimeError as e:
        #         logger.error(f"Failed to concatenate key/value caches: {e}")
        #         # If concatenation fails, we might need to handle this differently
        #         # For now, just use the first cache as fallback
        #         logger.warning("Using first cache as fallback")
        #         past_key_values = past_key_values1
        # else:
        #     past_key_values = torch.cat([past_key_values1, past_key_values2], dim=0)
        # position_ids = torch.cat([position_ids1, position_ids2], dim=0)
        
        attention_mask = None
        past_key_values = None
        position_ids = None
        return layer_output, attention_mask, past_key_values, position_ids
    def generate(self, input_ids):
        pass
    def stop(self):
        self.attn_manager.stop()
    def __def__(self):
        self.attn_manager.stop()
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
            set_module_tensor_to_device(model, name, param.device, param)
        # buffer_names = [name for name, _ in model.named_buffers()] 
        # logger.debug(f"{buffer_names}")
        # for name, param in model_state_dict.items():
        #     names = name.split(".")
        #     name = ".".join(names[:-2])
        #     logger.debug(f"{name}")
        #     set_module_buffer_to_device(model, name, param.device)
        
        model.eval()
    def restore_layer(self, layer, state_dict, layer_attr_loc):
        for name, param in state_dict.items():
            relative_layer_name = ".".join(name.split(".")[layer_attr_loc:])

            set_module_tensor_to_device(layer, relative_layer_name, param.device, param)
        # send_module_buffers_to_device(layer, {"": self.device_index})
        layer.eval()
        # send_module_buffers_to_device(model, device_map)

    def get_state_dict(self, layer_idx: int):
        for state_view in self.memory_queue_allocated.queue:
            if state_view.index == layer_idx:
                state_dict = state_view.state_dict
                return state_dict.copy()
        raise ValueError(f"state dict should not be empty or layer_idx not match")
    def updateState(self, layer_idx: int, update_name: str, if_mlp: bool):
        # the loc in queue
        if if_mlp:
            update_state_loc = self.get_state_loc_mlp()
        else:
            update_state_loc = self.get_state_loc_attn()
        update_state_loc = -1
        for i in range(len(self.memory_queue_allocated.queue)):
            if self.memory_queue_allocated.queue[i].index == layer_idx:
                update_state_loc = i
                break
        if update_state_loc == -1:
            raise ValueError(f"update with {layer_idx} but get {update_state_loc} not match")
        update_state_dict = self.memory_queue_allocated.queue[update_state_loc].state_dict

        ustate_dict = {}
        for src_name, tensor in update_state_dict.items():
            if update_name in src_name:
                ustate_dict[src_name] = tensor
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
@dataclass
class AttnOutputs:
    attn_output: torch.Tensor
@dataclass
class AttnTask:
    type: int
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
        pool_size: int=2, use_server_pool: bool = True
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
            self.pool_memory = PinnedMemoryPool(config.torch_dtype, pool_size)

        # 分离的队列系统实现真正并行
        self.out_queue: Queue[AttnOut] = Queue()
        self.gpu2cpu_queue: Queue[IOTask] = Queue()  # GPU→CPU移动队列
        self.cpu_compute_queue: Queue[AttnTask] = Queue()  # CPU计算队列  
        self.cpu2gpu_queue: Queue[IOTask] = Queue()  # CPU→GPU移动队列
        
        # 三个独立线程实现并行处理
        self.gpu2cpu_thread = GPU2CPUThread(
            device=device, pool_memory=self.pool_memory, 
            input_queue=self.gpu2cpu_queue, output_queue=self.cpu_compute_queue)
        self.cpu_compute_thread = CPUComputeThread(
            lpmodule_class=lpmodule_class, 
            input_queue=self.cpu_compute_queue, output_queue=self.cpu2gpu_queue)
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
        self.gpu2cpu_thread.stop()
        self.cpu_compute_thread.stop()
        self.cpu2gpu_thread.stop()
    def submit_task(self, bsz, q_len, layer, layer_idx, query_states, key_states, value_states, sin, cos, attention_mask, past_key_value, position_ids):
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
    def __init__(self, device: str, pool_memory: PinnedMemoryPool, input_queue: Queue[IOTask], output_queue: Queue[AttnTask]):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.io_stream = torch.cuda.Stream(device=device)
        self.thread = threading.Thread(target=self.run)
        self.device = device
        self.pool = pool_memory
        self._stop_event = threading.Event()
        
    def start(self):
        self.thread.start()
        
    def run(self):
        logger.debug(f"GPU2CPU thread started, id {get_thread_id()}")
        with torch.cuda.stream(self.io_stream):
            while not self._stop_event.is_set():
                try:
                    io_task: IOTask = self.input_queue.get(timeout=1.0)
                    if io_task.type < 0:  # 停止信号
                        break
                    self.process_gpu2cpu(io_task)
                except Empty:
                    continue
                    
    def process_gpu2cpu(self, io_task: IOTask):
        """处理GPU到CPU的数据移动"""
        if io_task.attn_inputs is None:
            return
            
        attn_inputs = io_task.attn_inputs
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
        
        if_nonblcok  = True 

        cuda_copy_(q_c, q_data, non_blocking=if_nonblcok)
        cuda_copy_(k_c, k_data, non_blocking=if_nonblcok) 
        cuda_copy_(v_c, v_data, non_blocking=if_nonblcok)
        cuda_copy_(s_c, s_data, non_blocking=if_nonblcok)
        cuda_copy_(c_c, c_data, non_blocking=if_nonblcok)
        
        logger.debug(f"GPU2CPU move cost {time.time()-time_start:.6f} seconds")
        
        # 更新attn_inputs中的数据
        attn_inputs.query_states = q_c
        attn_inputs.key_states = k_c
        attn_inputs.value_states = v_c
        attn_inputs.sin = s_c
        attn_inputs.cos = c_c
        
        # 提交到CPU计算队列
        attn_task = AttnTask(
            type=0,
            attn_inputs=attn_inputs,
            task_group_id=getattr(io_task, 'task_group_id', None),
            subtask_idx=getattr(io_task, 'subtask_idx', None)
        )
        self.output_queue.put(attn_task)
        
    def submit_task(self, task: IOTask):
        self.input_queue.put(task)
        
    def stop(self):
        self._stop_event.set()
        self.input_queue.put(IOTask(type=-1))

class CPUComputeThread:
    """专门处理CPU上的注意力计算"""
    def __init__(self, lpmodule_class: LPModuleWrapper, input_queue: Queue[AttnTask], output_queue: Queue[IOTask]):
        self.lpmodule_class = lpmodule_class
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.thread = threading.Thread(target=self.run)
        self._stop_event = threading.Event()
        torch.set_num_threads(128)  # Set intra-op threads
        
        
    def start(self):
        self.thread.start()
        
    def run(self):
        logger.info(f"CPU compute thread started, id {get_thread_id()}")
        while not self._stop_event.is_set():
            try:
                attn_task: AttnTask = self.input_queue.get(timeout=1.0)
                if attn_task.type < 0:  # 停止信号
                    break
                self.process_cpu_compute(attn_task)
            except Empty:
                continue
                
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

        time_start_attn = time.time()
        attn_result = self.lpmodule_class.decoder_attn(**attn_input)
        logger.debug(f"CPU attn cost {time.time() - time_start_attn:.6f} seconds")

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
        io_task = IOTask(
            type=1,
            attn_output=attn_output,
            attn_inputs=attn_inputs,
            task_group_id=getattr(attn_task, 'task_group_id', None),
            subtask_idx=getattr(attn_task, 'subtask_idx', None),
            updated_past_key_value=updated_past_key_value
        )
        self.output_queue.put(io_task)
        
    def stop(self):
        self._stop_event.set()
        self.input_queue.put(AttnTask(type=-1))

class CPU2GPUThread:
    """专门处理CPU到GPU的数据移动"""
    def __init__(self, device: str, pool_memory: PinnedMemoryPool, input_queue: Queue[IOTask], output_queue: Queue[AttnOut]):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.io_stream = torch.cuda.Stream(device=device)
        self.thread = threading.Thread(target=self.run)
        self.device = device
        self.pool = pool_memory
        self._stop_event = threading.Event()
        
    def start(self):
        self.thread.start()
        
    def run(self):
        logger.debug(f"CPU2GPU thread started, id {get_thread_id()}")
        with torch.cuda.stream(self.io_stream):
            while not self._stop_event.is_set():
                try:
                    io_task: IOTask = self.input_queue.get(timeout=1.0)
                    if io_task.type < 0:  # 停止信号
                        break
                    self.process_cpu2gpu(io_task)
                except Empty:
                    continue
                    
    def process_cpu2gpu(self, io_task: IOTask):
        """处理CPU到GPU的数据移动"""
        if io_task.attn_inputs is None:
            return
            
        move_attn_output = io_task.attn_output
        attn_inputs = io_task.attn_inputs
        
        time_start = time.time()
        

        if_nonblcok = True
        # 移动注意力输出到GPU - 优化CPU到CPU拷贝
        attn_cpu = self.pool.alloc_same_pin_tensor(move_attn_output)
        
        # 优化策略：使用最优拷贝方法（基于性能测试结果）
        if move_attn_output.is_contiguous():
            # 已经是连续内存，直接使用 non-blocking copy（最快）
            attn_cpu.copy_(move_attn_output, non_blocking=if_nonblcok)
        else:
            # 非连续内存，先转换为连续再拷贝
            move_attn_output = move_attn_output.contiguous()
            attn_cpu.copy_(move_attn_output, non_blocking=if_nonblcok)
        
        attn_output = attn_cpu.to(device=self.device, non_blocking=if_nonblcok)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(
            attn_inputs.query_states.shape[0], #bsz
            attn_inputs.query_states.shape[2], #q_len
            attn_inputs.query_states.shape[1] * attn_inputs.query_states.shape[3] #hidden_size
        )

        # 等待GPU拷贝完成
        gpu_copy_event = torch.cuda.Event(blocking=True)
        gpu_copy_event.record(stream=self.io_stream)
        gpu_copy_event.wait(stream=self.io_stream)
        
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
        self.pool.free(attn_cpu)
        query_states = attn_inputs.query_states
        sin = attn_inputs.sin
        cos = attn_inputs.cos
        self.pool.free(query_states)
        self.pool.free(sin)
        self.pool.free(cos)
        
    def stop(self):
        self._stop_event.set()
        self.input_queue.put(IOTask(type=-1))

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