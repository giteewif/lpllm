import threading
import time, torch
from lpllm.logger import init_logger
logger = init_logger(__name__)
def calculate_model_memory(model_tensor_index):
    all_model_memory = 0
    for name, (_, size) in model_tensor_index.items():
        all_model_memory += size
    return all_model_memory

def calculate_layer_memory(layers_tensor_index, layer_attr, mlp_attr):
    all_qkv_memory = 0
    max_layer_memory = 0
    for layer_index, layer_tensor_data_index in layers_tensor_index.items():
        layer_memory = 0
        for name, (_, size) in layer_tensor_data_index.items():
            if mlp_attr not in name:
                all_qkv_memory += size
            if layer_attr in name:
                layer_memory += size
            
        if layer_memory > max_layer_memory:
            max_layer_memory = layer_memory
    return all_qkv_memory, max_layer_memory

# only for single device here, multiple device in sllm_store caculate_tensor_device_offsets, for layer and qkv
def calculate_device_offset(tensor_index, device_idx):
    device_offset = 0
    tensor_device_offsets = {}
    tensor_copy_chunks = {}
    tensor_copy_chunks[device_idx] = []
    tensor_device_offsets[device_idx] = {}
    single_device_offset = tensor_device_offsets[device_idx]
    single_copy_chunks_list = tensor_copy_chunks[device_idx]
    for name, (offset, size) in tensor_index.items():
        single_device_offset[name] = device_offset
        single_copy_chunks_list.append(
            (offset, size, device_offset, 0)
        )
        device_offset += size
    return tensor_device_offsets, tensor_copy_chunks, device_offset

def get_thread_id():
        thread_id = threading.get_native_id()
        return thread_id

def scaled_dot_product_attention_help(
    query_states, 
    key_states, 
    value_states, 
    attn_mask=None, dropout_p=0.0, enable_gqa=False, is_causal=False, output_tensor=None):

    time_start = time.time()
   
    num_query_heads = query_states.shape[1]     # e.g. 32
    num_key_heads = key_states.shape[1]
    num_groups = int(num_query_heads//num_key_heads)   # 4 组
    if output_tensor is None:
        output_tensor = torch.zeros(
            query_states.shape, dtype=query_states.dtype, device=query_states.device, pin_memory=False
        )
    for group_idx in range(num_groups):
        # 不扩展kv，而是使用整个kv，query取值需变化
        # 每组使用整个key/value heads，但query heads按步数提取
        # 例如：32个query heads，8个key/value heads
        # 按步数提取query heads
        query_indices = torch.arange(group_idx, num_query_heads, num_groups)
        query_group = query_states[:, query_indices, :, :]  # (batch, heads_per_group, seq_len, head_dim)
        logger.debug(query_group.shape)
        key_group = key_states    # (batch, 8, seq_len, head_dim)
        value_group = value_states # (batch, 8, seq_len, head_dim)

        time_start_tmp = time.time()
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            query_group, key_group, value_group,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            enable_gqa=enable_gqa,
            is_causal=is_causal
        )
        logger.debug(f"single group {group_idx} real attn out cost {time.time() - time_start_tmp} s")
        # print(f"Group {group_idx}: query_indices={query_indices.tolist()}, kv_heads=all, attn_out.shape={attn_out.shape}")
        # instead of simply appending to a list, collect all group results in the correct head positions
        # (batch, num_query_heads, seq_len, head_dim) for all groups
        
        output_tensor[:, query_indices, :, :] = attn_out

    logger.debug(f"dot attn help cost {time.time()-time_start:.6f} seconds")
    return output_tensor