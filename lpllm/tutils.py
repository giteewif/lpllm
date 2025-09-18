import threading

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