from lpllm.utils import get_torch_gpu_memory, get_current_process_gpu_memory,get_gpu_memory_usage
from lpllm.pinpool import FixedSizePinnedMemoryPool, PinnedMemoryPool
import importlib
import os
import gc
import torch
import threading
from threading import Thread
from queue import Queue, Empty
from transformers.cache_utils import Cache, DynamicCache
from .task_struct import IO_Task
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import time

class LPLLM():
    def __init__(
        self, 
        model_name: str, 
        model_path: str, 
        class_name: str,
        device: str
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.class_name = class_name
        self.device=device
        self.running=False
        # config.json
        
        get_gpu_memory_usage("before load class")
        self.model = self._load_model_class()
        self.model.eval()
        get_gpu_memory_usage("after load class")
        
        self.config=self.get_config()
        self.layer_num=self.config.num_hidden_layers
        
        if device != "cuda:1":
            raise ValueError(
                f"Should use cuda:1 for test"
            )
        # self.compute_stream = torch.cuda.Stream(device=device)
        self.io_stream = torch.cuda.Stream(device=device)
        self.mlp_stream=torch.cuda.Stream(device=device)
        
        self.io_queue = Queue()
        self.layer_io_queue = Queue()
        self.cpu_queue = Queue()
        self.out_queue=Queue()
        self.layer_notify=Queue()
        
        self.attn_weights_loading=False       
        self.has_pause_layer_weights=False 
        # [0] start the layer1 tensor in deepseek
        self.layers_tensor=self.load_layers_state_dict(self.model_path)
        
        torch_dytpe = self.config.torch_dtype
        self.pool = PinnedMemoryPool(pool_size=4096, dtype=torch_dytpe)
        
        
        get_gpu_memory_usage("before load one layer")
        self.layerc = self.get_layer(
            layer_idx=1, 
            device=self.device, 
            dtype=self.config.torch_dtype
        )
        get_gpu_memory_usage("after load one layer")
        self.layern = self.get_empty_layer(
            layer_idx=2,
            device=self.device,
            dtype=self.config.torch_dtype
        )
        get_gpu_memory_usage("after load second layer")
    def _load_model_class(self):
        """动态加载模型实现类"""
        try:
            # 假设模型实现类在 models 目录下
            module_path = f"lpllm.models.{self.model_name.lower()}"
            module = importlib.import_module(module_path)
            
            # 获取类
            model_class = getattr(module, self.class_name)
            
            return model_class(path=self.model_path, device=self.device)
        except (ImportError, AttributeError) as e:
            raise Exception(f"Failed to load model class for {self.model_name}: {e}")
    def load_others_tensor(self, path):
        weight_tensor = self.model.load_others_tensor(path)
        return weight_tensor
    def load_layers_tensor(self, path):
        layers_tensor = self.model.load_layers_tensor(path)
        return layers_tensor
    
    def load_weight2layer_general(self, layer, layer_idx, layer_tensor):
        self.model.load_weight2layer_general(layer, layer_idx, layer_tensor)
    def load_layers_state_dict(self, path):
        layers_state_dict=self.model.load_layers_tensor_general(path)
        return layers_state_dict
    def load_layer(self, layer, layer_idx):
        with torch.cuda.Stream(self.io_stream):
            tensor = self.load_layer_tensor(layer_idx, self.model_path)
            self.load_weight2layer(layer, layer_idx, tensor)
            
    @staticmethod
    def access_by_string(layer, attr_string):
        attrs = attr_string.split('.')
        obj =  layer
        
        for attr in attrs:
            if attr.isdigit():  # 数字索引
                obj = obj[int(attr)]
            else:  # 属性访问
                obj = getattr(obj, attr)
        
        return obj

    def get_layer_chunks(self, layer_state_dict):

        chunks_load_array = []
        # 16MB
        chunk_size = 16*1024*1024
        for key, value in layer_state_dict.items():
            # get like self_attn.v_proj.weight
            # weight_obj = LPLLM.access_by_string(layer, key)
            weight_tensor = value
            
            tensor_memory = weight_tensor.element_size() * weight_tensor.numel()

            if tensor_memory > chunk_size and weight_tensor.dim() > 0:
                # 沿着第一个维度分割张量
                chunk_dim_size = chunk_size // (weight_tensor.element_size() * weight_tensor[0].numel())

                if chunk_dim_size > 0:
                    chunks = torch.split(weight_tensor, chunk_dim_size, dim=0)

                    # 为每个分块创建元数据
                    for i, chunk in enumerate(chunks):
                        start_idx = i * chunk_dim_size
                        end_idx = min((i + 1) * chunk_dim_size, weight_tensor.size(0))
                        
                        chunk_info = {
                            'key': key,
                            'chunk': chunk,
                            'indices': (start_idx, end_idx),
                            'original_shape': weight_tensor.shape,
                            'slice_position': (i, len(chunks))
                        }
                else:
                    chunk_info = {
                        'key': key,
                        'chunk': weight_tensor,
                        'indices': None,
                        'original_shape': weight_tensor.shape,
                        'slice_position': None
                    }
            else:
                # 对于小张量或0维张量，直接添加
                chunk_info = {
                    'key': key,
                    'chunk': weight_tensor,
                    'indices': None,
                    'original_shape': weight_tensor.shape,
                    'slice_position': None
                }
            chunks_load_array.append(chunk_info)
        return chunks_load_array    
            
    def load_layer_concurrent(self, layer, layer_idx):
        layer_state_dict = self.layers_tensor[layer_idx]
        layer_chunks = self.get_layer_chunks(layer_state_dict=layer_state_dict)
        layer_io_task = {
            "layer": layer,
            "layer_idx": layer_idx,
            "chunks": layer_chunks
        }
        self.layer_io_queue.put(layer_io_task)
        
    def load_layer_tensor(self, layer_idx, path):
        layer_tensor = self.model.load_layer_tensor(layer_idx, path)
        return layer_tensor
    def load_weight2layer(self, layer, layer_idx, layer_tensor):
        self.model.load_weight2layer(layer, layer_idx, layer_tensor)
    def get_layer(
        self, 
        layer_idx: int, 
        device: str, 
        dtype: str
    ):
        if layer_idx >= self.layer_num:
            raise ValueError(
                f"layer_idx error"
            )
        layer_tensor = self.layers_tensor[layer_idx]
        layer = self.model.get_layer(self.config, layer_idx)
        self.model.load_weight2layer_general(layer, layer_idx, layer_tensor)
        
        layer.eval()
        layer.to(device=device, dtype=dtype)
        return layer
    def get_empty_layer(
        self, 
        layer_idx: int,
        device: str, 
        dtype: str
    ):
        layer=self.model.get_layer(self.config, layer_idx)
        layer.eval()
        layer.to(device=device, dtype=dtype)
        return layer
    def get_config(self):
        return self.model.load_config(self.model_path)
    
    def stop_attn(self):
        cpu_task = {
            "type": -1
        }
        self.cpu_queue.put(cpu_task)
    def cpu_attn(self):
        print(f"cpu_attn id {self.get_thread_id()}")
        while self.running:
            attn_task = self.cpu_queue.get()
            
            type = attn_task["type"]
            if type < 0:
                break
            
            attn_inputs = attn_task["attn_inputs"]
            # print(type(attn_inputs))
            # on cpu
            time_start =time.time()
            
            ev = attn_inputs.pop("copy_event", None)
            if ev is not None:
                ev.synchronize()
                
            attn_output=self.model.attn(**attn_inputs)    
            
            print(f"cpu attn cost {time.time() - time_start}")
            io_type=1
            
            io_map = {
                "type": io_type,
                "attn_output": attn_output,
                "d_device": self.device,
                "attn_inputs": attn_inputs
            }
            self.io_queue.put(io_map)

    def has_attn_weights_loading(self):
        return self.attn_weights_loading  
    def set_attn_weights_loading(self):
        self.attn_weights_loading=True
    def unset_attn_weights_loading(self):
        self.attn_weights_loading=False
        
    def notify_layer_weights_loading(self):
        if self.has_pause_layer_weights:
            self.layer_notify.put(1)
            self.has_pause_layer_weights=False
    def pause_layer_weights_loading(self):
        # waiting for queue
        self.has_pause_layer_weights=True
        self.layer_notify.get()  
        
    def stop_io_layer(self):
        layer_stop = {
            "layer_idx": -1,
        }
        self.layer_io_queue.put(layer_stop)
    def release_attn_inputs(self, attn_inputs):
        q_data = attn_inputs["query_states"]
        s_data = attn_inputs["sin"]
        c_data = attn_inputs["cos"]
        
        if q_data != None:
            self.pool.free(q_data)
        if s_data != None:
            self.pool.free(s_data)
        if c_data != None:
            self.pool.free(c_data)
        
    def io_layer_func(self):
        print(f"setup layer io func")
        print(f"io_layer_func id {self.get_thread_id()}")
        while self.running:
            layer_io = self.layer_io_queue.get()
            
            layer_idx = layer_io["layer_idx"]
            if layer_idx < 0:
                break
            layer = layer_io["layer"]
            chunks = layer_io["chunks"]
            
            time_start = time.time()
            chunk_load_time_list = []
            # print("start load layer")
            # with torch.cuda.Stream(self.io_stream):
            #     for chunk_info in chunks:
            #         key = chunk_info['key']
            #         chunk = chunk_info['chunk']
            #         indices = chunk_info['indices']
                    
            #         # 获取目标weight对象
            #         weight_obj = LPLLM.access_by_string(layer, key)
                    
            #         if self.has_attn_weights_loading():
            #             self.pause_layer_weights_loading()
                    
            #         time_single_start = time.time()
            #         # 如果有索引信息，说明是分块数据，需要复制到特定位置
            #         if indices is not None:
            #             start_idx, end_idx = indices
            #             # 将分块数据复制到目标张量的相应位置
            #             weight_obj.data[start_idx:end_idx].copy_(chunk)
            #         else:
            #             # 直接复制整个张量
            #             weight_obj.data.copy_(chunk)
            #         chunk_load_time_list.append(time.time()-time_single_start)
            layer.layer_idx = layer_idx
            # print(f"load layer {layer_idx} duration {time.time()-time_start} avg {sum(chunk_load_time_list)/len(chunk_load_time_list)}")
    def stop_io(self):
        io_stop = {
            "type": -1,
        }
        self.io_queue.put(io_stop)

    def get_thread_id(self):
        thread_id = threading.get_native_id()
        return thread_id
    def io_func(self):
        print(f"setup io func")
        # f_layer = open("io_layer.log", "w")
        print(f"io_func id {self.get_thread_id()}")
        while self.running:
            # 立刻获取一次，然后阻塞获取
            try:
                io_task = self.io_queue.get(block=False)
            except Empty:
                self.unset_attn_weights_loading()
                self.notify_layer_weights_loading()
                io_task = self.io_queue.get()
                
            self.set_attn_weights_loading()
            # io_task = self.io_queue.get()
            io_type=io_task["type"]
            io_type=int(io_type)
            if io_type < 0:
                break
            # gpu2cpu
            if io_type == 0: 
                attn_inputs = io_task["attn_inputs"]
                q_data = attn_inputs["query_states"]
                k_data = attn_inputs["key_states"]
                v_data = attn_inputs["value_states"]
                s_data = attn_inputs["sin"]
                c_data = attn_inputs["cos"]
                time_start = time.time()
                # print(f"move qkv map")
                # torch.cuda.synchronize(device=self.device)
                # 将数据从GPU移动到CPU
                
                # with same shape
                time_start_alloc = time.time()
                q_c = self.pool.alloc_same_pin_tensor(q_data)
                k_c = self.pool.alloc_same_pin_tensor(k_data)
                v_c = self.pool.alloc_same_pin_tensor(v_data)
                
                s_c = self.pool.alloc_same_pin_tensor(s_data)
                c_c = self.pool.alloc_same_pin_tensor(c_data)
                print(f"alloc pin memory cost {time.time()-time_start_alloc:.6f} seconds")
                
                with torch.cuda.Stream(self.io_stream):
                    if_block=True
                    q_c.copy_(q_data, non_blocking=if_block)
                    k_c.copy_(k_data, non_blocking=if_block)
                    v_c.copy_(v_data, non_blocking=if_block)
                    
                    s_c.copy_(s_data, non_blocking=if_block)
                    c_c.copy_(c_data, non_blocking=if_block)
                    
                    
                    ev = torch.cuda.Event()
                    ev.record(stream=self.io_stream)
                    
                    # q_c = q_data.to("cpu", non_blocking=True)
                    # k_c = k_data.to("cpu", non_blocking=True)
                    # v_c = v_data.to("cpu", non_blocking=True)

                    # s_c = s_data.to("cpu", non_blocking=True)
                    # c_c = c_data.to("cpu", non_blocking=True)
                # torch.cuda.synchronize(device=self.device)
                print(f"move qkv finish {time.time()} cost {time.time()-time_start}")
                # print(type(attn_inputs))
                attn_inputs.update({
                    "query_states": q_c, 
                    "key_states": k_c, 
                    "value_states": v_c, 
                    "sin": s_c, "cos": c_c,
                    "copy_event": ev
                })
                # print(type(attn_inputs))
                attn_task = {
                    "attn_inputs": attn_inputs,
                    "type": 0
                }
                self.cpu_queue.put(attn_task)
                
            # cpu2gpu
            elif io_type == 1:
                # print(f"move attn_output")
                time_start=time.time()
                move_attn_output = io_task["attn_output"]
                d_device = io_task["d_device"]
                attn_inputs = io_task["attn_inputs"]
                with torch.cuda.Stream(self.io_stream):
                    attn_output = move_attn_output.to(device=d_device, non_blocking=True)
                # torch.cuda.synchronize(device=self.device)
                print(f"move attn_output finish {time.time()} {time.time()-time_start}")
                out_map = {
                    "attn_output": attn_output,
                    "attn_inputs": attn_inputs
                }
                self.out_queue.put(out_map)
            # else:
                
            #     layer=io_task["layer"]
            #     layer_idx=io_task["layer_idx"]
            #     time_start = time.time()
            #     print(f"load layer {layer_idx} start")
            #     if layer_idx >= self.layer_num:
            #         raise ValueError(
            #             f"layer_idx should less than layer_num"
            #         )
            #     with torch.cuda.Stream(self.io_stream):
            #         tensor=self.layers_tensor[layer_idx]
            #         self.load_weight2layer_general(layer, layer_idx, tensor)
            #     print(f"load layer {layer_idx} finish take {time.time() - time_start}")
    def decoder_qkv(
        self,
        layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions=False,
        use_cache=True
    ):
        return self.model.attn_qkv(
            layer=layer,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions
        )
    def decoder_attn_queue(self, attn_inputs, io_queue, out_queue):
        self.model.attn_compute_concurrent(
            attn_inputs=attn_inputs,
            io_queue=io_queue,
            out_queue=out_queue
        )
    def decoder_attn(
        self,
        layer,
        hidden_states,
        past_key_value,
        position_ids,
        attention_mask
    ):
        attn_weights, present_key_value = self.model.attn_compute_route(
            layer,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=True,
            io_queue=self.io_queue,
            out_queue=self.out_queue
        )
        return present_key_value
    # @torch.no_grad
    def decoder_mlp(
        self,
        mlp_layer,
        mlp_hidden_states,
        mlp_o_hidden_states,
    ):
        mlp_output=self.model.mlpc(
            layer=mlp_layer,
            hidden_states=mlp_hidden_states,
            o_hidden_states=mlp_o_hidden_states
        )
        return mlp_output
    def sync(self):
        torch.cuda.synchronize(device=self.device)
    def async_get_attn(self):
        attn_output = self.out_queue.get()
        return attn_output
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
        layer_num=self.layer_num
        
        
        layerc=self.layerc
        layern=self.layerc
        cur_layer_idx=0
        
        layer_output1 = None
        layer_output2 = None
        
        hidden_states_attn=hidden_states1
        
        past_key_value=past_key_value1
        attention_mask=attention_mask1
        position_ids=position_ids1
        time_start = time.time()
        
        attn_inputs = self.decoder_qkv(
            layer=layerc,
            hidden_states=hidden_states_attn,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        self.decoder_attn_queue(
            attn_inputs=attn_inputs,
            io_queue=self.io_queue,
            out_queue=self.out_queue
        )

        # self.sync()
        out_map = self.async_get_attn()
        print(f"first attn cost {time.time()-time_start}")
        
        attn_output = out_map["attn_output"]
        attn_inputs = out_map["attn_inputs"]
        
        self.release_attn_inputs(attn_inputs)
        
        j=1
        
        # get_gpu_memory_usage(f"after first attn")
        
        mlp_output=hidden_states2
        
        while True:
        # for i in range(20):
            load_next_layer=False
            
            time_start_decoder=time.time()
            # should first
            j = j+1
            if j%2==0:
                # layerc.layer_idx
                cur_layer_idx=cur_layer_idx+1
                # deal the layer 0-27
                if cur_layer_idx == layer_num-1:
                    break
                if cur_layer_idx < layer_num-1:
                    load_next_layer=True
                       
                layerc=layern
                layer_attn=layerc
                layer_mlp=layerc
                
                past_key_value=past_key_value2
                attention_mask=attention_mask2
                position_ids=position_ids2
            else:
                layer_attn=layern
                layer_mlp=layerc
                
                past_key_value=past_key_value1
                attention_mask=attention_mask1
                position_ids=position_ids1
    
            hidden_states_mlp =  hidden_states_attn
            hidden_states_mlp_o = attn_output
            hidden_states_attn = mlp_output    
            
            if load_next_layer:
                # print(f"need to load next layer")
                next_layer_idx=cur_layer_idx+1
                self.load_layer_concurrent(self.layern, next_layer_idx)
                layern=self.layern
            
            # _ = self.decoder_attn(
            #     layer=layer_attn,
            #     hidden_states=hidden_states_attn,
            #     past_key_value=past_key_value,
            #     attention_mask=attention_mask,
            #     position_ids=position_ids,
            # )
            attn_inputs = self.decoder_qkv(
                layer=layer_attn,
                hidden_states=hidden_states_attn,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            
            self.decoder_attn_queue(
                attn_inputs=attn_inputs,
                io_queue=self.io_queue,
                out_queue=self.out_queue
            )
              
                
            print(f"decoder mlp start {time.time()}")
            mlp_output = self.decoder_mlp(
                mlp_layer=layer_mlp,
                mlp_hidden_states=hidden_states_mlp,
                mlp_o_hidden_states=hidden_states_mlp_o,
            )
            print(f"decoder_mlp cost {time.time()}")
            
          

            time_start_asyncget = time.time()
            out_map = self.async_get_attn()
            print(f"async get cost {time.time()-time_start_asyncget} seconds")

            attn_output = out_map["attn_output"]
            attn_inputs = out_map["attn_inputs"]
            # print(f"before release {self.pool.get_usage_info()}")
            self.release_attn_inputs(attn_inputs)
            # print(f"after release {self.pool.get_usage_info()}")
            
            print(f"decoder layer {cur_layer_idx} j {j} {time.time()} cost {time.time()-time_start_decoder}\n")
            # get_gpu_memory_usage(f"layer {cur_layer_idx}")
            
            
        if cur_layer_idx != layer_num-1:
            raise ValueError(
                f"decoder process has something wrong"
            )
            
        layer_output1 = mlp_output
        
        hidden_states_mlp=hidden_states_attn
        hidden_states_mlp_o=attn_output
        with torch.cuda.Stream(self.mlp_stream):
            mlp_output = self.decoder_mlp(
                mlp_layer=layer_mlp,
                mlp_hidden_states=hidden_states_mlp,
                mlp_o_hidden_states=hidden_states_mlp_o,
            )
        layer_output2=mlp_output
        self.sync()
        return layer_output1, layer_output2
    
    
    def decoder(
        self,
        layer,
        mlp_layer,
        hidden_states,
        past_key_value,
        mlp_hidden_states,
        mlp_o_hidden_states,
    ):
        # this layer attn
        attn_output, attn_weights, present_key_value = self.model.attn_compute(
            layer,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=past_key_value,
            output_attentions=False,
            use_cache=True
        )
        
        # torch.cuda.synchronize(device="cuda:1")
        # mlp_output=self.model.mlpc(
        #     layer=mlp_layer,
        #     hidden_states=mlp_hidden_states,
        #     o_hidden_states=mlp_o_hidden_states
        # )
        
        # time_start=time.time()
        # last layer mlp
        mlp_output=self.model.mlpc(
            layer=mlp_layer,
            hidden_states=mlp_hidden_states,
            o_hidden_states=mlp_o_hidden_states
        )
        # torch.cuda.synchronize(device="cuda:1")
        # print(f"mlp cost {time.time()-time_start}")
        return attn_output, present_key_value, mlp_output
    def prepare_inputs_for_generation(
        self,
        inputs_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        pass
        
    
    def generate(
        self, 
        inputs,
        max_new_tokens: int = 32,
    ):

        pass
    
    def setup_thread(self):
        self.running = True
        thread_attn=Thread(target=self.cpu_attn, args=())
        thread_io_layer = Thread(target=self.io_layer_func, args=())
        thread_io=Thread(target=self.io_func, args=())
        
        thread_attn.start()
        thread_io_layer.start()
        thread_io.start()
        
        return thread_attn, thread_io_layer, thread_io

    def stop_thread(self):
        self.running=False
        self.stop_io()
        self.stop_io_layer()
        self.stop_attn()
        
def main():
    model_name = "deepseek_16b"
    model_path = "/mnt/zhengcf3/lpllm/models/deepseek_16b"
    class_name = "Deepseek"
    
    
    get_gpu_memory_usage()
    
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    lm = LPLLM(
        model_name=model_name, 
        model_path=model_path, 
        class_name=class_name,
        device="cuda:1"
    )
    config = lm.get_config()
    
    attn_thread, io_layer_thread, io_thread = lm.setup_thread()
    
    
    get_gpu_memory_usage("init lm")
    
    i_size=config.hidden_size
    device="cuda:1"
    batch_size=8
    seq_len = 512
    dtype = torch.bfloat16
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    input2_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    input_tensor = torch.randn(batch_size, seq_len, i_size, dtype=dtype, device=device)

    get_gpu_memory_usage()
    
    (input_embeds1, attention_mask1,
    past_key_values1, position_ids1) \
    =lm.model.forward_prepare(
        input_ids=input_ids,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        use_cache=True,
        attention_mask=None,
    )
    
    (input_embeds2, attention_mask2,
    past_key_values2, position_ids2) \
    =lm.model.forward_prepare(
        input_ids=input2_ids,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        use_cache=True,
        attention_mask=None,
    )
    
    get_gpu_memory_usage()
    

    time_start=time.time()
      
    layer_output1, layer_output2 = lm.decoders(
        hidden_states1=input_embeds1,
        hidden_states2=input_embeds2,
        past_key_value1=past_key_values1,
        past_key_value2=past_key_values2,
        attention_mask1=attention_mask1,
        attention_mask2=attention_mask2,
        position_ids1=position_ids1,
        position_ids2=position_ids2
    )
    
    
    # mlp_output=input_tensor
    # for i in range(20):
    #     mlp_output_new = lm.decoder_mlp(
    #         mlp_layer=lm.layerc,
    #         mlp_hidden_states=mlp_output,
    #         mlp_o_hidden_states=mlp_output
    #     )
    #     mlp_output=mlp_output_new
    #     get_gpu_memory_usage("mlp compute")
    
    print(f"decoders cost {time.time()-time_start} seconds")
    lm.stop_thread()
    
def main_new():
    q = Queue()
    weight=torch.randn(8,512,2048, device="cpu")
    def io_func():
        a = q.get()
        time_start=time.time()
        
        weight.to(device="cuda:1")
        
        print(f"cost {time.time()-time_start}")
        
    thread_io=Thread(target=io_func, args=())
    thread_io.start()
    
    q.put(1)
if __name__ == "__main__":
    main()
    # main_new()