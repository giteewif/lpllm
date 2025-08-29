from lpllm.base import Base
from lpllm.utils import get_gpu_memory_usage
from .configuration_deepseek import DeepseekConfig
from .modeling_deepseek import DeepseekDecoderLayer, DeepseekRMSNorm
from .modeling_deepseek import apply_rotary_pos_emb, repeat_kv
import json
import torch
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
import time
from transformers.cache_utils import Cache, DynamicCache
from torch import nn
from typing import List, Optional, Tuple, Union
from safetensors import safe_open
class Deepseek(Base, nn.Module):
    def __init__(self, path, device):
        super().__init__()

        self.path=path
        self.config=self.load_config(path)
        self._use_sdpa=True
        self.init_others(device)
        
    def get_layer(self, config: DeepseekConfig, layer_idx: int):
        layer = DeepseekDecoderLayer(config, layer_idx)
        return layer
    def get_qshape(self):
        config=self.config
        q_shape = (1, config.num_attention_heads, 1, config.head_dim)
        return q_shape
    def init_others(self, device):
        lm_head, embed_tokens, norm=self.get_others(self.config, self.path)
        
        lm_head=lm_head.to(device=device, dtype=self.config.torch_dtype)
        embed_tokens=embed_tokens.to(device=device, dtype=self.config.torch_dtype)
        norm=norm.to(device=device, dtype=self.config.torch_dtype)
        self.lm_head=lm_head
        self.embed_tokens=embed_tokens
        self.norm=norm
        
        # print(embed_tokens.weight.dtype)
    def get_others(self, config: DeepseekConfig, path):
        weight_tensor=self.load_others_tensor(path)
        embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        norm=DeepseekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        lm_head=nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        embed_tokens.weight.data.copy_(weight_tensor["embed_tokens"])
        norm.weight.data.copy_(weight_tensor["norm"])
        lm_head.weight.data.copy_(weight_tensor["lm_head"])
        
        return lm_head, embed_tokens, norm
    def load_config(self, path):
        config = DeepseekConfig.from_pretrained(path)
        return config
    def forward_prepare(
        self,
        input_ids,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        use_cache=True,
        attention_mask=None,
    ):
        batch_size, seq_length=input_ids.shape[:2]
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
            
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        
        inputs_embeds=self.embed_tokens(input_ids)
        if self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        return inputs_embeds, attention_mask, \
            past_key_values, position_ids
        
    def qkv_compute(
        self,
        layer, 
        hidden_states,
        position_ids,
        past_key_value,
        output_attentions,
    ):
        #sdpa
        if output_attentions:
            logger.warning_once(
                "DeepseekModel is using DeepseekSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return None
        bsz, q_len, _ = hidden_states.size()
        
        # start qkv this layer
        query_states = layer.self_attn.q_proj(hidden_states)
        key_states = layer.self_attn.k_proj(hidden_states)
        value_states = layer.self_attn.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, layer.self_attn.layer_idx)
        # print(kv_seq_len, query_states.shape)
        cos, sin = layer.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        return (query_states, key_states, 
                value_states, sin, cos)
    def move_qkv(
        self,
        query_states,
        key_states,
        value_states, 
        sin, cos        
    ):
        if query_states.device == "cpu":
            # error
            raise ValueError(
                f"qkv device should not be cpu here"
            )
        # move to cpu
        
        q = query_states.to(device="cpu")
        k = key_states.to(device="cpu")
        v = value_states.to(device="cpu")
        s = sin.to(device="cpu")
        c = cos.to(device="cpu")
        
        return (q,k,v,s,c)
    def move_qkv_concurrent(
        self,
        io_queue,
        attn_inputs,       
    ):
        query_states = attn_inputs["query_states"]
        if query_states.device == "cpu":
            # error
            raise ValueError(
                f"qkv device should not be cpu here"
            )
        # move to cpu
        io_type=0
        io_map = {
            "type": io_type,
            "attn_inputs":  attn_inputs,
        }
        io_queue.put(io_map)
        return 
    def move_attn_concurrent(self, io_queue, attn_output, d_device):
        io_type = 1

        io_map = {
            "type": io_type,
            "attn_output": attn_output,
            "d_device": d_device,
        }
        io_queue.put(io_map)
        return
    
    def move_attn(self, attn_output, d_device):
        cpu_device=torch.device(type="cpu")
        if attn_output.device != cpu_device:
            #error here
            raise ValueError(
                f"attn output device should be cpu, but is {attn_output.device}"
            )
        attn_output = attn_output.to(device=d_device)
        return attn_output
    def move_and_attn(
        self,
        io_queue,
        out_queue,
        attn_inputs 
    ):
        self.move_qkv_concurrent(
            io_queue,
            attn_inputs
        )
        # wait for the next attn_output
        # attn_output = out_queue.get()
        # return attn_output
    def attn(
        self,
        bsz, q_len,
        layer,
        query_states,
        key_states,
        value_states,
        sin, cos, attention_mask,
        past_key_value
    ):
        num_key_value_groups = layer.self_attn.num_key_value_groups
        attention_dropout = layer.self_attn.attention_dropout
        # should be zero in infer
        is_causal = layer.self_attn.is_causal
        hidden_size=layer.self_attn.hidden_size
        attention_dropout = 0
        
        # print(f"layer_idx {layer.layer_idx} before kv", key_states.shape, value_states.shape)
        time_past_key_value=time.time()
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, layer.layer_idx, cache_kwargs)
        # print(f"kv after device", key_states.shape, value_states.shape)
        print(f"past_key_value cost {time.time()-time_past_key_value}")
        
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)
        
        time_start = time.time()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if layer.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=is_causal and attention_mask is None and q_len > 1,
        )
        
        # print(f"shape of query_states {query_states.shape} \
        #         shape of key_states {key_states.shape} \
        #         shape of value_states {value_states.shape} \
        #         shape of attn_output {attn_output.shape} \
        #         scaled_dot_product_attention cost {time.time()-time_start}")
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        
        return attn_output
    
    def attn_compute_route(
        self,
        layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        io_queue,
        out_queue,
    ):
        return self.attn_compute_concurrent(
            layer,
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            io_queue,
            out_queue,
        )
    def attn_qkv(
        self,
        layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
    ):
        # layer_norm here
        hidden_states = layer.input_layernorm(hidden_states)
        # print(f"layernorm cost {time.time()-time_start}")
        if output_attentions:
            raise ValueError(
                "DeepseekModel is using DeepseekSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        bsz, q_len, _ = hidden_states.size()
        
        (q,k,v,s,c) = self.qkv_compute(
            layer=layer,
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions
        )
        # print(f"qkv compute cost {time.time()-time_start}")
        attn_inputs={
            "bsz": bsz,
            "q_len": q_len,
            "layer": layer,
            "query_states": q,
            "key_states": k,
            "value_states": v,
            "sin": s, "cos": c,
            "attention_mask": attention_mask,
            "past_key_value": past_key_value
        }
        return attn_inputs
    def attn_compute_concurrent(
        self,
        attn_inputs,
        io_queue,
        out_queue,
    ):
        past_key_value = attn_inputs["past_key_value"]
        self.move_and_attn(
            io_queue,
            out_queue,
            attn_inputs=attn_inputs
        )
        # print(f"move_and_attn cost {time.time()-time_start}")
        return None, past_key_value
        
    def attn_compute(
        self,
        layer,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
    ):
        
        # layer_norm here
        hidden_states = layer.input_layernorm(hidden_states)
          
        if output_attentions:
            raise ValueError(
                "DeepseekModel is using DeepseekSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        bsz, q_len, _ = hidden_states.size()
        layer_device=layer.self_attn.q_proj.weight.device
        # should compute on gpu
        (q,k,v,s,c) = self.qkv_compute(
            layer=layer,
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions
        )
        # move to cpu
        (q,k,v,s,c) = self.move_qkv(
            query_states=q,
            key_states=k,
            value_states=v,
            sin=s,cos=c,
        )
        # compute on cpu
        attn_output=self.attn(
            bsz=bsz, q_len=q_len,
            layer=layer,
            query_states=q,
            key_states=k,
            value_states=v,
            sin=s,cos=c,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        attn_output=self.move_attn(
            attn_output=attn_output,
            d_device=layer_device
        )
        return attn_output, None, past_key_value
    def o_compute(
        self,
        layer,
        attn_output,
    ):
        attn_output = layer.self_attn.o_proj(attn_output)
        return attn_output
    def mlp_compute(
        self,
        layer,
        hidden_states,
    ):
        return layer.mlp(hidden_states)
    # o_proj, post_attn, mlp
    
    @torch.no_grad
    def mlpc(
        self,
        layer,
        hidden_states,
        o_hidden_states,
    ):
        # print(f"start call mlpc {time.time()}")
        residual=hidden_states
        # print(f"start o_compute {time.time()}")
        hidden_states=self.o_compute(layer, o_hidden_states)
        
        hidden_states=residual+hidden_states
        
        # print(f"before post_attention_layernorm {time.time()}")
        residual=hidden_states
        # hidden_states=layer.post_attention_layernorm(hidden_states)
        
        
        # get_gpu_memory_usage("mlpc before mlp")
        # print(f"start mlp {time.time()}")
        hidden_states=layer.mlp(hidden_states)
        # print(f"end mlp {time.time()}")
        # get_gpu_memory_usage("mlpc after mlp")/
        
        hidden_states=residual+hidden_states
        
        return hidden_states
        
    
    def forward2layer(
            self,
            attn_layer: DeepseekDecoderLayer, 
            mlp_layer: DeepseekDecoderLayer,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        pass
        
    def load_layers_tensor(self, path):
        config=self.config
        layers_tensor=[]
        for layer_idx in range(config.num_hidden_layers):
            # do not deal the first layer
            if layer_idx==0:
                continue
            layer_tensor=self.load_layer_tensor_single_file(layer_idx, path)
            layers_tensor.append(layer_tensor)
        return layers_tensor
    # realize abstract method
    def load_layer_tensor_single_file(self, layer_idx, path):
        config = DeepseekConfig.from_pretrained(path)
        
        index_path = path + "/model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)
            weight_map = index["weight_map"]


        weight_tensor_layer = {}
        weight_tensor_layer['experts'] = [{} for _ in range(config.n_routed_experts)]
        weight_tensor_layer['shared_experts'] = {}
        weight_tensor_layer['input_layernorm'] = {}
        weight_tensor_layer['post_attention_layernorm'] = {}
        weight_tensor_layer['attention'] = {}
        weight_tensor_layer['gate'] = {}

        weight_file = {}

        # load gate weight
        # get the weight file name from the weight map, like model-00001-of-00007.safetensors
        gate_name = f"model.layers.{layer_idx}.mlp.gate.weight"

        weight_file_name = weight_map[gate_name]
        weight_file = safe_open(f"{path}/{weight_file_name}", framework="pt", device="cpu")


        weight_tensor_layer['gate'] = weight_file.get_tensor(gate_name)

        # print(f"load weight {weight_tensor_layer['gate'].dtype}, original dtype  {weight_file.get_tensor(gate_name).dtype}")

        # load expert weight
        experts_num = config.n_routed_experts
        for expert_idx in range(experts_num):

            expert_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"

            up_name = f"{expert_name}.up_proj.weight"
            gate_name = f"{expert_name}.gate_proj.weight"
            down_name = f"{expert_name}.down_proj.weight"
            
            # with the same safe file
            # weight_file_name = weight_map[up_name]        
            # weight_file = safe_open(f"{path}/{weight_file_name}", framework="pt", device="cpu")

            weight_tensor_layer['experts'][expert_idx]["up_proj"] = weight_file.get_tensor(up_name)
            weight_tensor_layer['experts'][expert_idx]["gate_proj"] = weight_file.get_tensor(gate_name)
            weight_tensor_layer['experts'][expert_idx]["down_proj"] = weight_file.get_tensor(down_name)
            
        # load shared_expert_weight
        shared_expert_name = f"model.layers.{layer_idx}.mlp.shared_experts"

        shared_expert_gate_name = f"{shared_expert_name}.gate_proj.weight"
        shared_expert_up_name = f"{shared_expert_name}.up_proj.weight"
        shared_expert_down_name = f"{shared_expert_name}.down_proj.weight"

        # weight_file_name = weight_map[shared_expert_gate_name]        
        # weight_file = safe_open(f"{path}/{weight_file_name}", framework="pt", device="cpu")

        weight_tensor_layer["shared_experts"]['gate_proj'] = weight_file.get_tensor(shared_expert_gate_name)
        weight_tensor_layer["shared_experts"]['up_proj'] = weight_file.get_tensor(shared_expert_up_name)
        weight_tensor_layer["shared_experts"]['down_proj'] = weight_file.get_tensor(shared_expert_down_name)

        # load layernorm weight and post attention layernorm weight
        input_layernorm_name = f"model.layers.{layer_idx}.input_layernorm.weight"
        post_attention_layernorm_name = f"model.layers.{layer_idx}.post_attention_layernorm.weight"

        # weight_file_name = weight_map[input_layernorm_name]        
        # weight_file = safe_open(f"{path}/{weight_file_name}", framework="pt", device="cpu")

        weight_tensor_layer["input_layernorm"] = weight_file.get_tensor(input_layernorm_name)
        weight_tensor_layer["post_attention_layernorm"] = weight_file.get_tensor(post_attention_layernorm_name)

        # load attention weight
        attention_name = f"model.layers.{layer_idx}.self_attn"
        attention_q_name = f"{attention_name}.q_proj.weight"
        attention_k_name = f"{attention_name}.k_proj.weight"
        attention_v_name = f"{attention_name}.v_proj.weight"
        attention_o_name = f"{attention_name}.o_proj.weight"

        # weight_file_name = weight_map[attention_q_name]        
        # weight_file = safe_open(f"{path}/{weight_file_name}", framework="pt", device="cpu")

        weight_tensor_layer["attention"] = {}

        weight_tensor_layer["attention"]["q_proj"] = weight_file.get_tensor(attention_q_name)
        weight_tensor_layer["attention"]["k_proj"] = weight_file.get_tensor(attention_k_name)
        weight_tensor_layer["attention"]["v_proj"] = weight_file.get_tensor(attention_v_name)
        weight_tensor_layer["attention"]["o_proj"] = weight_file.get_tensor(attention_o_name)

        return weight_tensor_layer
    def load_weight2layer_general(self, layer, layer_idx, layer_state_dict):        
        layer.load_state_dict(layer_state_dict, strict=True)
        layer.layer_idx = layer_idx
        
    def load_layers_tensor_general(self, path):
        config = DeepseekConfig.from_pretrained(path)
        
        index_path = path + "/model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)
            weight_map = index["weight_map"]
        
        layers_state_dict = {}
        
        # 提前打开weight_map value 指向的所有safetensors 文件
        weight_files = {}
        for value in set(weight_map.values()):
            weight_files[value] = safe_open(f"{path}/{value}", framework="pt", device="cpu")

        for key, value in weight_map.items():
            # 提取层索引
            if key.startswith("model.layers."):
                ksplits = key.split(".")
                layer_index = int(ksplits[2])
                if layer_index not in layers_state_dict:
                    layers_state_dict[layer_index] = {}

                weight_file = weight_files[value]
                tensor = weight_file.get_tensor(key)
                layer_key = ".".join(ksplits[3:])
                layers_state_dict[layer_index][layer_key] = tensor
        return layers_state_dict

    def load_layer_tensor(self, layer_idx, path):
        config = DeepseekConfig.from_pretrained(path)
        
        index_path = path + "/model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)
            weight_map = index["weight_map"]


        weight_tensor_layer = {}
        weight_tensor_layer['experts'] = [{} for _ in range(config.n_routed_experts)]
        weight_tensor_layer['shared_experts'] = {}
        weight_tensor_layer['input_layernorm'] = {}
        weight_tensor_layer['post_attention_layernorm'] = {}
        weight_tensor_layer['attention'] = {}
        weight_tensor_layer['gate'] = {}

        weight_file = {}

        # load gate weight
        # get the weight file name from the weight map, like model-00001-of-00007.safetensors
        gate_name = f"model.layers.{layer_idx}.mlp.gate.weight"

        weight_file_name = weight_map[gate_name]
        weight_file = safe_open(f"{path}/{weight_file_name}", framework="pt", device="cpu")


        weight_tensor_layer['gate'] = weight_file.get_tensor(gate_name)

        print(f"load weight {weight_tensor_layer['gate'].dtype}, original dtype  {weight_file.get_tensor(gate_name).dtype}")

        # load expert weight
        experts_num = config.n_routed_experts
        for expert_idx in range(experts_num):

            expert_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"

            up_name = f"{expert_name}.up_proj.weight"
            gate_name = f"{expert_name}.gate_proj.weight"
            down_name = f"{expert_name}.down_proj.weight"
            
            weight_file_name = weight_map[up_name]        
            weight_file = safe_open(f"{path}/{weight_file_name}", framework="pt", device="cpu")

            weight_tensor_layer['experts'][expert_idx]["up_proj"] = weight_file.get_tensor(up_name)
            weight_tensor_layer['experts'][expert_idx]["gate_proj"] = weight_file.get_tensor(gate_name)
            weight_tensor_layer['experts'][expert_idx]["down_proj"] = weight_file.get_tensor(down_name)
            
        # load shared_expert_weight
        shared_expert_name = f"model.layers.{layer_idx}.mlp.shared_experts"

        shared_expert_gate_name = f"{shared_expert_name}.gate_proj.weight"
        shared_expert_up_name = f"{shared_expert_name}.up_proj.weight"
        shared_expert_down_name = f"{shared_expert_name}.down_proj.weight"

        weight_file_name = weight_map[shared_expert_gate_name]        
        weight_file = safe_open(f"{path}/{weight_file_name}", framework="pt", device="cpu")

        weight_tensor_layer["shared_experts"]['gate_proj'] = weight_file.get_tensor(shared_expert_gate_name)
        weight_tensor_layer["shared_experts"]['up_proj'] = weight_file.get_tensor(shared_expert_up_name)
        weight_tensor_layer["shared_experts"]['down_proj'] = weight_file.get_tensor(shared_expert_down_name)

        # load layernorm weight and post attention layernorm weight
        input_layernorm_name = f"model.layers.{layer_idx}.input_layernorm.weight"
        post_attention_layernorm_name = f"model.layers.{layer_idx}.post_attention_layernorm.weight"

        weight_file_name = weight_map[input_layernorm_name]        
        weight_file = safe_open(f"{path}/{weight_file_name}", framework="pt", device="cpu")

        weight_tensor_layer["input_layernorm"] = weight_file.get_tensor(input_layernorm_name)
        weight_tensor_layer["post_attention_layernorm"] = weight_file.get_tensor(post_attention_layernorm_name)

        # load attention weight
        attention_name = f"model.layers.{layer_idx}.self_attn"
        attention_q_name = f"{attention_name}.q_proj.weight"
        attention_k_name = f"{attention_name}.k_proj.weight"
        attention_v_name = f"{attention_name}.v_proj.weight"
        attention_o_name = f"{attention_name}.o_proj.weight"

        weight_file_name = weight_map[attention_q_name]        
        weight_file = safe_open(f"{path}/{weight_file_name}", framework="pt", device="cpu")

        weight_tensor_layer["attention"] = {}

        weight_tensor_layer["attention"]["q_proj"] = weight_file.get_tensor(attention_q_name)
        weight_tensor_layer["attention"]["k_proj"] = weight_file.get_tensor(attention_k_name)
        weight_tensor_layer["attention"]["v_proj"] = weight_file.get_tensor(attention_v_name)
        weight_tensor_layer["attention"]["o_proj"] = weight_file.get_tensor(attention_o_name)

        return weight_tensor_layer
        
    def load_others_tensor(self, path):
        config = DeepseekConfig.from_pretrained(path)
        
        index_path = path + "/model.safetensors.index.json"
        with open(index_path, "r") as f:
            index = json.load(f)
            weight_map = index["weight_map"]
            
        weight_tensor = {}
        
        lm_head_name="lm_head.weight"
        embed_tokens_name="model.embed_tokens.weight"
        norm_name="model.norm.weight"
        
        weight_file_name=weight_map[lm_head_name]
        weight_file = safe_open(f"{path}/{weight_file_name}", framework="pt", device="cpu")
        weight_tensor["lm_head"]=weight_file.get_tensor(lm_head_name)
        weight_tensor["embed_tokens"]=weight_file.get_tensor(embed_tokens_name)
        weight_tensor["norm"]=weight_file.get_tensor(norm_name)
        return weight_tensor

    def load_weight2layer_concurrent(self, layer, layer_idx, weight_tensor_layer, non_blocking=True):
        experts_num = len(weight_tensor_layer['experts'])
        
        load_array = []
        
        l = layer.mlp.gate
        t = weight_tensor_layer
        load_array.append((l,t))
        
        for expert_idx in range(experts_num):
            l = layer.mlp.experts[expert_idx].up_proj
            t = weight_tensor_layer['experts'][expert_idx]["up_proj"]
            load_array.append((l,t))
            
            
            
            layer.mlp.experts[expert_idx].up_proj.weight.data.copy_(weight_tensor_layer['experts'][expert_idx]["up_proj"], non_blocking=non_blocking)
            layer.mlp.experts[expert_idx].gate_proj.weight.data.copy_(weight_tensor_layer['experts'][expert_idx]["gate_proj"], non_blocking=non_blocking)
            layer.mlp.experts[expert_idx].down_proj.weight.data.copy_(weight_tensor_layer['experts'][expert_idx]["down_proj"], non_blocking=non_blocking)
        
        pass
    # realize abstract method   
    def load_weight2layer(self, layer, layer_idx, weight_tensor_layer, non_blocking=True):
        experts_num = len(weight_tensor_layer['experts'])
        
        # load gate weight
        layer.mlp.gate.weight.data.copy_(weight_tensor_layer['gate'])
        # print(f"gate weight dtypse: {layer.mlp.gate.weight.dtype}")
        # load expert weight
        for expert_idx in range(experts_num):
            layer.mlp.experts[expert_idx].up_proj.weight.data.copy_(weight_tensor_layer['experts'][expert_idx]["up_proj"], non_blocking=non_blocking)
            layer.mlp.experts[expert_idx].gate_proj.weight.data.copy_(weight_tensor_layer['experts'][expert_idx]["gate_proj"], non_blocking=non_blocking)
            layer.mlp.experts[expert_idx].down_proj.weight.data.copy_(weight_tensor_layer['experts'][expert_idx]["down_proj"], non_blocking=non_blocking)

        # load shared expert weight
        layer.mlp.shared_experts.gate_proj.weight.data.copy_(weight_tensor_layer["shared_experts"]['gate_proj'], non_blocking=non_blocking)
        layer.mlp.shared_experts.up_proj.weight.data.copy_(weight_tensor_layer["shared_experts"]['up_proj'], non_blocking=non_blocking)
        layer.mlp.shared_experts.down_proj.weight.data.copy_(weight_tensor_layer["shared_experts"]['down_proj'], non_blocking=non_blocking)

        # load layernorm weight and post attention layernorm weight
        layer.input_layernorm.weight.data.copy_(weight_tensor_layer["input_layernorm"], non_blocking=non_blocking)
        layer.post_attention_layernorm.weight.data.copy_(weight_tensor_layer["post_attention_layernorm"], non_blocking=non_blocking)


        # load attention weight
        layer.self_attn.q_proj.weight.data.copy_(weight_tensor_layer["attention"]["q_proj"], non_blocking=non_blocking)
        layer.self_attn.k_proj.weight.data.copy_(weight_tensor_layer["attention"]["k_proj"], non_blocking=non_blocking)
        layer.self_attn.v_proj.weight.data.copy_(weight_tensor_layer["attention"]["v_proj"], non_blocking=non_blocking)
        layer.self_attn.o_proj.weight.data.copy_(weight_tensor_layer["attention"]["o_proj"], non_blocking=non_blocking)

        layer.layer_idx = layer_idx
        
class DeepseekDecoderLayerWrapper:
    def __init__(self):
        super().__init__()
        
    def get_layer(config: DeepseekConfig, layer_idx: int):
        layer = DeepseekDecoderLayer(config, layer_idx)
        return layer