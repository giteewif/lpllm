from logging import exception
from lpllm.lpmodel import LPModuleWrapper
import time
import torch
from .modeling_deepseek import DeepseekDecoderLayer
from .configuration_deepseek import DeepseekConfig
from .modeling_deepseek import apply_rotary_pos_emb, repeat_kv
from transformers.cache_utils import Cache, DynamicCache
from lpllm.logger import init_logger
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from .modeling_deepseek import DeepseekModel, DeepseekForCausalLM
from accelerate import init_empty_weights

logger = init_logger(__name__)

class LPModule(LPModuleWrapper):
    def get_model(config):
        config._attn_implementation = "sdpa"
        with init_empty_weights():
            model = DeepseekForCausalLM(config)
            return model
    def get_layer(model):
        layer_dense = model.model.layers[0]
        layer_moe = model.model.layers[1]
        # layer_dense = DeepseekDecoderLayer(self.config, 0)
        # layer_moe = DeepseekDecoderLayer(self.config, 1)
        return layer_dense, layer_moe
    def get_model_layer(model, layer_idx):
        return model.model.layers[layer_idx]
    def get_embed_tokens_norm_lm_head(model):
        model = model
        return model.model.embed_tokens, model.model.norm, model.lm_head
    def forward_prepare(
        model,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=True,
        attention_mask=None,
    ):
        # _use_sdpa = config._attn_implementation == "sdpa"

        embed_tokens, _, _ = LPModule.get_embed_tokens_norm_lm_head(model)
        _use_sdpa = True

        if input_ids is not None:
            batch_size, seq_length=input_ids.shape[:2]
        else:
            batch_size, seq_length=inputs_embeds.shape[:2]
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                raise ValueError(f"should get the cache")
            # use_legacy_cache = not isinstance(past_key_values, Cache)
            # if use_legacy_cache:
            #     past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_seq_length(0)
            
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        # logger.debug(f"position_ids {position_ids}")
        
        if input_ids != None:
            inputs_embeds=embed_tokens(input_ids)

        if _use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        return inputs_embeds, attention_mask, \
            past_key_values, position_ids
    # layer_info, layer_index_loc, attn_info, mlp_info
    def get_layer_attr_info() -> tuple[str, int, str, str]:
        # "model.layers.1.mlp.experts.0.down_proj.weight"
        return "model.layers.", int(2), ".self_attn.", ".mlp."
    def check_layer_is_dense(config, layer_idx):
        if (config.n_routed_experts is not None and  \
                layer_idx >= config.first_k_dense_replace and layer_idx % config.moe_layer_freq == 0):
            return False
        else:
            return True
    @torch.no_grad()
    def mlp_prepare(layer, mlp_hidden_states, mlp_o_hidden_states):

        bsz, q_len, hidden_size = mlp_hidden_states.shape
        # transpose and reshape here, move to mlp before
        mlp_o_hidden_states = mlp_o_hidden_states.transpose(1, 2).contiguous()
        mlp_o_hidden_states = mlp_o_hidden_states.reshape(
            bsz, #bsz
            q_len, #q_len
            hidden_size
        )

        residual=mlp_hidden_states
        hidden_states=layer.self_attn.o_proj(mlp_o_hidden_states)
        

        hidden_states=residual+hidden_states
        
        residual=hidden_states
        #post attention_layer_norm
        hidden_states=layer.post_attention_layernorm(hidden_states)
        return layer.mlp.moe_infer_prepare(hidden_states), residual
    @torch.no_grad()
    def decoder_mlp_post(layer, hidden_states, tokens_per_expert, topk_weight, token_idxs, idxs, identity, residual):
        hidden_states = layer.mlp.moe_infer_post(hidden_states, tokens_per_expert, topk_weight, token_idxs, idxs, identity)
        hidden_states=residual+hidden_states
        return hidden_states

    @torch.no_grad()
    def decoder_mlp(layer, mlp_hidden_states, mlp_o_hidden_states, attn_func):

        bsz, q_len, hidden_size = mlp_hidden_states.shape
        # transpose and reshape here, move to mlp before
        mlp_o_hidden_states = mlp_o_hidden_states.transpose(1, 2).contiguous()
        mlp_o_hidden_states = mlp_o_hidden_states.reshape(
            bsz, #bsz
            q_len, #q_len
            hidden_size
        )


        residual=mlp_hidden_states
        hidden_states=layer.self_attn.o_proj(mlp_o_hidden_states)
        

        hidden_states=residual+hidden_states
        
        residual=hidden_states

        #post attention_layer_norm
        hidden_states=layer.post_attention_layernorm(hidden_states)
    
        hidden_states=layer.mlp(hidden_states, attn_queue_func=attn_func)

    
        hidden_states=residual+hidden_states
        
        return hidden_states
    def decoder_attn(
        layer,
        bsz, q_len,
        layer_idx,
        query_states,
        key_states,
        value_states,
        sin, cos, 
        past_key_value,
        position_ids,
        attention_mask,
    ):
        num_key_value_groups = layer.self_attn.num_key_value_groups
        # should be zero in infer
        is_causal = layer.self_attn.is_causal
        hidden_size=layer.self_attn.hidden_size
        
        # logger.debug(f"attn layer_idx: {layer.layer_idx}")
        time_past_key_value=time.time()
        try:
            if past_key_value is not None:
                # 计算当前缓存位置：已有序列长度
                q_len = key_states.shape[-2]
                
                seq_length = past_key_value.get_seq_length(layer_idx)
                if q_len == 1:
                    cache_position = torch.arange(1, device="cpu")
                    cache_position[0] = seq_length
                else:
                    cache_position = torch.arange(q_len, device="cpu")
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)
                logger.debug(f"key_states device {key_states.device}")
            logger.debug(f"update past key value cost {time.time()-time_past_key_value:.6f} seconds")
        except Exception as e:
            logger.error(f"update past key value error {e}")
            raise e
        
        time_start = time.time()
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)
        logger.debug(f"q shape {query_states.shape}, k shape {key_states.shape}, v shape {value_states.shape}")
        time_start = time.time()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=is_causal and attention_mask is None and q_len > 1,
        )
        logger.debug(f"dot attn cost {time.time()-time_start:.6f} seconds")
        # 相比cpu执行，gpu执行会更快
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        return attn_output, past_key_value
    def decoder_qkv(layer, hidden_states, past_key_value, position_ids, attention_mask):
         #sdpa
        bsz, q_len, _ = hidden_states.size()
        
        logger.debug(f"hidden_states.shape {hidden_states.shape} hidden_states.dtype {hidden_states.dtype}")
        # input_layernorm
        hidden_states=layer.input_layernorm(hidden_states)
        # start qkv this layer
        query_states = layer.self_attn.q_proj(hidden_states)
        key_states = layer.self_attn.k_proj(hidden_states)
        value_states = layer.self_attn.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)



        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(layer.self_attn.layer_idx)
        cos, sin = layer.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        return (query_states, key_states, 
                value_states, sin, cos)