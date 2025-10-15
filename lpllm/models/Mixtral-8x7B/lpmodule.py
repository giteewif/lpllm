from logging import exception

from lpllm.lpmodel import LPModuleWrapper
import time
import torch

from lpllm.logger import init_logger
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, apply_rotary_pos_emb, repeat_kv
from accelerate import init_empty_weights
import torch.nn.functional as F
from lpllm.pinpool import PinnedMemoryPool
from lpllm.tutils import scaled_dot_product_attention_help

logger = init_logger(__name__)

class LPModule(LPModuleWrapper):
    def get_model(config):
        config._attn_implementation = "sdpa"
        with init_empty_weights():
            model = MixtralForCausalLM(config)
            return model
    def get_layer(model):
        layer_dense = model.model.layers[0]
        layer_moe = model.model.layers[1]
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
        _use_sdpa = True
        embed_tokens, _, _ = LPModule.get_embed_tokens_norm_lm_head(model)
        if input_ids is not None:
            batch_size, seq_length=input_ids.shape[:2]
        else:
            batch_size, seq_length=inputs_embeds.shape[:2]
        past_key_values_length = 0
        if use_cache:
            if past_key_values is None:
                raise ValueError(f"should get the cache")
            past_key_values_length = past_key_values.get_seq_length(0) if past_key_values is not None else 0

        if position_ids is None:
            # indices on cpu
            position_ids_device = "cpu"
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=position_ids_device
            )
            position_ids = position_ids.unsqueeze(0)
        
        if input_ids != None:
            inputs_embeds=embed_tokens(input_ids)
        else:
            raise ValueError(f"input_ids should not be None")

        cache_position = None
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=position_ids.device
        )

        logger.debug(f"before sdpa attention attention_mask {attention_mask}")
        if _use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        logger.debug(f"after sdpa attention attention_mask {attention_mask}")

        return inputs_embeds, attention_mask, \
            past_key_values, position_ids
    # layer flag, layer_index, attn flag, mlp flag
    def get_layer_attr_info() -> tuple[str, int, str, str]:
        return "model.layers.", int(2), ".self_attn.", ".block_sparse_moe."
    def check_layer_is_dense(config, layer_idx):
        # with no dense
        return False
    
    # helper
    # from mixtral_4_47_1 modeling_mixtral.py MixtralSparseMoeBlock.forward
    def moe_infer_prepare(layer, hidden_states):
        # 存在中间值 identity
        identity = hidden_states

        _, _, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = layer.block_sparse_moe.gate(hidden_states)


        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, layer.block_sparse_moe.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)


        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=layer.block_sparse_moe.num_experts)
        # 计算激活的expert数量
        # expert_mask的shape为 [batch*seq, top_k, num_experts]，其中每个token的top_k个被选中的expert位置为1，其余为0
        # 对batch*seq和top_k两个维度求和，得到每个expert被选中的总次数
        # 然后判断每个expert是否至少被选中过一次（>0），最后统计被激活的expert数量
        # 统计每个expert被激活（被分配到token）的token数
        tokens_per_expert = expert_mask.sum(dim=(0, 1)).tolist()
        num_active_experts = (expert_mask.sum(dim=(0, 1)) > 0).sum().item()
        logger.debug(f"Tokens per expert: {tokens_per_expert}")
        logger.debug(f"Number of active experts: {num_active_experts}")
        expert_mask = expert_mask.permute(2, 1, 0)


        return hidden_states, expert_mask, routing_weights, identity
    
    # helper
    # from mixtral_4_47_1 modeling_mixtral.py MixtralSparseMoeBlock.forward
    def moe_infer_post(layer, hidden_states, expert_mask, routing_weights, identity):
        batch_size, sequence_length, hidden_dim = identity.shape
        # expert_out_list = []

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(layer.block_sparse_moe.num_experts):
            expert_layer = layer.block_sparse_moe.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            # expert_out_list.append((top_x, current_hidden_states))
        
        # for (exp_token_idx, expert_out) in expert_out_list:
        #     final_hidden_states.index_add_(0, exp_token_idx, expert_out.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states


    @torch.no_grad()
    def mlp_prepare(layer, mlp_hidden_states, mlp_o_hidden_states):

        logger.debug(f"call mlp prepare")
        logger.debug(f"layer idx {layer.self_attn.layer_idx}")
        # transpose and reshape here, move to mlp before
        bsz, q_len = mlp_hidden_states.shape[:2]
        mlp_o_hidden_states = mlp_o_hidden_states.transpose(1, 2).contiguous()
        mlp_o_hidden_states = mlp_o_hidden_states.reshape(
            bsz, #bsz
            q_len, #q_len
            -1
        )

        residual=mlp_hidden_states

        hidden_states=layer.self_attn.o_proj(mlp_o_hidden_states)
        

        hidden_states=residual+hidden_states
        
        residual=hidden_states
        #post attention_layer_norm
        hidden_states=layer.post_attention_layernorm(hidden_states)

        hidden_states, expert_mask, routing_weights, identity = LPModule.moe_infer_prepare(layer, hidden_states)

        # return return hidden_states, tokens_per_expert, topk_weight, token_idxs, idxs, identity, residual
        return (hidden_states, expert_mask, routing_weights, None, None, identity), residual

    @torch.no_grad()
    def decoder_mlp_post(
        layer, hidden_states, tokens_per_expert, 
        topk_weight, token_idxs, idxs, 
        identity, residual
    ):
        logger.debug(f"call mlp post")
        # not use token_idxs, idxs here
        expert_mask = tokens_per_expert
        routing_weights = topk_weight

        hidden_states = LPModule.moe_infer_post(
            layer, hidden_states, 
            expert_mask=expert_mask, 
            routing_weights=routing_weights, 
            identity=identity
        )
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

        logger.debug(f"layer idx {layer.self_attn.layer_idx}")
        #post attention_layer_norm
        hidden_states=layer.post_attention_layernorm(hidden_states)
    
        hidden_states, router_logits =layer.block_sparse_moe(hidden_states)

    
        hidden_states=residual+hidden_states
        return hidden_states
    #more fast in cpu
    def repeat_kv_local(hidden_states: torch.Tensor, n_rep: int, pool: PinnedMemoryPool) -> torch.Tensor:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        empty_cache = torch.empty((batch, num_key_value_heads, n_rep, slen, head_dim))
        hidden_states_cache = pool.alloc_same_pin_tensor(empty_cache)
        for i in range(n_rep):
            hidden_states_cache[:, :, i, :, :] = hidden_states
        return hidden_states_cache.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
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
        pool_memory: PinnedMemoryPool
    ):
        num_key_value_groups = layer.self_attn.num_key_value_groups
        
        logger.debug(f"layer_idx {layer_idx} attn layer_idx: {layer.self_attn.layer_idx}")
        time_past_key_value=time.time()
        try:
            if past_key_value is not None:
                # 计算当前缓存位置：已有序列长度
                q_len = key_states.shape[-2]
                
                seq_length = past_key_value.get_seq_length(layer_idx)

                kv_seq_len = q_len + seq_length

                if q_len == 1:
                    cache_position = torch.arange(1, device="cpu")
                    cache_position[0] = seq_length
                else:
                    cache_position = torch.arange(q_len, device="cpu")
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
                
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)

                # if use static cache here, get real value shape
                key_states = key_states[:,:,:kv_seq_len,:]
                value_states = value_states[:,:,:kv_seq_len,:]

                logger.debug(f"key_states device {key_states.device}")
            logger.debug(f"update past key value cost {time.time()-time_past_key_value:.6f} seconds")
        except Exception as e:
            logger.error(f"update past key value error {e}")
            raise e
        # move kv to gpu
        # 生成 (bsz, kv_heads, q_len, num, head_dim )的空torch

        time_start = time.time()
        key_states = repeat_kv(key_states, num_key_value_groups)
        value_states = repeat_kv(value_states, num_key_value_groups)
        # key_states = LPModule.repeat_kv_local(key_states, num_key_value_groups, pool_memory)
        # value_states = LPModule.repeat_kv_local(value_states, num_key_value_groups, pool_memory)
        logger.debug(f"q shape {query_states.shape}, k shape {key_states.shape}, v shape {value_states.shape}")

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        is_causal = True if causal_mask is None and q_len > 1 else False

        time_start = time.time()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=is_causal,
        )
        logger.debug(f"dot attn cost {time.time()-time_start:.6f} seconds")
        # 相比cpu执行，gpu执行会更快
        # attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.reshape(bsz, q_len, hidden_size)
        
        # free kv cache after execute attn
        # attn finish release kv
        # pool_memory.free(key_states)
        # pool_memory.free(value_states)
        return attn_output, past_key_value

    def decoder_attn_batch(
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
        pool_memory
    ):
        torch.cuda.nvtx.range_push("update key_value")
        num_key_value_groups = layer.self_attn.num_key_value_groups
        
        logger.debug(f"layer_idx {layer_idx} attn layer_idx: {layer.self_attn.layer_idx}")
        time_past_key_value=time.time()
        try:
            if past_key_value is not None:
                # 计算当前缓存位置：已有序列长度

                q_len = key_states.shape[-2]
                key_bsz = key_states.shape[0]

                assert key_bsz == bsz, f"key_bsz {key_bsz} does not match bsz {bsz}"
                # batch here, do not need seq_length
                seq_length = past_key_value.get_seq_length(layer_idx)
                logger.debug(f"get seq_length {seq_length} in decoder_attn_batch")

                if q_len == 1:
                    logger.warning(f"generally decoder_attn_batch called with q_len == 1")
                    cache_position = torch.arange(1, device="cpu")
                    cache_position[0] = seq_length
                else:
                    cache_position = torch.arange(q_len, device="cpu")
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models

                start_batch = past_key_value.update_batch_loc(layer_idx)
                end_batch = start_batch + key_bsz
                logger.debug(f"decoder_attn_batch update batch_dim {start_batch}-{end_batch}")
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, cache_kwargs, mode="update_seq")

                kv_seq_len = q_len + seq_length
                logger.debug(f"update for kv cache {start_batch}-{end_batch} kv_seq_len {kv_seq_len} in decoder_attn_batch")
                # with kv_seq_len, contain the past key value
                key_states = key_states[start_batch:end_batch, :, :kv_seq_len, :]
                value_states = value_states[start_batch:end_batch, :, :kv_seq_len, :]
                
                logger.debug(f"key_states device {key_states.device}")
            logger.debug(f"update past key value cost {time.time()-time_past_key_value:.6f} seconds")
        except Exception as e:
            logger.error(f"update past key value error {e}")
            raise e
        torch.cuda.nvtx.range_pop()
        # 
        # torch.cuda.synchronize()
        # if torch.isnan(query_states).any():
        #     logger.warning("NaN detected in query_states")
        # if torch.isnan(key_states).any():
        #     logger.warning("NaN detected in key_states")
        # if torch.isnan(value_states).any():
        #     logger.warning("NaN detected in value_states")
            
        torch.cuda.nvtx.range_push("repeat kv value")
        logger.debug(f"before repeat qkv key shape {key_states.shape}, value shape {value_states.shape}")

        qry_len = query_states.shape[-2]
        time_start = time.time()
        # key_states = LPModule.repeat_kv_local(key_states, num_key_value_groups, pool_memory)
        # value_states = LPModule.repeat_kv_local(value_states, num_key_value_groups, pool_memory)
        if qry_len > 1:
            key_states = repeat_kv(key_states, num_key_value_groups)
            value_states = repeat_kv(value_states, num_key_value_groups)
        # else do not need repeat kv
        logger.debug(f"repeat qkv cost {time.time()-time_start:.6f} seconds")
        logger.debug(f"q shape {query_states.shape}, k shape {key_states.shape}, v shape {value_states.shape}")
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("attention_mask")
        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        is_causal = True if causal_mask is None and q_len > 1 else False
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("scaled dot product attention")
        time_start = time.time()
        if qry_len > 1:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=0.0,
                enable_gqa=False,
                # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
                is_causal=is_causal,
            )
        else:
            attn_output = scaled_dot_product_attention_help(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=0.0,
                # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
                is_causal=is_causal,
            )
        logger.debug(f"dot attn cost {time.time()-time_start:.6f} seconds")

        device = layer.self_attn.q_proj.weight.device
        time_start_move = time.time()
        logger.debug(f"layer self attn q_proj device {device}")
        # when use output_tensor in scaled_dot_product_attention_help better to non_blocking=False
        if not attn_output.is_pinned():
            attn_output = attn_output.pin_memory()
            attn_output = attn_output.to(device=device, non_blocking=True)
        else:
            attn_output = attn_output.to(device=device, non_blocking=True)
        logger.debug(f"time cost move to cuda:1 {time.time() - time_start_move} s")
        torch.cuda.nvtx.range_pop()
        return attn_output, past_key_value

    def decoder_qkv_batch(layer, hidden_states, past_key_value, position_ids, attention_mask):
         #sdpa
        bsz, q_len, _ = hidden_states.size()
        
        logger.debug(f"hidden_states.shape {hidden_states.shape} hidden_states.dtype {hidden_states.dtype} device {hidden_states.device}")
        # 检查
        # 检查 input_layernorm 的参数有无 nan/inf
        input_layernorm_params = list(layer.input_layernorm.parameters())
        logger.debug(f"input_layernorm num_params: {len(input_layernorm_params)}")
        for idx, param in enumerate(input_layernorm_params):
            logger.debug(f"input_layernorm param {idx}: shape={param.shape}, dtype={param.dtype}, device={param.device}")
            if torch.isnan(param).any():
                logger.warning(f"NaN detected in input_layernorm param {idx}")
            if torch.isinf(param).any():
                logger.warning(f"Inf detected in input_layernorm param {idx}")

        if torch.isnan(hidden_states).any():
            logger.warning("NaN detected in hidden_states before layernorm")
        # input_layernorm
        hidden_states=layer.input_layernorm(hidden_states)
        # start qkv this layer
        # 查看 q_proj 的参数，有无异常
        if hasattr(layer.self_attn.q_proj, "weight"):
            weight = layer.self_attn.q_proj.weight
            if torch.isnan(weight).any():
                logger.warning("NaN detected in q_proj weight")
            if torch.isinf(weight).any():
                logger.warning("Inf detected in q_proj weight")
            if weight is None:
                logger.error("q_proj weight is None (空值)")
        else:
            logger.error("q_proj has no 'weight' attribute")
        query_states = layer.self_attn.q_proj(hidden_states)
        key_states = layer.self_attn.k_proj(hidden_states)
        value_states = layer.self_attn.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, layer.self_attn.num_heads, layer.self_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 2)

        # torch.cuda.synchronize()
        if torch.isnan(hidden_states).any():
            logger.warning("NaN detected in hidden_states")
        if torch.isnan(query_states).any():
            logger.warning("NaN detected in query_states")
        if torch.isnan(key_states).any():
            logger.warning("NaN detected in key_states")
        if torch.isnan(value_states).any():
            logger.warning("NaN detected in value_states")

        kv_seq_len = key_states.shape[-2]
        logger.debug(f"kv_seq_len {kv_seq_len} past_key_value {past_key_value.get_seq_length(layer.self_attn.layer_idx)}")
        # in batch do not add seq_length, deprecated
        # now could use this, can get accurate kv_seq_len for the whole batch
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(layer.self_attn.layer_idx)
        # 未正确设置 cos, sin device 在 attn rotary_emb 中, 不应该move 这里
        logger.debug(f"inv_freq device {layer.self_attn.rotary_emb.inv_freq.device}")
        logger.debug(f"kv_seq_len {kv_seq_len} position_ids {position_ids}")
        cos, sin = layer.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        cos = cos.to(device=value_states.device)
        sin = sin.to(device=value_states.device)
        position_ids = position_ids.to(device=value_states.device)

        logger.debug(f"query_states device {query_states.device}")
        logger.debug(f"key_states device {key_states.device}")
        logger.debug(f"value_states device {value_states.device}")
        logger.debug(f"position_ids device {position_ids.device}")
        logger.debug(f"cos device {cos.device} shape {cos.shape}")
        logger.debug(f"sin device {sin.device} shape {sin.shape}")
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        return (query_states, key_states, 
                value_states, sin, cos)

    def decoder_qkv(layer, hidden_states, past_key_value, position_ids, attention_mask):
         #sdpa
        bsz, q_len, _ = hidden_states.size()
        
        logger.debug(f"hidden_states.shape {hidden_states.shape} hidden_states.dtype {hidden_states.dtype} device {hidden_states.device}")
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
        logger.debug(f"kv_seq_len {kv_seq_len} past_key_value {past_key_value.get_seq_length(layer.self_attn.layer_idx)}")
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_seq_length(layer.self_attn.layer_idx)
        # 未正确设置 cos, sin device 在 attn rotary_emb 中, 不应该move 这里
        logger.debug(f"inv_freq device {layer.self_attn.rotary_emb.inv_freq.device}")
        logger.debug(f"kv_seq_len {kv_seq_len} position_ids {position_ids.shape} value {position_ids[0]}")
        cos, sin = layer.self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        cos = cos.to(device=value_states.device)
        sin = sin.to(device=value_states.device)

        logger.debug(f"query_states device {query_states.device}")
        logger.debug(f"key_states device {key_states.device}")
        logger.debug(f"value_states device {value_states.device}")
        logger.debug(f"position_ids device {position_ids.device}")
        logger.debug(f"cos device {cos.device} shape {cos.shape}")
        logger.debug(f"sin device {sin.device} shape {sin.shape}")
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        return (query_states, key_states, 
                value_states, sin, cos)