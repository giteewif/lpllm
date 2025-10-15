from optparse import Option
from turtle import st
from transformers.cache_utils import Cache, StaticCache
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from lpllm.logger import init_logger

logger = init_logger(__name__)
class StaticCacheLen(StaticCache):
    def __init__(self, config, batch_size, max_cache_len, device, dtype):
        super().__init__(config, batch_size, max_cache_len, device, dtype)

        self.update_seq_length_batch_list = [ 0 for i in range(config.num_hidden_layers) ]
        self.seq_length_list = [ 0 for i in range(config.num_hidden_layers)]
        self.batch_length_list = [0 for i in range(config.num_hidden_layers)]
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        mode: str= "normal",
        # for mode other
        batch_dim: tuple[int,int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode == "normal":
            k_out, vout = super().update(key_states, value_states, layer_idx, cache_kwargs)

            self.seq_length_list[layer_idx] += key_states.shape[-2]
            self.batch_length_list[layer_idx] += key_states.shape[0]
            logger.debug(f"layer idx seq_length: {self.seq_length_list[layer_idx]}")
            return k_out, vout
        else:
            #update in other view
            cache_position = cache_kwargs.get("cache_position")

            k_out = self.key_cache[layer_idx]
            v_out = self.value_cache[layer_idx]
            key_states = key_states.to(k_out.dtype)
            value_states = value_states.to(v_out.dtype)

            if cache_position is None:
                raise ValueError(f"cache_position is None in mode {mode}")
            else:
                # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
                # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
                # operation, that avoids copies and uses less memory.
                if mode == "update_batch":
                    if batch_dim is None:
                        start_batch = 0
                        end_batch = key_states.shape[0]
                    try:
                        # start_batch, end_batch = batch_dim
                        # k_out[start_batch:end_batch].index_copy_(2, cache_position, key_states)
                        # v_out[start_batch:end_batch].index_copy_(2, cache_position, value_states)
                        start_batch, end_batch = batch_dim
                        k_out[start_batch:end_batch, :, cache_position] = key_states
                        v_out[start_batch:end_batch, :, cache_position] = value_states
                    except NotImplementedError:
                        start_batch, end_batch = batch_dim
                        k_out[start_batch:end_batch, :, cache_position] = key_states
                        v_out[start_batch:end_batch, :, cache_position] = value_states

                    self.batch_length_list[layer_idx] += key_states.shape[0]
                    self.seq_length_list[layer_idx] = key_states.shape[-2]
                    
                elif mode == "update_seq":

                    batch_size = key_states.shape[0]
                    update_batch_loc = self.update_seq_length_batch_list[layer_idx]
                    start_batch = update_batch_loc
                    end_batch = update_batch_loc + batch_size
                    self.update_seq_length_batch_list[layer_idx] = end_batch

                    try:
                        # k_out[start_batch:end_batch].index_copy_(2, cache_position, key_states)
                        # v_out[start_batch:end_batch].index_copy_(2, cache_position, value_states)
                        k_out[start_batch:end_batch, :, cache_position] = key_states
                        v_out[start_batch:end_batch, :, cache_position] = value_states
                    except NotImplementedError:
                        k_out[start_batch:end_batch, :, cache_position] = key_states
                        v_out[start_batch:end_batch, :, cache_position] = value_states
                    
                    logger.debug(f"static cache update layer_idx: {layer_idx}, start_batch: {start_batch}, end_batch: {end_batch}")
                    # max_batch_size in super class StaticCache
                    # if update batch to max_batch_size, then update the seq_length and reset update seq_length
                    # for every batch update, default they with same seq_length which is key_states.shape[-2]
                    if self.update_seq_length_batch_list[layer_idx] == self.max_batch_size:
                        self.seq_length_list[layer_idx] += key_states.shape[-2]
                        logger.debug(f"static cache update layer_idx: {layer_idx}, update seq_length to {self.seq_length_list[layer_idx]}")
                        self.update_seq_length_batch_list[layer_idx] = 0
                else:
                    raise ValueError(f"not support the mode {mode}")

            return k_out, v_out
    def get_batch_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.batch_length_list[layer_idx]
    def update_batch_loc(self, layer_idx: int) -> int:
        return self.update_seq_length_batch_list[layer_idx]
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        # origin_length = super().get_seq_length(layer_idx)
        # seq_length = max(self.seq_length_list[layer_idx], origin_length)
        seq_length = self.seq_length_list[layer_idx]
        return seq_length