from abc import ABC, abstractmethod
class LPModuleWrapper(ABC):
    """模型基类:定义通用接口"""
    @abstractmethod
    def get_model(config):
        pass
    # get init layer
    @abstractmethod
    # layers_map {layer_idx: layer}
    def get_layer(config):
        pass
    @abstractmethod
    def get_layer_attr_info() -> tuple[str, int]:
        pass
    @abstractmethod
    def check_layer_is_dense(config, layer_idx):
        pass
    @abstractmethod
    def decoder_mlp(layer, mlp_hidden_states, mlp_o_hidden_states, attn_func):
        pass
    @abstractmethod
    def decoder_qkv(layer, hidden_states, past_key_value, attention_mask, position_ids):
        pass
    @abstractmethod
    def decoder_attn(layer, bsz, q_len, layer_idx, query_states, key_states, value_states, sin, cos, attention_mask, past_key_value):
        pass
    @abstractmethod
    def get_embed_tokens_norm_lm_head(config):
        pass
    @abstractmethod
    def forward_prepare(
        model,
        input_ids,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        use_cache=True,
        attention_mask=None,
    ):
        pass
class DecoderLayerWrapper(ABC):
    """模型层:定义通用接口"""
    @abstractmethod
    def get_layer(config, layer_idx: int):
        pass
    