import torch
import time
from typing import Optional
import math
from torch.nn.attention import sdpa_kernel, SDPBackend
# def scaled_dot_product_attention_with_pinned_memory(
#     query: torch.Tensor,
#     key: torch.Tensor, 
#     value: torch.Tensor,
#     output_tensor: Optional[torch.Tensor] = None,
#     attn_mask: Optional[torch.Tensor] = None,
#     dropout_p: float = 0.0,
#     is_causal: bool = False,
#     scale: Optional[float] = None,
#     enable_gqa: bool = False
# ) -> torch.Tensor:
#     """
#     修改版的 scaled_dot_product_attention，支持将结果直接写入预分配的 pinned memory。
    
#     Args:
#         query: Query tensor of shape (..., L, E)
#         key: Key tensor of shape (..., S, E) 
#         value: Value tensor of shape (..., S, Ev)
#         output_tensor: 预分配的 pinned memory tensor，用于存储结果。如果为 None，则创建新的 tensor
#         attn_mask: 可选的注意力掩码
#         dropout_p: Dropout 概率
#         is_causal: 是否使用因果掩码
#         scale: 缩放因子，如果为 None 则使用 1/sqrt(E)
#         enable_gqa: 是否启用分组查询注意力
        
#     Returns:
#         注意力输出 tensor，如果提供了 output_tensor 则返回该 tensor
#     """
#     L, S = query.size(-2), key.size(-2)
#     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
#     # 计算输出形状
#     output_shape = query.shape[:-1] + (value.shape[-1],)
    
#     # 如果提供了预分配的输出 tensor，验证其形状和类型
#     if output_tensor is not None:
#         # 注意：对于 GPU tensor，即使原始是 pinned memory，移动到 GPU 后也不再是 pinned
#         if output_tensor.shape != output_shape:
#             raise ValueError(f"output_tensor 形状 {output_tensor.shape} 与期望形状 {output_shape} 不匹配")
#         if output_tensor.dtype != query.dtype:
#             raise ValueError(f"output_tensor 数据类型 {output_tensor.dtype} 与 query 数据类型 {query.dtype} 不匹配")
#         if output_tensor.device != query.device:
#             raise ValueError(f"output_tensor 设备 {output_tensor.device} 与 query 设备 {query.device} 不匹配")
    
#     # 创建注意力偏置
#     attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
#     if is_causal:
#         assert attn_mask is None, "is_causal 和 attn_mask 不能同时使用"
#         temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
#         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
#         attn_bias = attn_bias.to(query.dtype)

#     if attn_mask is not None:
#         if attn_mask.dtype == torch.bool:
#             attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
#         else:
#             attn_bias += attn_mask

#     # 处理分组查询注意力
#     if enable_gqa:
#         key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
#         value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

#     # 计算注意力权重
#     attn_weight = query @ key.transpose(-2, -1) * scale_factor
#     attn_weight += attn_bias
#     attn_weight = torch.softmax(attn_weight, dim=-1)
    
#     # 应用 dropout
#     if dropout_p > 0.0:
#         attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
#     # 计算最终输出
#     if output_tensor is not None:
#         # 直接写入预分配的 pinned memory
#         torch.matmul(attn_weight, value, out=output_tensor)
#         return output_tensor
#     else:
#         # 创建新的 tensor
#         return attn_weight @ value

batch_size = 1440
device="cuda:1"
query_states = torch.randn(batch_size, 32, 1, 128, dtype=torch.bfloat16, device="cpu").pin_memory()
# 480MB, *4 = 1920MB
key_states = torch.randn(batch_size, 8, 512, 128, dtype=torch.bfloat16, device="cpu").pin_memory()
value_states = torch.randn(batch_size, 8, 512, 128, dtype=torch.bfloat16, device="cpu").pin_memory()

hidden_states_cache = torch.zeros(batch_size, 8, 4, 512, 128, dtype=torch.bfloat16, device="cpu").pin_memory()
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: 
        return hidden_states
    # 正确的repeat_kv实现
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return expanded.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


time_start = time.time()
torch.cuda.nvtx.range_push("repeat kv")
new_key_states = repeat_kv(key_states, 4)
new_value_states = repeat_kv(value_states, 4)
torch.cuda.nvtx.range_pop()
print(f"time cost repeat kv {time.time() - time_start} s")

torch.cuda.nvtx.range_push("dot attention")
time_start = time.time()
attn_output = torch.nn.functional.scaled_dot_product_attention(
    query_states,
    new_key_states,
    new_value_states,
    attn_mask=None,
    dropout_p=0.0,
    enable_gqa = False,
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal=False,
)
print(f"dot attn cost {time.time()-time_start:.6f} seconds")
torch.cuda.nvtx.range_pop()


# 按多组分别对kv和query分头计算，再汇总结果
time_start = time.time()
num_groups = 4   # 4 组
num_query_heads = query_states.shape[1]     # e.g. 32
heads_per_group = num_query_heads // num_groups   # e.g. 8

attn_outputs_per_group = []

for group_idx in range(num_groups):
    # 不扩展kv，而是使用整个kv，query取值需变化
    # 每组使用整个key/value heads，但query heads按步数提取
    # 例如：32个query heads，8个key/value heads
    # 按步数提取query heads
    query_indices = torch.arange(group_idx, num_query_heads, num_groups)
    query_group = query_states[:, query_indices, :, :]  # (batch, heads_per_group, seq_len, head_dim)
    print(query_group.shape)
    key_group = key_states    # (batch, 8, seq_len, head_dim)
    value_group = value_states # (batch, 8, seq_len, head_dim)

    time_start_tmp = time.time()
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        query_group, key_group, value_group,
        attn_mask=None,
        dropout_p=0.0,
        enable_gqa=False,
        is_causal=False
    )
    print(f"real attn out cost {time.time() - time_start_tmp} s")
    # print(f"Group {group_idx}: query_indices={query_indices.tolist()}, kv_heads=all, attn_out.shape={attn_out.shape}")
    # instead of simply appending to a list, collect all group results in the correct head positions
    # (batch, num_query_heads, seq_len, head_dim) for all groups
    if group_idx == 0:
        attn_outputs_full = torch.zeros(
            query_states.shape, dtype=attn_out.dtype, device=attn_out.device
        )
    attn_outputs_full[:, query_indices, :, :] = attn_out

print(f"dot attn cost {time.time()-time_start:.6f} seconds")

if torch.allclose(attn_output, attn_outputs_full, atol=1e-6):
    print("attn_output == attn_output_group")
else:
    print("attn_output != attn_output_group")

# ==== 多线程版本 ====
import threading

print("\n=== Multithreaded group attention ===")
time_start = time.time()

attn_outputs_full_threaded = torch.zeros(
    query_states.shape, dtype=query_states.dtype, device=query_states.device
)

def compute_group_attn(group_idx):
    query_indices = torch.arange(group_idx, num_query_heads, num_groups)
    query_group = query_states[:, query_indices, :, :]
    key_group = key_states
    value_group = value_states
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        query_group, key_group, value_group,
        attn_mask=None,
        dropout_p=0.0,
        enable_gqa=False,
        is_causal=False
    )
    attn_outputs_full_threaded[:, query_indices, :, :] = attn_out

threads = []
for group_idx in range(num_groups):
    t = threading.Thread(target=compute_group_attn, args=(group_idx,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print(f"dot attn multithreaded cost {time.time()-time_start:.6f} seconds")

if torch.allclose(attn_output, attn_outputs_full_threaded, atol=1e-6):
    print("attn_output == attn_output_group_threaded")
else:
    print("attn_output != attn_output_group_threaded")