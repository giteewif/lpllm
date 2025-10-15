# from sllm_store.transformers import save_model

# # Load a model from HuggingFace model hub.
# import torch
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained('/mnt/zhengcf3/models/deepseek-moe-16b-base', torch_dtype=torch.float16, trust_remote_code=True)

# # Replace './models' with your local path.
# save_model(model, '/mnt/zhengcf3/models/models/deepseek-moe-16b-base')
import sys
import os



# Add the current directory to Python path
sys.path.insert(0, '/mnt/zhengcf3/lpllm')
from lpllm.lpllm import LPLLM
from lpllm.StaticCacheLen import StaticCacheLen
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache


import time
storage_path = "/mnt/zhengcf3/models/models"
tdevice = "cuda:0"

model_name = "deepseek-moe-16b-base"
# model_name = "Mixtral-8x7B"
model_path = f"/mnt/zhengcf3/models/models/{model_name}"


lp = LPLLM(model_name, tdevice, storage_path, pool_size=8, chunk_size_mb=2048)
config = lp.config


tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


batch_size = 256
seq_len = 512
vocab_size = config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=tdevice)

# text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
# input_ids = tokenizer(text, return_tensors="pt").input_ids.to(tdevice)
# input_ids = torch.cat([input_ids, input_ids], dim=0)
# batch_size = input_ids.shape[0]

key_states = torch.zeros(batch_size//2, 8, 1, 128, dtype=torch.bfloat16, device="cpu")

time_start_init_kv = time.time()
def init_kv_cache():
    # past_key_values = StaticCache(config, batch_size=batch_size//2, max_cache_len=544, device="cpu", dtype=config.torch_dtype)
    past_key_values = StaticCacheLen(config, batch_size=batch_size//2, max_cache_len=544, device="cpu", dtype=config.torch_dtype)
    # past_key_values = DynamicCache()
    return past_key_values
past_key_values1 = init_kv_cache()
past_key_values2 = init_kv_cache()
print(f"init kv cache time cost {time.time()-time_start_init_kv} s")
import psutil

process = psutil.Process(os.getpid())
mem_info = process.memory_info()
mem_used_mb = mem_info.rss / 1024 / 1024
print(f"当前程序内存占用: {mem_used_mb:.2f} MB")

# cache_kwargs={}
# cache_kwargs["cache_position"] = None
# past_key_values1.update(key_states, key_states, 0, cache_kwargs)
# print(f"get seq_length {past_key_values1.get_seq_length(0)} max {past_key_values1.get_max_cache_shape()}")

time_start = time.time()
generated_tokens = lp.generate(input_ids, past_key_values1, past_key_values2, max_new_tokens=1)
torch.cuda.synchronize()
elapsed = time.time() - time_start
print(f"cost {elapsed} s")
print(generated_tokens.shape)

# 计算token吞吐量
num_generated_tokens = generated_tokens.shape[0] * (generated_tokens.shape[1] - input_ids.shape[1])
throughput = num_generated_tokens / elapsed if elapsed > 0 else 0
print(f"Token throughput: {throughput:.2f} tokens/s")

result = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(result)
lp.stop()
