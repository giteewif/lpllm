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
import torch
from transformers import AutoTokenizer, AutoConfig

storage_path = "/mnt/zhengcf3/models/models"
model_path = "/mnt/zhengcf3/models/models/deepseek-moe-16b-base"
tdevice = "cuda:0"


lp = LPLLM("deepseek-moe-16b-base", tdevice, storage_path)
config = lp.config
tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

batch_size = 360
seq_len = 512
vocab_size = config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=tdevice)
layer_output, _, past_key_values, position_ids = lp.split_decoders(input_ids)
print(layer_output.shape)
lp.stop()
