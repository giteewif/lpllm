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
tdevice = "cuda:1"

# model_name = "deepseek-moe-16b-base"
model_name = "Mixtral-8x7B"
model_path = f"/mnt/zhengcf3/models/models/{model_name}"


lp = LPLLM(model_name, tdevice, storage_path, pool_size=8, chunk_size_mb=2048)
config = lp.config


tokenizer=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 60, 120, 240, 360, 480, 720, 960, 1440, 1920
batch_size = 60
seq_len = 512
vocab_size = config.vocab_size
# 随机可能导致nan值
# input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=tdevice)

# 给出一段长度为512的text
text = (
    # 有712个token
    "Deep learning has rapidly evolved over the past decade, transforming many fields through breakthroughs in neural architectures and large-scale models. One of the most significant advancements in this area is the emergence of attention mechanisms, particularly the transformer architecture. The transformer model, introduced by Vaswani et al. in 2017, innovated the way neural networks handle sequential data by allowing each element in an input sequence to attend to all other elements through self-attention. This design enables the effective modeling of dependencies regardless of their distance in the sequence. As a result, transformers have demonstrated exceptional performance in a variety of natural language processing (NLP) tasks such as machine translation, text summarization, sentiment analysis, and question answering. They accomplish this by projecting input embeddings into distinct query, key, and value vectors, then calculating attention scores, which determines how much focus should be applied to different parts of the input when building representations for downstream tasks. The scalability of transformers to large datasets and complex tasks has been supported by advances in computational hardware, optimization algorithms, and training techniques such as mixed precision and model parallelism. Recently, mixture-of-experts models and retrieval-augmented methods have pushed the boundaries of what language models are capable of, introducing dynamic parameter utilization and integrating external knowledge bases for even greater contextual understanding. Memory efficiency is a persistent challenge, especially as models grow into the billions of parameters and train across ever longer sequences. To address this, researchers have proposed several variants, including sparse attention, sliding window mechanisms, and memory-efficient transformer implementations. These approaches alleviate the quadratic complexity associated with full self-attention, enabling practical training and inference for lengthy text segments without excessive hardware demands. As NLP systems become increasingly integrated into products like virtual assistants, AI chatbots, automatic summarizers, and enterprise solutions, the reliability, interpretability, and safety of these models are paramount. Thus, explainability research has highlighted the role of attention patterns in providing interpretable signals about how models make decisions. Applications in biomedical text mining, code generation, creative writing, and scientific discovery further showcase the flexibility of attention-based architectures. With continual pretraining and transfer learning, transformers acquire broad world knowledge and generalize remarkably well across languages and domains. The integration of multimodal information, combining text, images, and audio, is another exciting frontier enabled by attention's capacity to handle diverse data sources in a unified framework. Collaboration between open-source communities, academia, and industry accelerates progress, democratizing access to powerful models and fostering fast-paced innovation. Evaluations on increasingly challenging benchmarks, such as SuperGLUE, Massive Multi-Task Language Understanding, and open-domain question answering, drive improvements and highlight both strengths and limitations. Despite remarkable progress, important questions remain around data efficiency, robustness to adversarial attacks, fairness, and societal impacts. Researchers are exploring methods for aligning large language models with human values and intent, leveraging reinforcement learning from human feedback and continual adaptation. The future of attention-based models is bright, with ongoing efforts poised to unlock even greater possibilities in language, reasoning, and creative problem solving, ensuring that artificial intelligence continues its upward trajectory in benefiting society in countless ways."
)
input_ids = tokenizer(text, return_tensors="pt").input_ids.to(tdevice)
input_ids = input_ids.repeat(batch_size, 1)
input_ids = input_ids[:,:seq_len]
print(input_ids.shape)

time_start_init_kv = time.time()
def init_kv_cache():
    # past_key_values = StaticCache(config, batch_size=batch_size//2, max_cache_len=544, device="cpu", dtype=config.torch_dtype)
    # past_key_values = StaticCacheLen(config, batch_size=batch_size//2, max_cache_len=544, device="cpu", dtype=config.torch_dtype)
    past_key_values = StaticCacheLen(config, batch_size=batch_size, max_cache_len=544, device="cpu", dtype=config.torch_dtype)
    # past_key_values = DynamicCache()
    return past_key_values
past_key_values1 = init_kv_cache()
# past_key_values2 = init_kv_cache()
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
generated_tokens, generated_new_tokens = lp.generate_batch(input_ids, past_key_values1, encoder_batch_size=60, gpu_batch_size=1, max_new_tokens=32)
torch.cuda.synchronize()
elapsed = time.time() - time_start
print(f"generate cost {elapsed} s")
print(generated_tokens.shape)
# 计算token吞吐量
num_generated_tokens = generated_tokens.shape[0] * (generated_tokens.shape[1] - input_ids.shape[1])
throughput = num_generated_tokens / elapsed if elapsed > 0 else 0
print(f"batch size {batch_size} seq_len {seq_len} generated_tokens len {generated_tokens.shape[1]} generated_new_tokens len {generated_new_tokens.shape[1]}")
print(f"Token throughput: {throughput:.2f} tokens/s")
# result = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
# print(result)
lp.stop()
 