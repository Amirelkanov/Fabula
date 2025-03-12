import time
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from sampling import autoregressive_sampling, speculative_sampling

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

device = "cuda:4" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
print("Warming up...")
target_model = AutoModelForCausalLM.from_pretrained("facebook/opt-13b", torch_dtype=torch.float16).to(device)
draft_model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-13b")
print("Warm up has ended.")

prompt_text = "Once upon a time, there was a brave knight named Sir Lancelot. He embarked on a quest to"
input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
max_tokens = 100


start_time = time.time()
ars_output = autoregressive_sampling(target_model, input_ids, max_tokens+input_ids.shape[0])
ars_time = time.time() - start_time
ars_text = tokenizer.decode(ars_output[0], skip_special_tokens=True)

print(ars_text)
print(ars_time)

start_time = time.time()
sps_output = speculative_sampling(target_model, draft_model, input_ids, max_tokens+input_ids.shape[0])
ars_time = time.time() - start_time
ars_text = tokenizer.decode(ars_output[0], skip_special_tokens=True)

print(ars_text)
print(ars_time)
