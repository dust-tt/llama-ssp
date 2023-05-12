import time
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

from lssp.ssp import ssp

MAX_NEW_TOKENS = 64
llama7b_name = 'decapoda-research/llama-7b-hf'
llama13b_name = 'decapoda-research/llama-13b-hf'
batch_size = 1

text = 'Hamburg is in which country?\n'
tokenizer = LlamaTokenizer.from_pretrained(llama7b_name)

# free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
# max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

# n_gpus = torch.cuda.device_count()
# max_memory = {i: max_memory for i in range(n_gpus)}


def create_model(model_name, max_memory):
    return LlamaForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        load_in_8bit=True,
        max_memory=max_memory
    )


def try_model(model):
    # time the first run
    start_time = time.time()
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = torch.stack([input_ids[0]] * batch_size).to(model.device)
    print("========")
    print("HF generation:")
    generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    mid_time = time.time()
    print(f"Time for HF generation: {mid_time - start_time:.2f}s")
    print("========\n")
    print("Manual generation (greedy)")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = torch.stack([input_ids[0]] * batch_size).to(model.device)
    for _ in range(MAX_NEW_TOKENS):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        # probs = torch.softmax(next_token_logits.float(), dim=-1)
        # next_token_id = torch.multinomial(
        #    probs / 0.2, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    mid_time2 = time.time()
    print(f"Time for manual generation: {mid_time2 - mid_time:.2f}s")
    print("========\n")
    print("SSP generation:")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = torch.stack([input_ids[0]] * batch_size).to(model.device)
    input_ids = ssp(model, MAX_NEW_TOKENS, model, input_ids, K=4)
    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    end_time = time.time()
    print(f"Time for SSP generation: {end_time - mid_time2:.2f}s")


if __name__ == '__main__':
    model = create_model(llama7b_name, max_memory={0: '7GB'})
    try_model(model)
