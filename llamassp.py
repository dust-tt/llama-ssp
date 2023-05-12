import sys
import time
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

from lssp.ssp import ssp, FakeModel

MAX_NEW_TOKENS = 64
llama7b_name = 'decapoda-research/llama-7b-hf'
llama13b_name = 'decapoda-research/llama-13b-hf'
batch_size = 1

text = 'In which country is Hamburg?\n'
tokenizer = LlamaTokenizer.from_pretrained(llama7b_name)

# free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
# max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

# n_gpus = torch.cuda.device_count()
# max_memory = {i: max_memory for i in range(n_gpus)}


def create_model(model_name, max_memory, load_in_8bit=True):
    return LlamaForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        load_in_8bit=load_in_8bit,
        max_memory=max_memory
    )


def time_model(model):
    # time the first run
    start_time = time.time()
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = torch.stack([input_ids[0]] * batch_size).to(model.device)
    generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
    nb_tokens = generated_ids.shape[1] - input_ids.shape[1]
    token_per_sec = nb_tokens / (time.time() - start_time)
    return generated_ids, token_per_sec


def print_results(tokens_s, outputs, name='Noname'):
    print("Results for", name)
    print(f"Tokens per second: {tokens_s:.2f}/s")
    print("========\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("========\n")
    print(f"Tokens per second: {tokens_s:.2f}/s")


models_params = {
    '7B_8bit': {'model_name': llama7b_name,
                'max_memory': {0: '10GB'},
                'load_in_8bit': True},
    '13B_8bit': {'model_name': llama13b_name,
                 'max_memory': {0: '19GB'},
                 'load_in_8bit': True},
    '7B': {'model_name': llama7b_name,
           'max_memory': {0: '18GB'},
           'load_in_8bit': False},
}


def print_speeds(speeds):
    print("Speeds:")
    for model_name, tokens_s in speeds.items():
        print('-'*20)
        print(f"{model_name} |  {tokens_s:.2f}/s")
        print('-'*20)


if __name__ == '__main__':
    speeds = {}
    for model_name, params in models_params.items():
        model = create_model(**params)
        outputs, tokens_s = time_model(model)
        speeds[model_name] = tokens_s
        print_results(tokens_s, outputs, model_name)
        del model
        torch.cuda.empty_cache()
    print_speeds(speeds)
