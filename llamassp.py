import sys
import time
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

from lssp.ssp import ssp, FakeModel

MAX_NEW_TOKENS = 32
llama7b_name = 'decapoda-research/llama-7b-hf'
llama13b_name = 'decapoda-research/llama-13b-hf'
batch_size = 1

texts = [
    'In which country is Hamburg?\n',
    'How are you doing today?\n',
    'It was a dark and stormy night.',
    'The sun rose slowly over the horizon, casting a warm glow on the world below.',
    'I never believed in ghosts until the day I met one.',
    'The sound of the train whistle echoed through the valley as I stood at the station, waiting.',
    'She walked into the room and everything changed.',
    'The smell of freshly baked bread filled the air as I entered the bakery.',
    'The first time I saw her, I knew she was trouble.'
    'The world was ending, and I was the only one who knew.',
    'It was the best of times, it was the worst of times.',
    'The forest was alive with the sound of animals as I walked deeper into the woods.',
    'As I looked out over the city, I knew that anything was possible.',
    'The sound of gunfire echoed through the streets as I ran for cover.',
    'The waves crashed against the shore, a never-ending cycle of destruction and creation.',
    'I woke up to find myself in a strange place, with no memory of how I got there.',
    'The clock struck midnight, and I knew that my life would never be the same.',]
tokenizer = LlamaTokenizer.from_pretrained(llama7b_name)

# free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
# max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

# n_gpus = torch.cuda.device_count()
# max_memory = {i: max_memory for i in range(n_gpus)}


def create_model(model_name, max_memory, load_in_8bit=True):
    return LlamaForCausalLM.from_pretrained(
        model_name,
        device_map='sequential',
        load_in_8bit=load_in_8bit,
        max_memory=max_memory
    )


def time_model(model):
    # time the first run
    start_time = time.time()
    nb_tokens = 0
    for text in texts:
        print("Completing text:", text)
        intermediate_time = time.time()
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = torch.stack([input_ids[0]] * batch_size).to(model.device)
        generated_ids = model.generate(input_ids, max_length=MAX_NEW_TOKENS)
        nb_tokens += generated_ids.shape[1] - input_ids.shape[1]
        print("Completion: ", tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        print("Time: {:.2f}s".format(time.time() - intermediate_time))
        print("========\n")
    token_per_sec = nb_tokens / (time.time() - start_time)
    return generated_ids, token_per_sec


def print_results(tokens_s, outputs, name='Noname'):
    print("Results for", name)
    print(f"Tokens per second: {tokens_s:.2f}/s")
    print("========\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("========\n")


models_params = {
    '7B_8bit': {'model_name': llama7b_name,
                'max_memory': {0: '10GB'},
                'load_in_8bit': True},
    '13B_8bit': {'model_name': llama13b_name,
                 'max_memory': {0: '20GB'},
                 'load_in_8bit': True},
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
