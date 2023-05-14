from lssp.ssp import ssp, stream_token_if_required, sample_fn
import sys
import time
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from termcolor import colored
torch.manual_seed(1339)

MAX_NEW_TOKENS = 64
llama7b_name = 'decapoda-research/llama-7b-hf'
llama13b_name = 'decapoda-research/llama-13b-hf'
llama30b_name = 'decapoda-research/llama-30b-hf'
llama65b_name = 'decapoda-research/llama-65b-hf'

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

free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_mem = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'

n_gpus = torch.cuda.device_count()


def max_memory(gpus, starting_gpu=0):
    assert 1 <= gpus <= n_gpus
    return {i: max_mem for i in range(starting_gpu, starting_gpu+gpus)}


def create_model(model_name, max_memory, load_in_8bit=True):
    return LlamaForCausalLM.from_pretrained(
        model_name,
        device_map='sequential',
        load_in_8bit=load_in_8bit,
        max_memory=max_memory
    )


def sample_model(model, input_ids, display=False):
    for _ in range(MAX_NEW_TOKENS):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = sample_fn(next_token_logits)
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
        stream_token_if_required(input_ids, stream=display)
    return input_ids


def time_model(model):
    # time the first run
    input_ids = tokenizer(texts[0], return_tensors="pt").input_ids
    input_ids = torch.stack([input_ids[0]] * batch_size).to(model.device)
    generated_ids = sample_model(model, input_ids)

    start_time = time.time()
    nb_tokens = 0
    for text in texts[1:]:
        print("Completing text:", text)
        intermediate_time = time.time()
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = torch.stack([input_ids[0]] * batch_size).to(model.device)
        generated_ids = sample_model(model, input_ids)
        nb_tokens += generated_ids.shape[1] - input_ids.shape[1]
        print("Completion: ", tokenizer.decode(
            generated_ids[0], skip_special_tokens=True))
        print("Time: {:.2f}s".format(time.time() - intermediate_time))
        print("========\n")
    ms_per_token = (time.time() - start_time)*1000 / nb_tokens
    return generated_ids, ms_per_token


def print_results(tokens_s, outputs, name='Noname'):
    print("Results for ", name)
    print(f"Ms per token: {tokens_s:.2f}ms")
    print("========\n")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("========\n")


models_params = {
    '7B_8bit': {'model_name': llama7b_name,
                'max_memory': max_memory(1),
                'load_in_8bit': True},
    '7B_8bit_2GPUs': {'model_name': llama7b_name,
                      'max_memory': max_memory(2),
                      'load_in_8bit': True},
    '7B_8bit_4GPUs': {'model_name': llama7b_name,
                      'max_memory': max_memory(4),
                      'load_in_8bit': True},
    '7B': {'model_name': llama7b_name,
           'max_memory': max_memory(1),
           'load_in_8bit': False},
    '7B_4GPUs': {'model_name': llama7b_name,
                 'max_memory': max_memory(4),
                 'load_in_8bit': False},
    '13B_8bit': {'model_name': llama13b_name,
                 'max_memory': max_memory(1),
                 'load_in_8bit': True},
    '13B': {'model_name': llama13b_name,
            'max_memory': max_memory(2, 1),
            'load_in_8bit': False},
    '30B_8bit': {'model_name': llama30b_name,
                 'max_memory': max_memory(2),
                 'load_in_8bit': True},
    '30B': {'model_name': llama30b_name,
            'max_memory': max_memory(4, 1),
            'load_in_8bit': False},
    '65B_8bit': {'model_name': llama65b_name,
                 'max_memory': max_memory(4),
                 'load_in_8bit': True},
    '65B': {'model_name': llama65b_name,
            'max_memory': max_memory(8),
            'load_in_8bit': False},
}


def time_ssp(target_name, draft_name, K=4):
    draft_model = create_model(**models_params[draft_name])
    target_model = create_model(**models_params[target_name])
    nb_tokens = 0
    # Warmup
    input_ids = tokenizer(texts[0], return_tensors="pt").input_ids
    input_ids = torch.stack(
        [input_ids[0]] * batch_size).to(draft_model.device)
    generated_ids = ssp(target_model,
                        draft_model,
                        MAX_NEW_TOKENS,
                        input_ids, K=K)

    start_time = time.time()
    for text in texts[1:]:
        print("Completing text:", text)
        intermediate_time = time.time()
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = torch.stack(
            [input_ids[0]] * batch_size).to(draft_model.device)
        generated_ids = ssp(target_model,
                            draft_model,
                            MAX_NEW_TOKENS,
                            input_ids, K=K)
        nb_tokens += generated_ids.shape[1] - input_ids.shape[1]
        print("Completion: ", tokenizer.decode(
            generated_ids[0], skip_special_tokens=True))
        print("Time: {:.2f}s".format(time.time() - intermediate_time))
        print("========\n")
    ms_per_token = (time.time() - start_time)*1000 / nb_tokens
    return generated_ids, ms_per_token


def print_speeds(speeds):
    print("Speeds:")
    for model_name, tokens_s in speeds.items():
        print('-'*20)
        print(f"{model_name} |  {tokens_s:.2f}ms")
    print('-'*20)


def models_raw_speed():
    speeds = {}
    del models_params['7B'], models_params['13B'], models_params['30B']
    for model_name, params in sorted(models_params.items()):
        print(f"Testing {model_name}")
        print('-'*20)
        model = create_model(**params)
        outputs, tokens_s = time_model(model)
        speeds[model_name] = tokens_s
        print_results(tokens_s, outputs, model_name)
        del model
        torch.cuda.empty_cache()
        print_speeds(speeds)
    draft_name = '7B_8bit'
    target_name = '65B_8bit'
    print(f"Testing SSP {draft_name} / {target_name}")
    tokens_s, outputs = time_ssp(draft_name, target_name)
    speeds[f"{draft_name} / {target_name}"] = tokens_s
    print(speeds)


def show_comparative_speeds(text, model, draft_model):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(colored("=> Regular sampling with target model",
                  attrs=['bold']))
    sys.stdout.write(text)
    start_time = time.time()
    sample_model(model, input_ids, display=True)
    print("\nTime: "
          + colored(f"{time.time() - start_time:.2f}s", 'red', attrs=['bold']))
    print(colored(
        "=> Speculative sampling with target model helped by draft model",
        attrs=['bold']))
    sys.stdout.write(text)
    start_time = time.time()
    ssp(model, draft_model, MAX_NEW_TOKENS,
        input_ids, K=4, display=True)
    print("\nTime: "
          + colored(f"{time.time() - start_time:.2f}s", 'green', attrs=['bold']))


if __name__ == "__main__":
    model_name = sys.argv[1]
    if sys.argv[1] == 'compare':
        model = create_model(**models_params[sys.argv[2]])
        draft_model = create_model(**models_params[sys.argv[3]])
        print("Warming up")
        ssp(model, draft_model, MAX_NEW_TOKENS,
            tokenizer(texts[0], return_tensors="pt").input_ids, K=4)
        print(
            f"Comparing {sys.argv[2]} model regular sampling and {sys.argv[2]} SSp with {sys.argv[3]} draft model\n====\n")
        # Read from stdin until EOF
        while True:
            try:
                sys.stdout.write("> ")
                sys.stdout.flush()
                text = input()
            except EOFError:
                break
            show_comparative_speeds(text, model, draft_model)
            """
            draft_time = time.time()
            gen_ids_draft = sample_model(draft_model,
                                            tokenizer(text, return_tensors="pt").input_ids)
            completion = tokenizer.decode(
                gen_ids_draft[0], skip_special_tokens=True)
            draft_time = time.time() - draft_time
            print(f"\n---\n Draft model completion: {completion}\nTime: {draft_time:.2f}s\n")
            """

    elif len(sys.argv) == 3:
        draft_name = sys.argv[2]
        print(f"Testing {model_name} with draft {draft_name}")
        print('-'*20)
        gen_ids, ms_per_token = time_ssp(model_name, draft_name)
        print_results(ms_per_token, gen_ids, model_name)

    else:
        print(f"Testing {model_name}")
        print('-'*20)
        model = create_model(**models_params[model_name])
        gen_ids, ms_per_token = time_model(model)
        print_results(ms_per_token, gen_ids, model_name)
