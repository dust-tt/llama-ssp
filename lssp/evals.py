from collections import namedtuple
from copy import deepcopy
from logging import debug, info
from unittest.mock import patch
import numpy as np

from lssp import base
from lssp.ssp import ssp

base_prompt = """8 * 12 = 96
25 * 4 = 100
37 * 85 = 3145
"""


def create_multiplication_prompts(seed, number):
    """Create a batch of `number` prompts for additions."""
    np.random.seed(seed)
    prompts = []
    results = []
    for _ in range(number):
        a, b = np.random.randint(1, 99, size=2)
        prompts.append(f"{base_prompt}{a} * {b} = ")
        results.append(a * b)
    return prompts, results


def valid_multiplication(output_string, result):
    """Check if a multiplication string of the form "a * b = c" is valid."""
    try:
        if int(output_string) == result:
            return 1
        else:
            return 0
    except (ValueError, IndexError):
        return False


def confidence_interval(successes, trials):
    """Compute the 95% confidence interval of a binomial distribution."""
    p = successes / trials
    interval_size = 1.96 * np.sqrt(p * (1 - p) / trials)
    return (p - interval_size, p + interval_size)


def measure_model_score(model, tokenizer, nb_prompts, seed, draft_model=None):
    """Measure the model's score on the prompts."""
    # Generate prompts
    prompts, results = create_multiplication_prompts(seed, nb_prompts)
    info(f"Measuring model score on {len(prompts)} additions prompts")

    # Get model outputs
    outputs = []
    for prompt in prompts:
        debug(f"prompt = {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        if draft_model is not None:
            input_ids = ssp(model, draft_model, 10, input_ids)
        else:
            input_ids = base.sample_model(model, input_ids, nb_tokens=6)
        output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        debug(f"outputs = {output}")
        outputs.append(output)

    # for each output retain only the result of the first multiplication
    outputs = [output[len(prompt):]
               for output, prompt in zip(outputs, prompts)]
    outputs = [output.split("\n")[:1] for output in outputs]

    # for each line, check the multiplication is correct and compute the
    # percentage of success
    successes = 0  # percentage of success on prompted additions
    for output, result in zip(outputs, results):
        successes += valid_multiplication(output[0], result)

    return {
        'trials': len(prompts),
        'successes': successes,
        'success_rate': successes / len(prompts),
        'confidence_interval': confidence_interval(successes, len(prompts)),
        'seed': seed,
    }


def print_results(results, model_name, draft_name=None):
    """Print results of the multiplication eval"""
    print("-"*20)
    print(f"Eval Results for {model_name}"
          + (f" with {draft_name}" if draft_name else ""))
    print(f"Prompted success rate:\n{results}")
