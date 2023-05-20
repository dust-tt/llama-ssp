from collections import namedtuple
from copy import deepcopy
from logging import debug, info
import math
from unittest.mock import patch
import numpy as np

from lssp import base

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


def valid_multiplication(mult_string):
    """Check if a multiplication string of the form "a * b = c" is valid."""
    try:
        c = int(mult_string.split(" = ")[1])
        a, b = map(int, mult_string.split(" = ")[0].split(" * "))
        return a * b == c
    except (ValueError, IndexError):
        return False


def measure_model_score(model, tokenizer, prompts, results):
    """Measure the model's score on the prompts."""
    info(f"Measuring model score on {len(prompts)} additions prompts")

    # Get model outputs
    outputs = []
    for prompt in prompts:
        debug(f"prompt = {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        input_ids = base.sample_model(model, input_ids, nb_tokens=25)
        output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        debug(f"outputs = {output}")
        outputs.append(output)

    # for each output retain only the parts after the prompt, first two lines
    outputs = [output[len(prompt):]
               for output, prompt in zip(outputs, prompts)]
    outputs = [output.split("\n")[:2] for output in outputs]

    # for each line, check the addition is correct and compute the percentage
    # of success
    prompted_success_rate = 0  # percentage of success on prompted additions
    generated_success_rate = 0  # percentage of success on generated additions
    for output, result in zip(outputs, results):
        score_prompted = int(output[0]) == result
        # on second line, parse the addition and check the result
        debug(f"Parsing addition on second line: {output[1]}")
        prompted_success_rate += score_prompted
        generated_success_rate += valid_multiplication(output[1])

    # return results
    prompted_success_rate /= len(outputs)
    generated_success_rate /= len(outputs)
    return prompted_success_rate, generated_success_rate


def print_results(prompted_success_rate,
                  generated_success_rate,
                  model_name, draft_name=None):
    """Print results of the addition eval"""
    print("-"*20)
    print(f"Results for {model_name}"
          + (f" with {draft_name}" if draft_name else ""))
    print(f"Prompted success rate: {prompted_success_rate}")
    print(f"Generated success rate: {generated_success_rate}")


@patch('base.sample_model')
def test_measure_model_score(mock_sample_model):
    """Test the measure_model_score function."""
    prompts, results = create_multiplication_prompts(1339, 10)

    class FakeTokenizer:
        def __call__(self, texts, return_tensors=None, padding=None):
            return namedtuple('Output', ['input_ids'])(deepcopy(texts))

        def decode(self, input_ids, skip_special_tokens=None):
            return input_ids

    def fake_sample_model(model, input_ids, nb_tokens):
        for i in range(len(input_ids)):
            # if i is even, the addition is correct
            if i % 2 == 0:
                input_ids[i] += f"{results[i]}"
                # another addition is generated
                input_ids[i] += f"\n{results[i]} + {results[i]} = "
                # the result is incorrect
                input_ids[i] += f"{results[i] - 1}"
            else:
                # the addition is incorrect
                input_ids[i] += f"{results[i] - 1}"
                # another addition is generated with a correct result
                input_ids[i] += (f"\n{results[i] - 1} + {results[i] - 1}"
                                 f" = {2 * results[i] - 2}")
        return input_ids

    # Use the fake tokenizer and fake sampling to test measure_model_score
    mock_sample_model.side_effect = fake_sample_model
    prompted_success_rate, generated_success_rate = measure_model_score(
        None, FakeTokenizer(), prompts, results
    )
    print(f"prompted_success_rate = {prompted_success_rate}")
    print(f"generated_success_rate = {generated_success_rate}")
    assert prompted_success_rate == 0.5
    assert generated_success_rate == 0.5


if __name__ == '__main__':
    test_measure_model_score()
