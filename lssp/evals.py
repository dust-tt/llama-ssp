# Prompts

from collections import namedtuple
from copy import deepcopy
from unittest.mock import patch
import numpy as np

# from base import sample_model
import base

base_prompt = """1 + 4 = 5
18 + 12 = 30
27 + 38 = 65
123 + 456 = 579
"""


def create_additions_prompts(seed, number):
    """Create a batch of `number` prompts for additions."""
    np.random.seed(seed)
    prompts = []
    results = []
    for _ in range(number):
        a, b = np.random.randint(0, 200, size=2)
        prompts.append(f"{base_prompt}{a} + {b} = ")
        results.append(a + b)
    return prompts, results


def measure_model_score(model, tokenizer, prompts, results):
    """Measure the model's score on the prompts."""
    # Get model outputs
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = base.sample_model(model, inputs.input_ids, nb_tokens=15)
    outputs = tokenizer.decode(input_ids, skip_special_tokens=True)

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
        score_generated = (int(output[1].split("=")[1])
                           == sum(map(int, output[1].split("=")[0].split("+"))))
        prompted_success_rate += score_prompted
        generated_success_rate += score_generated
    prompted_success_rate /= len(outputs)
    generated_success_rate /= len(outputs)
    return prompted_success_rate, generated_success_rate


@patch('base.sample_model')
def test_measure_model_score(mock_sample_model):
    """Test the measure_model_score function."""
    prompts, results = create_additions_prompts(1339, 10)
    # fake tokenizer and model classes

    class FakeTokenizer:
        def __call__(self, texts, return_tensors=None, padding=None):
            return namedtuple('Output', ['input_ids'])(deepcopy(texts))

        def decode(self, input_ids, skip_special_tokens=None):
            return input_ids

    def fake_sample_model(model, input_ids, nb_tokens):
        for i in range(len(input_ids)):
            # if i is even, the addition is correct
            if i % 2 == 0:
                input_ids[i] +=f"{results[i]}"
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
