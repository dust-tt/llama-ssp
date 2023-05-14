# Implementation of speculative sampling as per
# https://arxiv.org/abs/2302.01318
from collections import namedtuple
import sys
import torch

from torch import nn
from transformers import LlamaTokenizer
from logging import info, debug, warning, error, critical
torch.manual_seed(1339)
llama7b_name = 'decapoda-research/llama-7b-hf'
tokenizer = LlamaTokenizer.from_pretrained(llama7b_name)

TEMPERATURE = 0.7

def stream_token_if_required(input_ids, stream=False):
    if stream is True:
        output_string = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        previous_output_string = tokenizer.decode(input_ids[0][:-1], skip_special_tokens=True)
        sys.stdout.write(output_string[len(previous_output_string):])
        sys.stdout.flush()


def get_temperature_distribution(logits, temperature=TEMPERATURE):
    return torch.softmax(logits / temperature, dim=-1)


def sample_fn(logits, temperature=TEMPERATURE):
    probs = get_temperature_distribution(logits, temperature)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _draft_sample_k(model, input_ids, K):
    """sample K tokens from the draft model autoregressively
    draft_logits are a (B, K, V) tensor
    inputs_plus_k are a (B, T+K) tensor
    """
    inputs_plus_k = input_ids
    draft_logits = []
    for t in range(K):
        outputs = model(inputs_plus_k)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = sample_fn(next_token_logits)
        inputs_plus_k = torch.cat(
            [inputs_plus_k, next_token_id.unsqueeze(1)],
            dim=1)
        draft_logits.append(next_token_logits)
    draft_logits = torch.stack(draft_logits, dim=1)
    return inputs_plus_k, draft_logits


def _target_sample_from_distribution(target_distribution, draft_distribution):
    distribution = (target_distribution - draft_distribution)
    distribution = torch.max(distribution,
                             torch.zeros_like(distribution))
    distribution = distribution / distribution.sum(dim=-1, keepdim=True)
    return torch.multinomial(distribution, num_samples=1).squeeze(-1)


def _ssp_iteration(target_model, draft_model, input_ids, K=4, display=False):

    _, T = input_ids.shape
    # sample K tokens from the draft model autoregressively
    # draft_logits are a (B, K, V) tensor
    # inputs_plus_k are a (B, T+K) tensor
    inputs_plus_k, draft_logits = _draft_sample_k(
        draft_model, input_ids, K
    )
    debug(
        f"Possible continuations: {tokenizer.decode(inputs_plus_k[0,T:], skip_special_tokens=True)}")
    # get the logits for the same tokens from the target model
    # target_logits are a (B, K+1, V) tensor
    # TODO avoid using .logits since it is HF-specific
    target_logits = target_model(inputs_plus_k).logits[:, -K-1:, :]
    target_distribution = get_temperature_distribution(target_logits)
    draft_distribution = get_temperature_distribution(draft_logits)
    # Accept-reject token loop
    all_accepted = True
    for t in range(1, K+1):
        sampled_ratios = (
            target_distribution[:1, t-1, inputs_plus_k[0, T+t-1]]
            / draft_distribution[:1, t-1, inputs_plus_k[0, T+t-1]]
        )
        # print(f"Sampled ratios: {sampled_ratios}")
        sampled_ratios = torch.min(sampled_ratios,
                                   torch.ones_like(sampled_ratios))
        rs = torch.rand_like(sampled_ratios)

        if (rs < sampled_ratios).any():  # TODO for B > 1, change this
            input_ids = torch.cat(
                [input_ids, inputs_plus_k[:, T + t-1].unsqueeze(1)],
                dim=1)
            stream_token_if_required(input_ids, stream=display)

        else:
            all_accepted = False
            next_token_id = _target_sample_from_distribution(
                target_distribution[:1, t-1, :],
                draft_distribution[:1, t-1, :])
            input_ids = torch.cat(
                [input_ids, next_token_id.unsqueeze(1)],
                dim=1)
            stream_token_if_required(input_ids, stream=display)
            break

    # if all tokens were accepted, sample a last one
    if all_accepted:
        next_token_id = sample_fn(target_logits[:1, -1, :])
        input_ids = torch.cat(
            [input_ids, next_token_id.unsqueeze(1)],
            dim=1)
        stream_token_if_required(input_ids, stream=display)
    debug(
        f"Accepted continuations: {tokenizer.decode(input_ids[0,T:], skip_special_tokens=True)}")
    return input_ids


def ssp(target_model, draft_model, min_nb_tokens, input_ids, K=4, display=False):
    B, T = input_ids.shape
    assert B == 1, "Batch size must be 1, implement the fixes for B > 1"

    while input_ids.shape[1] < T + min_nb_tokens:
        debug("Current length:", input_ids.shape[1])
        input_ids = _ssp_iteration(target_model, draft_model, input_ids, K, display)
    return input_ids


class FakeModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def __call__(self, input_ids):
        # Create fake logits randomly in the range [-1, 1]
        B, T = input_ids.shape
        logits = torch.rand(B, T, self.vocab_size) * 2 - 1
        return namedtuple('Output', ['logits'])(logits)


if __name__ == '__main__':
    # Test the SSP implementation
    vocab_size = 10
    target_model = FakeModel(vocab_size)
    draft_model = FakeModel(vocab_size)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    print(ssp(target_model, draft_model, 10, input_ids))
