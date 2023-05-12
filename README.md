# Llama SSp

Experiments on speedups that Speculative Sampling (SSp) can bring to Llama models.

Prepare a GPU machine with 4 or more GPUs: ``./machine-install.sh``

Setup the project (virtual env, requirements): ``./setup.sh``

Run the XPs (not on your local machine): 

```python3 llamassp.py MODEL_NAME [DRAFT_NAME] 2> err.log```

If DRAFT_NAME is not specified, models are run with regular sampling on a few examples, and model latency is measured.
If DRAFT_NAME is specified, speculative sampling latency is measured.

## Regular sampling speed
Example:
```python3 llamassp.py 13B_8bit```
This will perform completions on 15 examples (+ 1 warmup not shown), and finally output the model generation latency in ms/token (so lower is better).

MODEL_NAME correspond to various flavors of Llama models (7B to 65B), with or without quantization. Possible values are `7B_8bit, 7B, 13B_8bit, 13B, 30B_8bit, 65B_8bit`

The full model configs are defined as `model_params` in `llamassp.py` and can be completed/changed as required -- ensuiring that there is enough memory for the model to run.

Models run on the number of GPUs specified in `model_params`, e.g. `7B_8bit` runs on a single GPU, and a specific version `7B_8bit_4GPUs` runs on 4 gpus.

Currently, on a g5.12xlarge AWS instance, timings should look like those:


|Model_type | Ms/token|
|---|---|
|7B_4GPUs |  86ms|
|7B_8bit | 210ms|
|7B_8bit_4GPUs |  216ms|
|13B_8bit |  270ms|
|30B_8bit |  405ms|
|65B_8bit |  530ms|


## Speculative Sampling speed
Example:
```python3 llamassp.py 30B_8bit 7B_8bit_4GPUs```
will run speculative sampling using the second model name as draft model 

Results of the `13B_8bit / 7B_8bit_4GPUs` : 330ms/token

=> No speedup, even a speed down, which is expected since the target model is already pretty fast.

Results of the `30B_8bit / 7B_8bit_4GPUs` : 370ms/token

=>  a bit less than 10% improvement over the 30B_8bit model

## Distribution recovery
In addition to measuring the speed improvement, it is necessary to check speculative sampling samples with distribution similar to original sampling, in other words that the SSP generations are as good as the regular ones.

An intuitive check can be done by looking at the completions : those of the regularly sampled 13B model (8bit) seem to be of the same quality level than those of SSp 13B/7B, while seeming different in quality from the completions of other regularly sampled models.

A similar observation can be made for the regularly sampled 30B 8bit model and the SSp 30B/7B.

Note that since there is inherent randomness in the acceptation of tokens from the draft model, the completion should not expected to be exactly the same. 

To confirm the validity of the experiments, it is necessary to find a more precise metric of the fact that the completions are of the same quality. TODO. 

