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
|13B_8bit |  265ms|
|30B_8bit |  401ms|
|65B_8bit |  528ms|


## Speculative Sampling speed
Example:
```python3 llamassp.py 30B_8bit 7B_8bit_4GPUs```
Will run speculative sampling using the second model name as draft model

Results of the `13B_8bit / 7B_8bit_4GPUs` : 285ms/token -- recovers the actual sample distribution

Results of the `30B_8bit / 7B_8bit_4GPUs` : 510ms/token -- shaky: sometimes recovers the actual distribution, sometimes not

