# Llama SSp

Using Speculative Sampling (SSp) to speed up larger Llama models. 

Main result: 60% speed improvement with same quality.

The main result of this experiment is to show that a 30B llama model running with a latency of ~ 260ms per token, can be sped up using Speculative sampling with a to get ~ 160ms per token.


## Setup
Prepare a GPU machine with 4 or more GPUs: ``./machine-install.sh``

Setup the project (virtual env, requirements): ``./setup.sh``

Run the XPs (not on your local machine): 

```python3 llamassp.py MODEL_NAME [DRAFT_NAME]```

If DRAFT_NAME is not specified, models are run with regular sampling on a few examples, and model latency is measured.
If DRAFT_NAME is specified, speculative sampling latency is measured.


### Memory requirements
To run these experiments, the sum of available memory should be about 4 times the model size for a regular model, and 2 times the model size for a quantized model

E.g. to test the 30B model => you need 120GB GPU memory. To test the 7B_8bit model => you can do it with 14GB so a single GPU might be enough.
To test speculative sampling of a 30B model with a 7B draft, you need 148GB GPU memory (4*37). 

### Specific GPU config
If you stumble on this kind of error: `Error 802: system not yet initialized`
On some machines -- e.g. p4d.24xlarge that was used for these experiments -- additional setup might be required 
```
sudo apt-get install cuda-drivers-fabricmanager
sudo systemctl start nvidia-fabricmanager
```

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

On a harder, faster, better, stronger p4d.24xlarge (8 A100 40GB gpus)

|Model_type | Ms/token|
|---|---|
|7B_8bit |  210ms|
|7B|60ms|
|7B_4GPUs|70ms|
|13B|100ms|
|30B|260ms|
|65B|510ms|




## Speculative Sampling speed
Example:
```python3 llamassp.py 30B_8bit 7B_8bit_4GPUs```
will run speculative sampling using the second model name as draft model 

Results of the `13B_8bit / 7B_8bit_4GPUs` : 330ms/token

=> No speedup, even a speed down, which is expected since the target model is already pretty fast.

Results of the `30B_8bit / 7B_8bit_4GPUs` : 370ms/token

=>  a bit less than 10% improvement over the 30B_8bit model

##
## Distribution recovery
In addition to measuring the speed improvement, it is necessary to check speculative sampling samples with distribution similar to original sampling, in other words that the SSP generations are as good as the regular ones.

An intuitive check can be done by looking at the completions : those of the regularly sampled 13B model (8bit) seem to be of the same quality level than those of SSp 13B/7B, while seeming different in quality from the completions of other regularly sampled models.

A similar observation can be made for the regularly sampled 30B 8bit model and the SSp 30B/7B.

Note that since there is inherent randomness in the acceptation of tokens from the draft model, the completion should not expected to be exactly the same. 

To confirm the validity of the experiments, it is necessary to find a more precise metric of the fact that the completions are of the same quality. TODO. 

## Example

### Completing text: The smell of freshly baked bread filled the air as I entered the bakery.
#### Regular 7B
**Completion**: The smell of freshly baked bread filled the air as I entered the bakery. The smell of freshly baked bread filled the air as I entered the bakery. The smell of freshly baked bread filled the air

**Time**: 2.44s

#### Regular 30B 
**Completion**:  The smell of freshly baked bread filled the air as I entered the bakery. The smell of freshly baked bread is one of the best smells in the world. I was greeted by a friendly employee who asked me

**Time**: 8.93s


#### SSp 30B / 7B
**Completion**:  The smell of freshly baked bread filled the air as I entered the bakery. The smell of fresh bread and cakes made me feel hungry. I looked around the bakery. I saw different kinds of bread and cakes.

**Time**: 5.26s

### Completing text: The sound of the train whistle echoed through the valley as I stood at the station, waiting.

#### Regular 7B
**Completion**:  The sound of the train whistle echoed through the valley as I stood at the station, waiting. I had been waiting for this moment for a long time. I had been waiting for this moment for a long time. I had been waiting for this moment for
**Time**: 2.51s

#### Regular 30B
**Completion**:  The sound of the train whistle echoed through the valley as I stood at the station, waiting.
I was waiting for the train to take me to the city.
I was waiting for the train to take me to the city.
I was waiting
**Time**: 9.16s

#### SSp 30B / 7B
**Completion**:  The sound of the train whistle echoed through the valley as I stood at the station, waiting. I was waiting for the train to take me to the city. I could not wait to see him. I had not seen him in a long time. I had not seen
**Time**: 6.23s



30B

30B / 7B

