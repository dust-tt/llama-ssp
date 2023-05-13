# Generate tokens 1.5x to 3x faster : Experiments on Speculative sampling


Using Speculative Sampling (SSp), a large language model can generate tokens quite faster using a smaller model as help. This repo shows how it's done using Llama models. Here's the idea:

[TODO: THE GIF]

Above, you see a 30B llama model generating tokens (on an 8-GPU A100 machine), then you see the same model going 70% faster (i.e. in 40% less time) using speculative sampling -- with the same completion quality. Go to [Try it yourself](#try-it-yourself) to try it yourself :)


This repo is based on [this paper](https://arxiv.org/pdf/2302.01318.pdf) whose authors are warmly thanked for their work.

## Benefits and Caveats of Ssp

### Benefits
+ Almost identical memory footprint (since the draft model is lot smaller than the target model)
+ Same completion quality (not "a little lower quality but don't worry you can't see it", same quality)
+ 1.5 to 3 times faster token generation
+ Relatively simple code

#### Is it really the same quality? Speculative sampling returns different outputs than regular sampling
They return different outputs because speculative sampling has inherent randomness. However, it is demonstrated in [this paper](https://arxiv.org/pdf/2302.01318.pdf) that using speculative sampling provides the same output probability distributions than regular sampling.

The paper also performs experiments to show this on typical benchmarks.

### Caveats
- useful mostly for live token generation, less useful in batch settings -- and not related to model training
- works best in settings where there are often "easily guessable" tokens. This is very often the case, especially for written language
    - there is often a dot at the end of a sentence, a capitalized letter next, determinants or common words like "the, a, it..." 
	- and given the way tokenization is done too. Eg token "amaz" => when the beginning is "It was amaz", even tiny models will guess the next token is "ing" (and the following one probably ! or :))

### Requirements
- draft model & target model must have the same vocabulary size
- draft model should be at least twice faster than target model (small speedup)
## Try it yourself

### Setup
Get access to a machine with 4 or more GPUs, then:

```
ssh [fat_machine_with_gpus]
git clone https://github.com/philipperolet/llama-ssp.git
cd llama-ssp
./machine-install.sh  # global setup
./setup.sh  # project setup(virtual env, requirements)
python3 llamassp.py compare 30B 7B  # compare regular & ssp as in the gif
```

Note: the above was tested on an Ubuntu 22.04 OS.

### Run a comparison of regular Vs speculative sampling
To try exactly what's in the gif
```python3 llamassp.py compare TARGET_MODEL_NAME DRAFT_MODEL_NAME```

### Run raw timing measurements
```python3 llamassp.py TARGET_MODEL_NAME [DRAFT_MODEL_NAME]```
TARGET_MODEL_NAME correspond to various flavors of Llama models (7B to 30B), with or without quantization. Possible values are `7B, 13B, 30B, 7B_8bit, 13B_8bit, 30B, 30B_8bit`

The full model configs are defined as `model_params` in `llamassp.py` and can be completed/changed as required -- ensuiring that there is enough memory for the model to run.

Models run on the number of GPUs specified in `model_params`, e.g. `7B_8bit` runs on a single GPU, and a specific version `7B_8bit_4GPUs` runs on 4 gpus.


If DRAFT_MODEL_NAME is not specified, the target model is run with regular sampling on a few examples, and model latency is measured.

If DRAFT_MODEL_NAME is specified, speculative sampling latency is measured.

#### Memory requirements
To run these experiments, the sum of available memory should be about 4 times the model size for a regular model, and 2 times the model size for a quantized model

E.g. to test the 30B model => you need 120GB GPU memory. To test the 7B_8bit model => you can do it with 14GB so a single GPU might be enough.
To test speculative sampling of a 30B model with a 7B draft, you need 148GB GPU memory (4*37). 

#### Specific GPU config
If you stumble on this kind of error: `Error 802: system not yet initialized`
On some machines -- e.g. p4d.24xlarge that was used for these experiments -- additional setup might be required 
```
sudo apt-get install cuda-drivers-fabricmanager
sudo systemctl start nvidia-fabricmanager
```


## Experiments & results
The main result that you can reproduce easily using this repo is with the Llama 30B model
Using speculative sampling with a 7B draft model provides a  50% to 80% speed improvement with same quality of completions over the regular sampling of the 30B model.

### Notes on the experiments
- the per-token timings may seem slow. There are two reasons for this: 1/ we are not batching, many models can generate a lot faster if requests can be batched
    - 

  - possible speeds optimized e.g. //ion not optimal. but the general point: if you have a fast one and a slow one, you can get the slow one to be twice faster


### Regular sampling speed of various models
Measure those like this:
```python3 llamassp.py 13B_8bit```
This will perform completions on 15 examples (+ 1 warmup not shown), and finally output the model generation latency in ms/token (so lower is better).

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

