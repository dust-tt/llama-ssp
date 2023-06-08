# Generate tokens 1.5x to 3x faster 

With Speculative Sampling (SSp), a large language model can [generate tokens quite faster](https://blog.dust.tt/2023-06-02-speculative-sampling) using a smaller model as help.

This repo shows how it's done and measures timing improvements using Llama models. Here's the idea:

![](https://github.com/philipperolet/llama-ssp/blob/main/example.gif)

Above, you see a 30B llama model generating tokens (on an 8-GPU A100 machine), then you see the same model going ~50% to 100% faster (i.e. in 33% to 50% less time) using speculative sampling -- with the same completion quality. Go to [Try it yourself](#try-it-yourself) to try it yourself :)

This repo implements an algorithm published in [this paper](https://arxiv.org/pdf/2302.01318.pdf) whose authors are warmly thanked for their work. For non-academic folks, we attempt a shortened, didactic explanation of how it works in our [our blog post](https://blog.dust.tt/2023-06-02-speculative-sampling)

In the following, the large model we try to speed up is called the `target model`, and the smaller model that helps sampling is called the `draft model`. We compare regular sampling (RSp) of the target model with speculative sampling (SSp) of the target model with the draft model.

Side-Note: this repo is an example of the kind of work we do at [Dust](https://dust.tt) ✨

## Benefits and Caveats of SSp
### Benefits
+ Almost identical memory footprint
+ Same completion quality
+ 1.5 to 3 times faster token generation
+ Relatively simple code

### Caveats
- useful mostly for live token generation, less useful in batch settings (and irrelevant to model training)
- works best in settings where there are often "easily guessable" tokens. This is very often the case, especially for written language, e.g.:
  - there is often a dot at the end of a sentence, a capitalized letter next, determinants or common words like "the, a, it..." 
  - given the way tokenization is done too; Eg token "amaz" => when the beginning is "It was amaz", even tiny models will guess the next token is "ing" (and the following one probably ! or :))
- draft model & target model must have the same vocabulary size
- draft model should be at least 2-3 faster than target model

#### Is it really the same quality? Speculative sampling returns different outputs than regular sampling
They return different outputs because speculative sampling has inherent randomness. However, it is demonstrated in [the paper on which this repo is based](https://arxiv.org/pdf/2302.01318.pdf)  that using speculative sampling provides the same output probability distributions than regular sampling.

The paper also performs experiments to show this on typical benchmarks. In this repo, [experiments on completing multiplications](#distribution-recovery) provide further empirical evidence that SSp completion and RSp completion are of similar quality.

## Try it yourself

### Setup
Get access to a machine with 4 or more GPUs, then:

```
ssh [fat_machine_with_gpus]
git clone https://github.com/philipperolet/llama-ssp.git
cd llama-ssp
chmod a+x machine-install.sh
sudo ./machine-install.sh  # global setup
. setup.sh  # project setup(virtual env, requirements)
python llamassp.py compare 30B 7B  # compare regular & ssp as in the gif
```

Get the cli details with
```
python llamassp.py -h
```

Note: the above was tested on an Ubuntu 22.04 OS.

### Run raw timing measurements
To run experiments measuring average model latency in ms/token:
```python3 llamassp.py latency TARGET_MODEL_NAME [--draft DRAFT_MODEL_NAME]```

Results of such measurements are below. 

TARGET_MODEL_NAME correspond to various flavors of Llama models (7B to 30B), with or without quantization. Possible values are `7B, 13B, 30B, 7B_8bit, 13B_8bit, 30B, 30B_8bit, 65B, 65B_8bt`. These are the models published on HuggingFace by `decapoda-research`. 

Using 65B versions, however, requires providing the weights yourself. Do so as explained below in "Use of custom model" 

The full model configs are defined as `model_params` in `llamassp.py` and can be completed/changed as required -- make sure that there is enough memory for the model to run. Models run in parallel in all the GPUs available.

If DRAFT_MODEL_NAME is not specified, the target model is run with regular sampling on a few examples, and model latency is measured.

If DRAFT_MODEL_NAME is specified, speculative sampling latency is measured.

#### Use of custom model
To use a custom model, add it in `model_params`. You can provide a path to HF weights as `model_name`. Find weights on the internet, then convert them to HF weights following [these instructions](https://huggingface.co/docs/transformers/main/en/model_doc/llama).

This step is required if you want to use a 65B model. This is because the 65B model provided by decapoda-research performed very bad completions through the code of this repo at the time of the writing on the settings. Using 65B weights provided by another source then converting them to HF weights was tested and works fine.

!! Any huggingface model can be tested (not only Llama), however speculative sampling requires the draft and the target models to have the same tokenizer.

#### Memory requirements
To run these experiments, the sum of available memory should be about 4 times the model size for a regular model, and 2 times the model size for a quantized model

E.g. to test the 30B model => you need 120GB GPU memory. To test the 7B_8bit model => you can do it with 14GB so a single GPU might be enough.
To test speculative sampling of a 30B model with a 7B draft, you need 148GB GPU memory (4*37). 

#### Specific GPU config
You may stumble on this kind of error: `Error 802: system not yet initialized`. On some machines--e.g. p4d.24xlarge that was used for these experiments--additional setup might be required:
```
sudo apt-get install cuda-drivers-fabricmanager
sudo systemctl start nvidia-fabricmanager
```

### Run a comparison of regular Vs speculative sampling
To try exactly what's in the gif with various target models and draft models
```python3 llamassp.py compare TARGET_MODEL_NAME DRAFT_MODEL_NAME```

### Run evals
Evals measure how well models (regular or speculative sampling) perform on a simple multiplication benchmark:
```
python llamassp.py eval [--seed SEED] [--nb-prompts NB_PROMPTS] TARGET_MODEL DRAFT_MODEL
```
The command will assess the model by making it perform NB_PROMPTS multiplications and return the ratio of success. The multiplications are randomly drawn with numbers between 1 and 100, with a seed for repeatability.

## Measurements
The main result is the Llama 30B and 65B speed improvements using SSp.

Using speculative sampling with a 7B draft model provides a  ~80% speed improvement with same quality of completions over the regular sampling of the 30B model, and a ~125% speed improvement for the 65B model.

### Measured sampling speed of various models

On a p4d.24xlarge (8 A100 40GB gpus):

|Model_type | Ms/token|
|---|---|
|7B_8bit|215ms|
|7B|70ms|
|7B_4GPUs|70ms|
|13B|125ms|
|30B|330ms|
|65B|610ms|

Comparison SSp / RSp (regular sampling):

|Model_type | Ms/token| Speed Improvement|
|---|---|---|
|13B|125ms|-|
|SSp 13B/7B|**114ms**|**1.1x**|
|30B|330ms|-|
|SSp 30B/7B|**180ms**|**1.8x**|
|65B|610ms|-|
|SSp 65B/7B |**270ms**|**2.25x**|

### Notes on the measures
The timings perform completions on 15 examples (+ 1 warmup not shown), and finally output the model generation latency in ms/token (so lower is better). The measurements are on relatively small prompts (~ 1 sentence) and small completions (64 tokens); longer prompts / completion would of course decrease the speed.

The per-token timings may seem slow. There are two reasons for this: 1/ we are not batching, many models can generate a lot faster in batch settings 2/ we did not optimize for per-model speed.  This was not needed since the general point to make the case for speculative sampling is that if you have a fast model and a slow one, you can get the slow one to be a lot faster for (almost) free.

### Older measurements
Experiments with quantized models on a g5.12xlarge AWS instance:

|Model_type | Ms/token|
|---|---|
|7B_4GPUs |  86ms|
|7B_8bit | 210ms|
|7B_8bit_4GPUs |  216ms|
|13B_8bit |  270ms|
|30B_8bit |  405ms|
|65B_8bit |  530ms|

Comparison SSp / RSp:

|Model_type | Ms/token| Speed Improvement|
|---|---|---|
|13B_8bit| 270ms|-|
|SSp 13B/7B_8bit| 330ms| -
|30B_8bit|405ms|
|SSp 30B/7B_8bit|370ms|

## Distribution recovery
In addition to measuring the speed improvement, it is necessary to check speculative sampling samples with the same distribution than regular sampling, in other words that the SSp generations are as good as the regular ones.

There is inherent randomness in the acceptation of tokens from the draft model; therefore the completions from SSp and RSp naturally differ and we cannot expect the outputs to be exactly the same.

The paper cited at the beginning provides theoretical proof that the output token distributions are the same, as well as evaluations on common benchmarks. With this repo, an intuitive check can be done by looking at the completions : those of the regularly sampled 30B model seem to be of the same quality level than those of SSp 30B/7B.

In order to further show that SSp provides same quality results as the target model with the code of this repo, we perform evaluations by prompting models for random multiplications of numbers between 1 and 99. Each model is assessed by its percentage of successes, in regular sampling (RSp) and SSp mode with a 7B draft model. Results are displayed below with 95% confidence bounds (Wald).

|Model_type | RSp perf| SSp perf|
|---|---|---|
|7B| 28.4% (+- 1.4%)| 28.1% (+- 1.4%)|
|13B| 42.9% (+-1.5%) | 43.1 (+- 1.5%)|
|30B| 50.1% (+- 1.5%) | 49.6% (+- 1.5%) |
|65B| TBD | TBD |

The results show that SSp and RSp perform the same with high confidence. The script for these experiments can be found [here](xp.sh).
