# TODO
## Numbers

## Readme
- make the ideal readme structure
- perf of 30B/7B => 160ms/token
- explain 65B does not perform well
- caveats
  - this technique is to speedup autoregressive generation - not training, or not batch inference. Basically, it's suited to live completion use cases
  - the per-token timings may seem slow => because we are not batching
	- mem reqs expl (4* mod size, 2* if quantized)
  - possible speeds optimized e.g. //ion not optimal. but the general point: if you have a fast one and a slow one, you can get the slow one to be twice faster
  - conditions: setups with "kind of obvious" tokens. Very often the case, especially for written language
	- often a space, a dot at the end of a sentence, a capitalized letter
	- and given the way tokenization is done too. Eg token "amaz" => models see "It was amaz" => even tiny model will guess the next token is "ing" (and the following one probably ! or :) )
  - any models as long as they have the same vocabulary

- Experiments to get a good speedup on p4

## Working code for video

## Video
- make the vid script and describe the view

## 
- Cleaner logs
  - Clear completions & timings for each example
  - use logging for debugging output
  - change command line to get output & errors on files too
- Run timings on all models on A100 machine, get all the speeds
- Run SSP timings for various model configurations

- Try sampling with temperature 1 & observing on additions
- Capture in readme all the stuff





