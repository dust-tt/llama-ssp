# TODO
+ change llama-ssp to handle ssp evals
  + in measure_model_score, call to ssp rather than sample_model when draft given
  + in llama_ssp, allow call with draft model
  + in xp script, allow ssp evals

+ cleaner output: Nb of multiplications, nb of successes, ratio, confidence intervals
  + Output confidence intervals

- Get valid evals -- with confidence intervals -- for 7 to 65B and 13 to 65B ssp
  - split in 2: 7 & 13B, and 13B ssp on the xpg5 machine
  - 30 & 65B / regular and ssp / on the xpfat machine
- Write the results in the readme
- Comm with the link to the results

## Done
+ ability to make small runs that pass as integration tests
  + change cli code to accept an eval size
  + change xp.sh to have a --test arg

## Next
- readme with the new cli
- fabricmanager handled
- import base in sample model


+ repeated sequential prompting
+ 8bit issue with temperature sampling
+ change evals to multiplications
+ add script to run regular eval xps
## Leg
- Additions benchmark; few shot prompt with additions from 1 to 300
  - measure bench results on all models + ssp models
- More "releasable" version (same)
  - clean readme recipe to reproduce everything
- (opt) Use meta code to do the same

## Readme
- make the ideal readme structure
- correct the timings for Ssp improvements
- explain 65B does not perform well
- caveats


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





