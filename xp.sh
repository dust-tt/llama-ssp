#!/bin/bash
set -e

# run llamassp xps
source venv/bin/activate

# Test mode: run only 10 prompts (2 * 5)
if [ "$1" == "--test" ]; then
    nb_prompts=5
    shift
else
    nb_prompts=250
fi

# for model names in the list, run llamassp.py
for model_name in "$@"; do
    python3 llamassp.py -v eval --seed 0 --nb-prompts $nb_prompts $model_name 2>&1 | tee -a eval_$model_name.log
    python3 llamassp.py -v eval --seed 1 --nb-prompts $nb_prompts $model_name 2>&1 | tee -a eval_$model_name.log
done