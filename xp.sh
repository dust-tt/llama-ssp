#!/bin/bash
set -e

# run llamassp xps
source venv/bin/activate

# Test mode: run only 10 prompts (2 * 5)
if [ "$1" == "--test" ]; then
    nb_prompts=5
    shift
else
    nb_prompts=1000
fi

# for model names in the list, run llamassp.py
for model_name in "$@"; do
    # if model_name contains a slash, split it into model_name and draft_name
    if [[ $model_name == */* ]]; then
        draft_name=${model_name#*/}
        draft_cmd="--draft $draft_name"
        model_name=${model_name%/*}
    else
        draft_name=""
        draft_cmd=""
    fi
    python3 llamassp.py -v eval --seed 0 --nb-prompts $nb_prompts $model_name $draft_cmd 2>&1 | tee -a eval_$model_name-$draft_name.log
    python3 llamassp.py -v eval --seed 1 --nb-prompts $nb_prompts $model_name $draft_cmd 2>&1 | tee -a eval_$model_name-$draft_name.log
    python3 llamassp.py -v eval --seed 2 --nb-prompts $nb_prompts $model_name $draft_cmd 2>&1 | tee -a eval_$model_name-$draft_name.log
    python3 llamassp.py -v eval --seed 3 --nb-prompts $nb_prompts $model_name $draft_cmd 2>&1 | tee -a eval_$model_name-$draft_name.log
done