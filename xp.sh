#!/bin/bash
set -e

# run llamassp xps
source venv/bin/activate


# for model names in the list, run llamassp.py
for model_name in "$@"; do
    python3 llamassp.py -v eval  --seed 0 $model_name 2>&1 | tee -a eval_$model_name.log
    python3 llamassp.py -v eval  --seed 1 $model_name 2>&1 | tee -a eval_$model_name.log
    python3 llamassp.py -v eval  --seed 2 $model_name 2>&1 | tee -a eval_$model_name.log
done