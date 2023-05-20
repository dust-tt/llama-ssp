#!/bin/bash
set -e

# install everything for llama-ssp to run

# create virtual environment named llama-ssp
python3 -m venv venv
source venv/bin/activate

# install python packages
pip3 install -r requirements.txt

# run llama-ssp if needed
# python3 llamassp.py
