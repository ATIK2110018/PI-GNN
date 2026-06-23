#!/bin/bash
# Run PIGNN training in torch310 conda environment with GPU
conda run --no-capture-output -n torch310 python main.py --device cuda "$@"
