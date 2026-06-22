#!/bin/bash
# Run PIGNN training in torch310 conda environment with GPU
conda run -n torch310 python main.py --device cuda "$@"
