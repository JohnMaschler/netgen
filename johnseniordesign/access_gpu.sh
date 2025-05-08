#!/bin/bash

# Request GPU resources and start an interactive session
# if it is taking too long to get resoruces, run the command below instead:
# srun --partition=hub --cpus-per-task=8 --mem=24G --nodes=1 --pty /bin/bash
echo "Requesting GPU resources..."
srun --partition=gpu --cpus-per-task=8 --mem=24G --nodes=1 --gres=gpu:1 --pty /bin/bash

# Load the NVHPC module
echo "Loading NVHPC module..."
module load NVHPC