#!/bin/bash
#
#SBATCH --job-name=train_model
#SBATCH --partition=hub
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH --output=train_model-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=USER@scu.edu

module purge
module load CUDA/12.6.2
module load Python/3.12.3-GCCcore-14.2.0

set -e

# Step 1: Ensure the virtual environment exists
if [ ! -d "env" ]; then
    echo "Virtual environment not found. Running setup_env.sh..."
    bash setup_env.sh
fi

# Step 2: Activate virtual environment
echo "Activating virtual environment..."
source env/bin/activate

echo "Using Python from: $(which python)"

# Step 3: Start the training script
echo "Starting training model5..."
python train_model5.py

echo "Training complete!"