#!/bin/bash

# Slurm submission script for UCL Myriad Cluster
# Track B: HPC Mode - Momentum Transformer Training

#SBATCH --job-name=mom_trans_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# 1. Load Modules (UCL Myriad defaults - adjust if necessary)
module purge
module load python/3.10
module load cuda/11.8

# 2. Activate Environment
# Use the absolute path to your anaconda3/bin/conda
source /home/firoozye/anaconda3/bin/activate mome_env

# 3. Setup Logging
mkdir -p logs
mkdir -p weights

# 4. Run Training
# We use train_hpc.py with --config config.yaml
python train_hpc.py --config config.yaml

