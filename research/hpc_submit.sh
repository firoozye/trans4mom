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
# Replace 'mom_env' with your actual conda environment name
source activate mom_env

# 3. Setup Logging
mkdir -p logs
mkdir -p weights

# 4. Run Training
# We use torch.distributed.run for potential multi-GPU expansion
# For 1 GPU, nproc_per_node=1
python -m torch.distributed.run \
    --nproc_per_node=1 \
    --master_port=29500 \
    train_hpc.py \
    --epochs 100 \
    --batch_size 64 \
    --hidden_dim 64 \
    --lr 0.001 \
    --trans_cost 0.001
