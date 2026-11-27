#!/bin/bash
# Dedicated SGE submission script for Seed 5 restart (with higher patience)
#$ -N pinnacle_seed_5
#$ -o logs/pinnacle_seed_5_$JOB_ID.out
#$ -e logs/pinnacle_seed_5_$JOB_ID.err
#$ -l h_rt=48:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -S /bin/bash
#$ -cwd

# 1. Activate Cluster Environment
if [ -f /home/firoozye/mome_cluster_env/bin/activate ]; then
    source /home/firoozye/mome_cluster_env/bin/activate
elif [ -f /home/firoozye/anaconda3/bin/activate ]; then
    source /home/firoozye/anaconda3/bin/activate mome_env
elif [ -f /cs/research/fca/home0/firoozye/mome_cluster_env/bin/activate ]; then
    source /cs/research/fca/home0/firoozye/mome_cluster_env/bin/activate
elif command -v conda &> /dev/null; then
    conda activate mome_env 2>/dev/null || true
fi

mkdir -p logs weights reports/figures reports/data
unset PYTORCH_CUDA_ALLOC_CONF

echo "Restarting Seed 5 training with patience=30 on host: $(hostname)"
echo "Using GPU Card: $CUDA_VISIBLE_DEVICES"

PYTHONPATH=. python -u train_hpc.py \
    --config config_pinnacle.yaml \
    --prefix pinnacle \
    --seed 5
