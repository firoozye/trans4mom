#!/bin/bash
#$ -N eval_5seeds
#$ -o logs/eval_pinnacle_5seeds_$JOB_ID.out
#$ -e logs/eval_pinnacle_5seeds_$JOB_ID.err
#$ -l h_rt=01:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l h='!gonzo*'
#$ -R y
#$ -S /bin/bash
#$ -cwd

# 1. Activate environment across cluster venv or conda
if [ -f /home/firoozye/mome_cluster_env/bin/activate ]; then
    source /home/firoozye/mome_cluster_env/bin/activate
elif [ -f /home/firoozye/anaconda3/bin/activate ]; then
    source /home/firoozye/anaconda3/bin/activate mome_env
elif [ -f /cs/research/fca/home0/firoozye/mome_cluster_env/bin/activate ]; then
    source /cs/research/fca/home0/firoozye/mome_cluster_env/bin/activate
elif command -v conda &> /dev/null; then
    conda activate mome_env 2>/dev/null || true
fi

# Ensure output directories exist
mkdir -p logs reports/figures reports/data

# 2. Run Portfolio Evaluation on ALL 5 seeds (1, 2, 3, 4, 5) with GPU
PYTHONPATH=. python -u research/evaluate_pinnacle_portfolio.py --seeds 1,2,3,4,5
