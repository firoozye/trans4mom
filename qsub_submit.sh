#!/bin/bash
# SGE submission script for UCL CS HPC Cluster (using qsub)

#$ -N mom_trans_fast
#$ -o logs/train_$JOB_ID.out
#$ -e logs/train_$JOB_ID.err
#$ -l h_rt=24:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -S /bin/bash
#$ -wd /home/firoozye/dev/trans4mom

# 1. Activate Cluster Environment
source /home/firoozye/mome_cluster_env/bin/activate

# 2. Setup Logging
mkdir -p logs
mkdir -p weights

# 3. Run Training (Unbuffered output for real-time logs)
PYTHONPATH=. python -u train_hpc.py --config config.yaml

