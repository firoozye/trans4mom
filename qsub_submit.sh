#!/bin/bash
# SGE submission script for UCL CS HPC Cluster (using qsub)

#$ -N strict_mom_trans
#$ -o logs/strict_$JOB_ID.out
#$ -e logs/strict_$JOB_ID.err
#$ -l h_rt=48:00:00
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

# 3. Run Training
PYTHONPATH=. python -u train_hpc.py --config config_1990.yaml

