#!/bin/bash
# SGE submission script for UCL CS HPC Cluster (using qsub)

#$ -N ensemble_mom_trans
#$ -o logs/ensemble_$JOB_ID_$TASK_ID.out
#$ -e logs/ensemble_$JOB_ID_$TASK_ID.err
#$ -l h_rt=48:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -t 1-5
#$ -S /bin/bash
#$ -wd /home/firoozye/dev/trans4mom

# 1. Activate Cluster Environment
source /home/firoozye/mome_cluster_env/bin/activate

# 2. Run Training (Seed = Task ID)
PYTHONPATH=. python -u train_hpc.py \
    --config config_1990.yaml \
    --seed $SGE_TASK_ID

