#!/bin/bash
# SGE submission script for UCL CS HPC Cluster (Resuming Training)

#$ -N mom_trans_resume
#$ -o logs/resume_$JOB_ID.out
#$ -e logs/resume_$JOB_ID.err
#$ -l h_rt=12:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -S /bin/bash
#$ -wd /home/firoozye/dev/trans4mom

# 1. Activate Cluster Environment
source /home/firoozye/mome_cluster_env/bin/activate

# 2. Run Training (Resuming from epoch 251)
PYTHONPATH=. python -u train_hpc.py \
    --config config.yaml \
    --load_weights weights/upgraded_v2_epoch_250.pt \
    --start_epoch 251
