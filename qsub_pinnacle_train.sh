#!/bin/bash
# SGE submission script for UCL CS HPC Cluster ('wise') using qsub
# Replicating Wood, Roberts & Zohren (2022) with Pinnacle CLC Data

#$ -N pinnacle_mom_trans
#$ -o logs/pinnacle_$JOB_ID_$TASK_ID.out
#$ -e logs/pinnacle_$JOB_ID_$TASK_ID.err
#$ -l h_rt=48:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -t 1-5
#$ -S /bin/bash
#$ -cwd

# 1. Activate Cluster Environment (supports both cluster virtualenv and conda mome_env)
if [ -f /home/firoozye/mome_cluster_env/bin/activate ]; then
    source /home/firoozye/mome_cluster_env/bin/activate
elif [ -f /home/firoozye/anaconda3/bin/activate ]; then
    source /home/firoozye/anaconda3/bin/activate mome_env
elif [ -f /cs/research/fca/home0/firoozye/mome_cluster_env/bin/activate ]; then
    source /cs/research/fca/home0/firoozye/mome_cluster_env/bin/activate
fi

# Ensure output logs and weights directories exist
mkdir -p logs weights

# Prevent CUDA memory fragmentation on shared GPU nodes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Run Training on Pinnacle CLC dataset (Seed = Task ID)
PYTHONPATH=. python -u train_hpc.py \
    --config config_pinnacle.yaml \
    --prefix pinnacle \
    --seed $SGE_TASK_ID
