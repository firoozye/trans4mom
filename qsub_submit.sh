#!/bin/bash
# SGE submission script for UCL CS HPC Cluster (using qsub)

#$ -N mom_trans_train
#$ -o logs/train.out
#$ -e logs/train.err
#$ -l h_rt=4:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=1
#$ -pe gpu 1
#$ -R y
#$ -S /bin/bash
#$ -wd /home/firoozye/dev/trans4mom

# 1. Load Modules (Disabled as environment is self-contained)
# module load python/3.10
# module load cuda/11.8

# 2. Activate Cluster Environment
source /home/firoozye/mome_cluster_env/bin/activate

# 3. Setup Logging
mkdir -p logs
mkdir -p weights

# 4. Run Training (Unbuffered output for real-time logs)
python -u train_hpc.py --config config.yaml

