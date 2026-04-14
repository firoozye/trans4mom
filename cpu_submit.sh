#!/bin/bash
# SGE submission script for UCL CS HPC Cluster (using qsub)

#$ -N mom_trans_cpu
#$ -o logs/train.out
#$ -e logs/train.err
#$ -l h_rt=4:00:00
#$ -l mem_free=16G

#$ -q cpu.q
#$ -wd /home/firoozye/dev/trans4mom

# 1. Load Modules
module purge
module load python/3.10
module load cuda/11.8

# 2. Activate Cluster Environment
# We use a standard virtualenv on the cluster for compatibility
source /home/firoozye/mome_cluster_env/bin/activate

# 3. Setup Logging
mkdir -p logs
mkdir -p weights

# 4. Run Training
python train_hpc.py --config config.yaml

