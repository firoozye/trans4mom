#!/bin/bash
# SGE submission script for UCL CS HPC Cluster ('wise') using qsub
# 100% Faithful Reproduction of Wood, Roberts & Zohren (2022)
# Decoder-Only TFT: Non-overlapping 252-day sequences, 8 dedicated GRNs, direct position regression (NO prior).

#$ -N wood_pure
#$ -o logs/wood_pure_$JOB_ID_$TASK_ID.out
#$ -e logs/wood_pure_$JOB_ID_$TASK_ID.err
#$ -l h_rt=48:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -t 1-5
#$ -S /bin/bash
#$ -wd /SAN/intelsys/ABO_Research/dev/trans4mom

# 1. Activate Cluster Environment
if [ -f /home/firoozye/mome_cluster_env/bin/activate ]; then
    source /home/firoozye/mome_cluster_env/bin/activate
elif [ -f /home/firoozye/anaconda3/bin/activate ]; then
    source /home/firoozye/anaconda3/bin/activate mome_env
elif [ -f /cs/research/fca/home0/firoozye/mome_cluster_env/bin/activate ]; then
    source /cs/research/fca/home0/firoozye/mome_cluster_env/bin/activate
fi

# Ensure output logs and weights directories exist
mkdir -p logs weights

echo "=== Host Diagnostics ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "Task ID (Seed): $SGE_TASK_ID"
echo "========================="

# 2. Run Pure Replication Training
PYTHONPATH=. python -u train_wood_pure.py \
    --config config_pinnacle_wood.yaml \
    --prefix wood_pure_best \
    --seed $SGE_TASK_ID \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 80 \
    --patience 25
