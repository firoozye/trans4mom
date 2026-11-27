#!/bin/bash
# SGE submission script for UCL CS HPC Cluster ('wise') using qsub
# Wood et al. (2022) Exact Multi-Seed Training (Seeds 1-5)

#$ -N wood_train
#$ -o logs/train_$JOB_ID_$TASK_ID.out
#$ -e logs/train_$JOB_ID_$TASK_ID.err
#$ -l h_rt=48:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l h='!gonzo*'
#$ -R y
#$ -t 1-5
#$ -S /bin/bash
#$ -wd /SAN/intelsys/ABO_Research/dev/trans4mom

# 1. Activate Environment
if [ -f /home/firoozye/mome_cluster_env/bin/activate ]; then
    source /home/firoozye/mome_cluster_env/bin/activate
elif [ -f /home/firoozye/anaconda3/bin/activate ]; then
    source /home/firoozye/anaconda3/bin/activate mome_env
fi

mkdir -p logs weights

# Parse optional CPD flag passed via qsub -v CPD=1
EXTRA_ARGS=""
CONFIG="configs/baseline_wood.yaml"
PREFIX="wood_pure_best"

if [ "$CPD" == "1" ]; then
    EXTRA_ARGS="--include_cpd"
    CONFIG="configs/cpd_wood.yaml"
    PREFIX="wood_cpd_best"
fi

echo "=== Node: $(hostname) | Task (Seed): $SGE_TASK_ID | Config: $CONFIG | Prefix: $PREFIX ==="

python -u training/train.py \
    --config $CONFIG \
    --prefix $PREFIX \
    --seed $SGE_TASK_ID \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 80 \
    --patience 25 \
    $EXTRA_ARGS
