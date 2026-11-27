#!/bin/bash
# SGE submission script for UCL CS HPC Cluster ('wise') using qsub
# Exact Wood, Roberts & Zohren (2022) Replication: Train <= 2015, Val 2016-2018, OOS 2019+

#$ -N wood_mom_trans
#$ -o logs/pinnacle_wood_$JOB_ID_$TASK_ID.out
#$ -e logs/pinnacle_wood_$JOB_ID_$TASK_ID.err
#$ -l h_rt=48:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l h='!gonzo*'
#$ -R y
#$ -t 1-5
#$ -tc 2
#$ -S /bin/bash
#$ -cwd

# 1. Activate Cluster Environment
if [ -f /home/firoozye/mome_cluster_env/bin/activate ]; then
    source /home/firoozye/mome_cluster_env/bin/activate
elif [ -f /home/firoozye/anaconda3/bin/activate ]; then
    source /home/firoozye/anaconda3/bin/activate mome_env
elif [ -f /cs/research/fca/home0/firoozye/mome_cluster_env/bin/activate ]; then
    source /cs/research/fca/home0/firoozye/mome_cluster_env/bin/activate
elif command -v conda &> /dev/null; then
    conda activate mome_env 2>/dev/null || true
fi

# Ensure output logs and weights directories exist
mkdir -p logs weights

# Clear any allocation flags and stagger startup
unset PYTORCH_CUDA_ALLOC_CONF
sleep $(( (SGE_TASK_ID - 1) * 15 ))

echo "=== SGE GPU Diagnostics ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi 2>/dev/null || true
echo "============================"

# 2. Run Training on Pinnacle CLC dataset with exact Wood et al. splits
PYTHONPATH=. python -u train_hpc.py \
    --config config_pinnacle_wood.yaml \
    --prefix pinnacle_wood \
    --seed $SGE_TASK_ID
