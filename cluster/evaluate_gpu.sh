#!/bin/bash
# SGE submission script for GPU Evaluation of Wood et al. Ensembles

#$ -N wood_eval
#$ -o logs/eval_wood_$JOB_ID.out
#$ -e logs/eval_wood_$JOB_ID.err
#$ -l h_rt=01:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -l h='!gonzo*'
#$ -R y
#$ -S /bin/bash
#$ -wd /SAN/intelsys/ABO_Research/dev/trans4mom

source /home/firoozye/mome_cluster_env/bin/activate
mkdir -p logs

echo "================================================================="
echo "=== EVALUATING WOOD ET AL. (2022) BASELINE ENSEMBLE (5 SEEDS) ==="
echo "================================================================="
python -u training/evaluate.py --prefix wood_pure_best

echo ""
echo "================================================================="
echo "=== EVALUATING WOOD ET AL. (2022) + GP-CPD ENSEMBLE (5 SEEDS) ==="
echo "================================================================="
python -u training/evaluate.py --prefix wood_cpd_best --include_cpd
