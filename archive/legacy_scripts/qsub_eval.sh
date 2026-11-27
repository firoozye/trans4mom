#!/bin/bash
#$ -N eval_wood
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

echo "=== RUNNING BASELINE WOOD ET AL. EVALUATION ON GPU ==="
python -u research/evaluate_wood_pure.py --prefix wood_pure_best

echo ""
echo "=== RUNNING WOOD ET AL. + GP-CPD EVALUATION ON GPU ==="
python -u research/evaluate_wood_pure.py --prefix wood_cpd_best --include_cpd
