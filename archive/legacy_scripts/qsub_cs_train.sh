#!/bin/bash
# SGE submission script for Cross-Sectional Training

#$ -N cs_mom_macro
#$ -o logs/cs_train_$JOB_ID.out
#$ -e logs/cs_train_$JOB_ID.err
#$ -l h_rt=12:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -S /bin/bash
#$ -wd /home/firoozye/dev/trans4mom

source /home/firoozye/mome_cluster_env/bin/activate
mkdir -p logs weights

PYTHONPATH=. python -u train_cross_sectional.py --config config.yaml
