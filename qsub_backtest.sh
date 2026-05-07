#!/bin/bash
#$ -N backtest_v2
#$ -o logs/backtest_$JOB_ID.out
#$ -e logs/backtest_$JOB_ID.err
#$ -l h_rt=01:00:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -S /bin/bash
#$ -wd /home/firoozye/dev/trans4mom

source /home/firoozye/mome_cluster_env/bin/activate
PYTHONPATH=. python research/backtest_upgraded_v2.py
