#!/bin/bash
#$ -N ensemble_backtest
#$ -o logs/ensemble_backtest.out
#$ -e logs/ensemble_backtest.err
#$ -l h_rt=02:00:00
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l gpu=true
#$ -pe gpu 1
#$ -R y
#$ -S /bin/bash
#$ -wd /home/firoozye/dev/trans4mom

source /home/firoozye/mome_cluster_env/bin/activate
PYTHONPATH=. python research/backtest_ensemble.py
