#!/bin/bash
# SGE script to rebuild pinnacle_futures_1990_2026.parquet from cleaned RAD files
#$ -N ingest_pinnacle
#$ -o logs/ingest_$JOB_ID.out
#$ -e logs/ingest_$JOB_ID.err
#$ -l h_rt=00:30:00
#$ -l tmem=16G
#$ -l h_vmem=16G
#$ -R y
#$ -S /bin/bash
#$ -cwd

if [ -f /home/firoozye/mome_cluster_env/bin/activate ]; then
    source /home/firoozye/mome_cluster_env/bin/activate
elif [ -f /home/firoozye/anaconda3/bin/activate ]; then
    source /home/firoozye/anaconda3/bin/activate mome_env
elif command -v conda &> /dev/null; then
    conda activate mome_env 2>/dev/null || true
fi

mkdir -p logs data/processed

echo "Starting ingestion on $(hostname) at $(date)"
PYTHONPATH=. python -u data/ingest_pinnacle.py
echo "Ingestion finished at $(date)"
