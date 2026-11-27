# Momentum Transformer Replication: Wood et al. (2022)

Official PyTorch replication of *Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture* (Wood, Giegerich, Roberts, and Zohren, 2022; arXiv:2112.08534v3 / Journal of Financial Data Science).

## Directory Structure

*   `configs/`: Model and training configuration files.
    *   `configs/baseline_wood.yaml`: Canonical 8-feature baseline configuration.
    *   `configs/cpd_wood.yaml`: 10-feature Gaussian Process Changepoint Detection (GP-CPD) configuration.
*   `models/`: Core neural architecture components.
    *   `models/momentum_transformer.py`: Canonical `WoodMomentumTransformer` (380,462 params baseline, 425,016 params CPD).
    *   `models/layers.py`: Single-Bias LSTM, GateAddNorm, Gated Residual Network, and Masked Interpretable MHA.
*   `training/`: Unified training and evaluation scripts.
    *   `training/train.py`: Multi-seed training engine with early stopping and turnover penalty.
    *   `training/evaluate.py`: Multi-asset portfolio out-of-sample benchmark calculator across 1990–2026.
*   `cluster/`: UCL Computer Science SGE HPC cluster execution scripts (`qsub`).
    *   `cluster/train_array.sh`: Multi-seed job array submission script (Seeds 1–5).
    *   `cluster/evaluate_gpu.sh`: GPU out-of-sample evaluation script.
*   `data/`: Data ingestion, processing, and PyTorch dataset loaders.
    *   `data/datasets.py`: `WoodNonOverlappingDataset` adhering to Exhibit 15 non-overlapping 252-day sequences.
*   `engine/`: Core algorithmic components.
    *   `engine/loss.py`: Net Sharpe ratio loss with turnover penalty ($C = 1\text{ bp}$).
    *   `engine/cpd.py`: Online Bayesian Gaussian Process Changepoint Detector (Matérn 3/2 kernel).
*   `reports/`: Comprehensive audits, architectural breakdowns, and PDF reports.
    *   `reports/wood_2022_replication_assessment_and_empirical_comparison.md`: Point-by-point ground truth audit and empirical comparison.
*   `tests/`: Unit test suite.
    *   `tests/test_replication.py`: Parameter verification, forward/backward passes, and universe validation.
*   `weights/`: Saved model checkpoint weights.
*   `archive/`: Quarantined legacy models, experimental scripts, and audit logs.

## Quickstart

### 1. Verification Test
```bash
python tests/test_replication.py
```

### 2. Model Training
To train the canonical 8-feature baseline:
```bash
python training/train.py --config configs/baseline_wood.yaml --prefix wood_pure_best --seed 1
```

To train the 10-feature GP-CPD model:
```bash
python training/train.py --config configs/cpd_wood.yaml --prefix wood_cpd_best --include_cpd --seed 1
```

### 3. Cluster Submission (UCL HPC 'wise')
```bash
qsub cluster/train_array.sh           # Baseline ensemble (Seeds 1-5)
qsub -v CPD=1 cluster/train_array.sh  # GP-CPD ensemble (Seeds 1-5)
qsub cluster/evaluate_gpu.sh          # Full GPU evaluation of both ensembles
```

### 4. Out-of-Sample Evaluation
```bash
python training/evaluate.py --prefix wood_pure_best
python training/evaluate.py --prefix wood_cpd_best --include_cpd
```
