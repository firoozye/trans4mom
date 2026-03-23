import os
import argparse
import torch
import pandas as pd
import numpy as np
from engine.trainer import run_training_job
from data.processor import FeatureProcessor

def main():
    parser = argparse.ArgumentParser(description="HPC Training Script for Momentum Transformer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--trans_cost", type=float, default=0.001)
    parser.add_argument("--data_path", type=str, default="data/processed/crypto_data.parquet")
    args = parser.parse_args()

    # 1. Load Data
    # Expecting a multi-asset dataframe with a hierarchical index or multiple asset columns
    if not os.path.exists(args.data_path):
        print(f"Data not found at {args.data_path}. Generating toy data for script verification...")
        # Placeholder for real data loading
        num_assets = 10
        time_steps = 1000
        x_toy = torch.randn(100, 64, 5, 1) # (batches, time, vars, in_dim)
        y_toy = torch.randn(100, 64, num_assets)
        num_assets = 10
    else:
        # Load your real parquet/csv here
        # df = pd.read_parquet(args.data_path)
        # process into tensors ...
        pass

    # 2. Setup Hyperparameters
    hparams = {
        'input_dim': 1,
        'num_vars': 5, # MACD [10, 21, 63, 126, 252]
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'num_assets': num_assets if 'num_assets' in locals() else 1,
        'lr': args.lr,
        'trans_cost': args.trans_cost,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }

    # 3. Launch Training
    print(f"Launching HPC Training with {hparams}")
    run_training_job(x_toy, y_toy, hparams=hparams)

if __name__ == "__main__":
    main()
