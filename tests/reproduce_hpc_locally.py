import torch
import pandas as pd
import numpy as np
from train_hpc import prepare_hpc_data, load_config, get_annualization_factor
from engine.trainer import run_training_job

def test_local_reproduction():
    print("--- Starting Local HPC Reproduction Test ---")
    
    # 1. Load the real config
    config = load_config('config.yaml')
    data_path = 'data/processed/futures_1990_present.parquet'
    
    # 2. Load a tiny subset of data (first 5000 rows to catch broadcasting/logic errors)
    df = pd.read_parquet(data_path).iloc[:5000]
    print(f"Loaded {len(df)} rows for testing.")
    
    # 3. Setup features exactly like the pipeline
    feat_cfg = config['features']
    features = [f'ret_{k}' for k in feat_cfg['return_windows']] + \
               [f'macd_{S}_{L}' for S, L in feat_cfg['macd_pairs']]
    
    # 4. Prepare data using the same Pooled Strategy
    print("Preparing data...")
    x, y, s, aid = prepare_hpc_data(df, features, config, seq_len=64)
    print(f"Prepared {x.shape[0]} sequences. X shape: {x.shape}, Y shape: {y.shape}")
    
    # 5. Force CPU for this test
    hparams = {
        'input_dim': 1,
        'num_vars': len(features),
        'hidden_dim': 64,
        'num_heads': 4,
        'num_assets': 1,
        'lr': 1e-4,
        'trans_cost': 0.002,
        'annualization': 252.0,
        'batch_size': 32,
        'epochs': 2
    }
    
    print("\n--- Launching Training Loop ---")
    # This will run the forward/backward pass and catch any shape/logic mismatches
    try:
        run_training_job(x, y, train_spreads=s, hparams=hparams)
        print("\nSUCCESS: Local reproduction passed 2 epochs without errors.")
    except Exception as e:
        print(f"\nFAILURE: Local reproduction failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_local_reproduction()
