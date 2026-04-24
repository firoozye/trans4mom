import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from engine.trainer import Trainer
from engine.loss import SharpeLoss
from models.architecture import MomentumTransformer

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class CrossSectionalLoss(SharpeLoss):
    """
    Extends SharpeLoss to include a cross-sectional market-neutrality penalty.
    Encourages sum(positions) = 0 across assets at each timestep.
    """
    def __init__(self, neutral_beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.neutral_beta = neutral_beta

    def forward(self, positions, returns, spreads=None):
        # 1. Base Sharpe Loss (Turnover, Bias, SR)
        # positions: (batch, time, num_assets)
        # returns: (batch, time, num_assets)
        base_loss = super().forward(positions, returns, spreads)
        
        # 2. Market Neutrality Penalty: (Sum of positions at each t)^2
        # We want the portfolio to be dollar-neutral
        net_exposure = torch.mean(positions, dim=-1) # (batch, time)
        neutral_penalty = self.neutral_beta * torch.mean(torch.pow(net_exposure, 2))
        
        return base_loss + neutral_penalty

def prepare_cross_sectional_data(df, feat_cols, seq_len=64):
    """
    Reshapes data for multi-asset training.
    Output shapes:
        x: (N, seq_len, num_assets, num_features)
        y: (N, seq_len, num_assets)
    """
    symbols = df['symbol'].unique()
    num_assets = len(symbols)
    
    # Pivot data to have (time, assets, columns)
    pivoted = df.pivot_table(index=df.index, columns='symbol', values=feat_cols + ['scaled_returns', 'spread'])
    pivoted = pivoted.sort_index().ffill().bfill()
    
    # Extract arrays
    x_list = []
    for f in feat_cols:
        x_list.append(pivoted[f].values)
    x_vals = np.stack(x_list, axis=-1) # (time, assets, num_features)
    
    y_vals = pivoted['scaled_returns'].values # (time, assets)
    s_vals = pivoted['spread'].values # (time, assets)
    
    all_x, all_y, all_s = [], [], []
    for i in range(len(pivoted) - seq_len):
        all_x.append(x_vals[i:i+seq_len])
        all_y.append(y_vals[i:i+seq_len])
        all_s.append(s_vals[i:i+seq_len])
        
    return (torch.tensor(np.array(all_x), dtype=torch.float32), 
            torch.tensor(np.array(all_y), dtype=torch.float32),
            torch.tensor(np.array(all_s), dtype=torch.float32))

def main():
    parser = argparse.ArgumentParser(description="Cross-Sectional Momentum Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config['data']
    train_cfg = config['training']
    model_cfg = config['model']

    # 1. Load 1d Data
    data_path = data_cfg['processed_path']
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run data_pipeline.py first.")
        return
    
    df = pd.read_parquet(data_path)
    
    # 2. Temporal Split (Pre-2025)
    train_df = df[df.index <= '2024-12-31']
    print(f"Training on {len(train_df)} rows up to 2024-12-31")

    # 3. Prepare Multi-Asset Sequences
    feat_cols = [f'macd_{w}' for w in config['features']['window_sizes']]
    x_train, y_train, s_train = prepare_cross_sectional_data(train_df, feat_cols)
    print(f"Prepared {x_train.shape[0]} multi-asset sequences of length 64")

    # 4. Model (Output dim = num_assets)
    num_assets = len(df['symbol'].unique())
    num_features = len(feat_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We treat each asset as a 'variable' to be selected, 
    # and each asset has multiple 'features' (input_dim)
    model = MomentumTransformer(
        input_dim=num_features, 
        num_vars=num_assets, 
        hidden_dim=model_cfg['hidden_dim'],
        num_heads=model_cfg['num_heads'],
        output_dim=num_assets
    ).to(device)

    # 5. Trainer with Neutrality Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])
    loss_fn = CrossSectionalLoss(
        neutral_beta=2.0, 
        trans_cost=train_cfg['trans_cost'],
        annualization=365.0, 
        smoothing_beta=train_cfg['smoothing_beta']
    )
    
    # Data is (N, 64, assets, features) -> Model expects this
    train_loader = DataLoader(TensorDataset(x_train, y_train, s_train), batch_size=train_cfg['batch_size'], shuffle=True)
    
    trainer = Trainer(model, optimizer, loss_fn, device)
    
    print("Starting Cross-Sectional Training...")
    for epoch in range(1, train_cfg['epochs'] + 1):
        loss = trainer.train_epoch(train_loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{train_cfg['epochs']} | Loss: {loss:.4f}")
            trainer.save_checkpoint(f"weights/model_cross_sectional_latest.pt")

    print("Training Complete. Model saved to weights/model_cross_sectional_latest.pt")

if __name__ == "__main__":
    main()
