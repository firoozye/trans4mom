import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from engine.trainer import Trainer
from engine.loss import SharpeLoss
from models.architecture import EnsembleMomentumTransformer

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
        net_exposure = torch.mean(positions, dim=-1) # (batch, time)
        neutral_penalty = self.neutral_beta * torch.mean(torch.pow(net_exposure, 2))
        
        return base_loss + neutral_penalty

def prepare_cross_sectional_data(df, feat_cols, seq_len=252):
    """
    Reshapes data for multi-asset training.
    """
    symbols = df['symbol'].unique()
    num_assets = len(symbols)
    
    pivoted = df.pivot_table(index=df.index, columns='symbol', values=feat_cols + ['scaled_returns', 'spread', 'asset_id'])
    pivoted = pivoted.sort_index().ffill().bfill()
    
    x_list = [pivoted[f].values for f in feat_cols]
    x_vals = np.stack(x_list, axis=-1) 
    y_vals = pivoted['scaled_returns'].values 
    s_vals = pivoted['spread'].values 
    aid_vals = pivoted['asset_id'].values # (time, assets)
    
    all_x, all_y, all_s, all_aid = [], [], [], []
    for i in range(len(pivoted) - seq_len):
        all_x.append(x_vals[i:i+seq_len])
        all_y.append(y_vals[i:i+seq_len])
        all_s.append(s_vals[i:i+seq_len])
        all_aid.append(aid_vals[i:i+seq_len])
        
    return (torch.tensor(np.array(all_x), dtype=torch.float32), 
            torch.tensor(np.array(all_y), dtype=torch.float32),
            torch.tensor(np.array(all_s), dtype=torch.float32),
            torch.tensor(np.array(all_aid), dtype=torch.long))

def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Ensemble Momentum Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to ensemble")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config['data']
    train_cfg = config['training']
    model_cfg = config['model']

    data_path = data_cfg['processed_path']
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    df = pd.read_parquet(data_path)
    train_df = df[df.index <= '2024-12-31']
    feat_cols = [c for c in df.columns if c.startswith('ret_') or c.startswith('macd_')]
    
    seq_len = 252
    x_train, y_train, s_train, aid_train = prepare_cross_sectional_data(train_df, feat_cols, seq_len=seq_len)
    print(f"Prepared {x_train.shape[0]} sequences of length {seq_len}")

    num_assets = len(df['symbol'].unique())
    num_features = len(feat_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("weights", exist_ok=True)
    
    print(f"Starting Multi-Seed Ensemble Training ({args.seeds} seeds)...")
    
    for seed in range(args.seeds):
        print(f"\n--- Training Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = EnsembleMomentumTransformer(
            input_dim=num_features, 
            num_vars=num_assets, 
            hidden_dim=model_cfg['hidden_dim'],
            num_heads=model_cfg['num_heads'],
            output_dim=num_assets
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])
        loss_fn = CrossSectionalLoss(
            neutral_beta=2.0, 
            trans_cost=train_cfg['trans_cost'],
            annualization=train_cfg['annualization'], 
            smoothing_beta=train_cfg['smoothing_beta']
        )
        
        train_loader = DataLoader(TensorDataset(x_train, y_train, s_train, aid_train), batch_size=train_cfg['batch_size'], shuffle=True)
        
        # Manually handling the trainer forward pass to include asset_id
        for epoch in range(1, train_cfg['epochs'] + 1):
            model.train()
            total_loss = 0
            for batch in train_loader:
                x, y, spreads, aids = [b.to(device) for b in batch]
                optimizer.zero_grad()
                
                # aids is (B, T, N) but model wants (B, N) for static context
                output_pos = model(x, asset_id=aids[:, 0, :])
                loss = loss_fn(output_pos, y, spreads=spreads)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if epoch % 50 == 0:
                print(f"Seed {seed} | Epoch {epoch}/{train_cfg['epochs']} | Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), f"weights/model_ensemble_seed_{seed}.pt")

    print(f"\nEnsemble Training Complete. {args.seeds} models saved to weights/")

if __name__ == "__main__":
    main()
