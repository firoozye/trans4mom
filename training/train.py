import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.momentum_transformer import WoodMomentumTransformer
from data.datasets import WoodNonOverlappingDataset, custom_collate
from engine.loss import SharpeLoss

def get_best_gpu():
    if not torch.cuda.is_available():
        print("CUDA is not available on this host. Falling back to CPU.", flush=True)
        return torch.device("cpu")
    
    num_gpus = torch.cuda.device_count()
    print(f"Probing {num_gpus} CUDA device(s) on host {os.uname().nodename}:", flush=True)
    best_device_id = 0
    max_free_mem = -1
    
    for i in range(num_gpus):
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {free_mem / (1024**3):.2f} GB free / {total_mem / (1024**3):.2f} GB total", flush=True)
            if free_mem > max_free_mem:
                max_free_mem = free_mem
                best_device_id = i
        except Exception as e:
            print(f"  Could not probe GPU {i}: {e}", flush=True)
            
    print(f"Selected GPU {best_device_id} with {max_free_mem / (1024**3):.2f} GB free memory.", flush=True)
    torch.cuda.set_device(best_device_id)
    torch.cuda.empty_cache()
    return torch.device(f"cuda:{best_device_id}")

def main():
    parser = argparse.ArgumentParser(description="Wood et al. (2022) Pure Replication Training")
    parser.add_argument("--config", type=str, default="configs/baseline_wood.yaml", help="Path to config YAML")
    parser.add_argument("--prefix", type=str, default="wood_pure_best", help="Weights checkpoint prefix")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--include_cpd", action="store_true", help="Include GP-CPD fast and slow severity features")
    args = parser.parse_args()

    # Reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    print(f"=== WOOD ET AL. (2022) REPLICATION TRAINING (Seed {seed}, CPD={args.include_cpd}) ===", flush=True)
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    data_path = cfg['data']['processed_path']
    train_split = cfg['training'].get('train_split', '2015-12-31')
    val_split = cfg['training'].get('val_split', '2018-12-31')
    
    print(f"Loading continuous futures data from {data_path}...", flush=True)
    df = pd.read_parquet(data_path)
    
    train_df = df[df.index <= train_split]
    val_df = df[(df.index > train_split) & (df.index <= val_split)]
    
    print(f"Partitions: Train <= {train_split} ({len(train_df):,} rows) | Val {train_split} to {val_split} ({len(val_df):,} rows)", flush=True)
    
    # Non-overlapping 252-day sequences (stride=252)
    seq_len = cfg['model'].get('seq_len', 252)
    stride = cfg['model'].get('stride', 252)
    
    train_ds = WoodNonOverlappingDataset(train_df, seq_len=seq_len, stride=stride, include_cpd=args.include_cpd)
    val_ds = WoodNonOverlappingDataset(val_df, seq_len=seq_len, stride=stride, include_cpd=args.include_cpd)
    
    print(f"Non-Overlapping Blocks (stride={stride}, CPD={args.include_cpd}): Train = {len(train_ds):,} | Val = {len(val_ds):,}", flush=True)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate, num_workers=2)
    
    device = get_best_gpu()
    model = WoodMomentumTransformer(
        d_model=cfg['model'].get('d_model', 64),
        n_heads=cfg['model'].get('n_heads', 4),
        d_head=cfg['model'].get('d_head', 16),
        num_asset_classes=cfg['model'].get('num_asset_classes', 5),
        dropout=cfg['model'].get('dropout', 0.1),
        eps=cfg['model'].get('eps', 0.001),
        include_cpd=args.include_cpd
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized WoodMomentumTransformer: Total Parameters = {total_params:,} on {device}", flush=True)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = SharpeLoss(trans_cost=cfg['training'].get('trans_cost', 0.0001), annualization=252.0)
    
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    best_weight_path = f"weights/{args.prefix}_seed_{seed}.pt"
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_feats, batch_aids, batch_y, batch_vs, batch_spread in train_loader:
            b_feats = {k: v.to(device) for k, v in batch_feats.items()}
            b_aids = batch_aids.to(device)
            b_y = batch_y.to(device).unsqueeze(-1)
            b_vs = batch_vs.to(device).unsqueeze(-1)
            b_spread = batch_spread.to(device).unsqueeze(-1)
            
            optimizer.zero_grad()
            positions = model(b_feats, b_aids)
            
            loss = loss_fn(positions, b_y, spreads=b_spread, vol_scales=b_vs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_feats, batch_aids, batch_y, batch_vs, batch_spread in val_loader:
                b_feats = {k: v.to(device) for k, v in batch_feats.items()}
                b_aids = batch_aids.to(device)
                b_y = batch_y.to(device).unsqueeze(-1)
                b_vs = batch_vs.to(device).unsqueeze(-1)
                b_spread = batch_spread.to(device).unsqueeze(-1)
                
                positions = model(b_feats, b_aids)
                loss = loss_fn(positions, b_y, spreads=b_spread, vol_scales=b_vs)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch:02d}: Train Loss {avg_train_loss:.4f} (Sharpe {-avg_train_loss:.2f}) | Val Loss {avg_val_loss:.4f} (Sharpe {-avg_val_loss:.2f})", flush=True)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_weight_path)
            print(f"  --> Saved new best checkpoint to {best_weight_path} (Val Sharpe {-avg_val_loss:.2f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs (Patience {args.patience}).", flush=True)
                break

    print(f"Training Complete for Seed {seed}. Best Val Sharpe: {-best_val_loss:.2f}.", flush=True)

if __name__ == "__main__":
    main()
