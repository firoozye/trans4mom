import os
import sys
import torch
import pandas as pd
import numpy as np

PROJECT_ROOT = '/cs/research/fca/home0/firoozye/dev/trans4mom'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.architecture import EnsembleMomentumTransformer
from engine.loss import SharpeLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

data_path = os.path.join(PROJECT_ROOT, 'data/processed/futures_1990_present.parquet')
df = pd.read_parquet(data_path)
df_asset = df[df['symbol'] == 'ES=F'].sort_index()
print(f"ES=F rows: {len(df_asset)}")

feat_cols = [f'ret_{k}' for k in [1, 21, 63, 126, 252]] + \
            [f'macd_{S}_{L}' for S, L in [[8, 24], [16, 48], [32, 96]]] + \
            ['cpd_sev_21', 'cpd_sev_126']

print(f"Features ({len(feat_cols)}): {feat_cols}")

# Instantiate model
model = EnsembleMomentumTransformer(
    input_dim=1,
    num_vars=len(feat_cols),
    hidden_dim=160,
    num_heads=4,
    num_layers=1,
    dropout=0.1,
    output_dropout=0.1,
    num_static_vars=5
).to(device)

weights_path = os.path.join(PROJECT_ROOT, 'weights/modulated_baseline_final.pt')
ckpt = torch.load(weights_path, map_location=device)
model.load_state_dict(ckpt, strict=False)
model.eval()

# Test 252-day forward pass
sample_slice = df_asset.tail(252)
x_window = sample_slice[feat_cols].values.astype(np.float32)
pm_window = sample_slice['prior_mom'].values.astype(np.float32).reshape(-1, 1)

x_tensor = torch.from_numpy(x_window).unsqueeze(0).unsqueeze(-1).to(device)
pm_tensor = torch.from_numpy(pm_window).unsqueeze(0).to(device)
aid_tensor = torch.tensor([0], dtype=torch.long).to(device)

with torch.no_grad():
    pos_out, attn_weights = model(x_tensor, asset_id=aid_tensor, prior_mom=pm_tensor, return_attention=True)

print(f"pos_out shape: {pos_out.shape}, attn_weights shape: {attn_weights.shape}")
print(f"Sample positions: {pos_out[0, -5:, 0].cpu().numpy()}")
print("ALL TESTS PASSED!")
