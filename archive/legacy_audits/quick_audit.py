import os
import matplotlib
matplotlib.use('Agg')
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.architecture import EnsembleMomentumTransformer
from data.processor import FeatureProcessor
from research.final_audit import run_modulated_backtest

print("--- Backtest Start ---")
data_path = 'data/processed/futures_1990_present.parquet'
df_raw = pd.read_parquet(data_path)
print(f"Loaded {len(df_raw)} rows.")

processor = FeatureProcessor(
    return_windows=[1, 21, 63, 126, 252],
    macd_pairs=[(8, 24), (16, 48), (32, 96)]
)

processed_groups = []
for sym, group in df_raw.groupby('symbol'):
    processed_groups.append(processor.process_features(group))
df = pd.concat(processed_groups)

feat_cols = ['ret_1', 'ret_21', 'ret_63', 'ret_126', 'ret_252', 'macd_8_24', 'macd_16_48', 'macd_32_96']
device = torch.device("cpu")
model = EnsembleMomentumTransformer(
    input_dim=1, num_vars=len(feat_cols) + 2, hidden_dim=160, num_heads=4, num_layers=2
)
model.load_state_dict(torch.load('weights/modulated_baseline_final.pt', map_location='cpu'))
model.eval()

results = run_modulated_backtest(model, df, feat_cols, ann_factor=252)
print(f"Portfolio Sharpe: {results['sr']:.2f}")

# Individual Asset Sharpes
asset_pnls = pd.DataFrame({sym: results['positions'] for sym, results in results.items() if isinstance(results, dict) and 'cum_pnl' in results})
# Wait, run_modulated_backtest returns a dict with 'combined_df' logic. 
# Let me just calculate it from combined_df in run_modulated_backtest.
# Actually, I'll modify run_modulated_backtest to return individual SRs.

# Annual Breakdown
combined_rets = pd.Series(results['cum_pnl'], index=results['dates']).diff().fillna(results['cum_pnl'][0])
annual_sr = combined_rets.groupby(combined_rets.index.year).apply(
    lambda x: (x.mean() / (x.std() + 1e-9)) * np.sqrt(252)
)
print("\nAnnual Sharpe Ratios:")
print(annual_sr)

plt.figure(figsize=(10, 6))
plt.plot(results['dates'], results['cum_pnl'])
plt.savefig("research/final_audit_results.png")
print("Done.")
