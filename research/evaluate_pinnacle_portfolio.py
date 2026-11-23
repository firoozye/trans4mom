#!/usr/bin/env python3
"""
Evaluate the trained Pinnacle Momentum Transformer models
across the 55-asset multi-asset portfolio.
Calculates individual seed performance and the multi-seed Ensemble.
"""

import os
import sys
import glob
import torch
import pandas as pd
import numpy as np
from models.architecture import EnsembleMomentumTransformer

def evaluate_models(data_path: str, weight_paths: list, seq_len: int = 252):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}", flush=True)
    
    df = pd.read_parquet(data_path)
    feat_cols = [f'ret_{k}' for k in [1, 21, 63, 126, 252]] + \
                [f'macd_{S}_{L}' for S, L in [[8, 24], [16, 48], [32, 96]]] + \
                ['cpd_sev_21', 'cpd_sev_126']
                
    models = []
    for wp in weight_paths:
        m = EnsembleMomentumTransformer(
            input_dim=1,
            num_vars=len(feat_cols),
            hidden_dim=64,
            num_heads=4,
            num_layers=1,
            dropout=0.1,
            output_dropout=0.1,
            num_static_vars=5
        ).to(device)
        ckpt = torch.load(wp, map_location=device)
        m.load_state_dict(ckpt)
        m.eval()
        models.append((os.path.basename(wp), m))

    print(f"Loaded {len(models)} models for evaluation.", flush=True)

    seed_portfolios = {name: [] for name, _ in models}
    ensemble_asset_pnls = []
    prior_pnls = []

    symbols = df['symbol'].unique()
    print(f"Running inference across {len(symbols)} assets...", flush=True)

    with torch.no_grad():
        for sym_idx, (sym, group) in enumerate(df.groupby('symbol')):
            group = group.sort_index().ffill().bfill()
            if len(group) <= seq_len:
                continue

            x_vals = group[feat_cols].values.astype(np.float32)
            rets = group['scaled_returns'].values.astype(np.float32)
            pm = group['prior_mom'].values.astype(np.float32)
            mon = group.index.month.values.astype(np.int64)
            day = group.index.dayofweek.values.astype(np.int64)
            aid = int(group['asset_id'].iloc[0])

            # Prepare sliding windows
            num_steps = len(x_vals)
            windows_x, windows_pm, windows_mon, windows_day = [], [], [], []

            for i in range(num_steps):
                if i < seq_len:
                    wx = np.zeros((seq_len, len(feat_cols)), dtype=np.float32)
                    wx[-(i+1):] = x_vals[:i+1]
                    wpm = np.zeros((seq_len, 1), dtype=np.float32)
                    wpm[-(i+1):] = pm[:i+1].reshape(-1, 1)
                    wmon = np.zeros(seq_len, dtype=np.int64)
                    wmon[-(i+1):] = mon[:i+1]
                    wday = np.zeros(seq_len, dtype=np.int64)
                    wday[-(i+1):] = day[:i+1]
                else:
                    wx = x_vals[i-seq_len+1 : i+1]
                    wpm = pm[i-seq_len+1 : i+1].reshape(-1, 1)
                    wmon = mon[i-seq_len+1 : i+1]
                    wday = day[i-seq_len+1 : i+1]
                windows_x.append(wx)
                windows_pm.append(wpm)
                windows_mon.append(wmon)
                windows_day.append(wday)

            batch_size = 1024
            seed_positions = {name: [] for name, _ in models}

            for b in range(0, len(windows_x), batch_size):
                bx = torch.from_numpy(np.stack(windows_x[b:b+batch_size])).to(device)
                bpm = torch.from_numpy(np.stack(windows_pm[b:b+batch_size])).to(device)
                bmon = torch.from_numpy(np.stack(windows_mon[b:b+batch_size])).to(device)
                bday = torch.from_numpy(np.stack(windows_day[b:b+batch_size])).to(device)
                baid = torch.tensor([aid], dtype=torch.long, device=device).expand(bx.size(0))

                for name, m in models:
                    pos_out = m(bx, asset_id=baid, month=bmon, day=bday, prior_mom=bpm)
                    seed_positions[name].append(pos_out[:, -1, 0].cpu().numpy())

            # Position calculations
            all_model_positions = []
            for name, _ in models:
                m_pos = np.concatenate(seed_positions[name])
                all_model_positions.append(m_pos)
                costs = np.abs(np.diff(m_pos, prepend=0)) * 0.0001
                pnl = (m_pos[:-1] * rets[1:]) - costs[1:]
                seed_portfolios[name].append(pd.Series(pnl, index=group.index[1:], name=sym))

            # Multi-seed Ensemble Position (Average of models)
            ens_pos = np.mean(all_model_positions, axis=0)
            costs_ens = np.abs(np.diff(ens_pos, prepend=0)) * 0.0001
            pnl_ens = (ens_pos[:-1] * rets[1:]) - costs_ens[1:]
            ensemble_asset_pnls.append(pd.Series(pnl_ens, index=group.index[1:], name=sym))

            # Classical Prior Baseline
            prior_pos = np.tanh(pm)
            costs_prior = np.abs(np.diff(prior_pos, prepend=0)) * 0.0001
            pnl_prior = (prior_pos[:-1] * rets[1:]) - costs_prior[1:]
            prior_pnls.append(pd.Series(pnl_prior, index=group.index[1:], name=sym))

    # Portfolios
    port_prior = pd.concat(prior_pnls, axis=1).mean(axis=1)
    port_ensemble = pd.concat(ensemble_asset_pnls, axis=1).mean(axis=1)
    
    def print_period_stats(model_s, prior_s, start_date, end_date, name):
        m = model_s.loc[start_date:end_date]
        p = prior_s.loc[start_date:end_date]
        sr_m = (m.mean() / (m.std() + 1e-9)) * np.sqrt(252)
        sr_p = (p.mean() / (p.std() + 1e-9)) * np.sqrt(252)
        ret_m = m.mean() * 252 * 100
        ret_p = p.mean() * 252 * 100
        vol_m = m.std() * np.sqrt(252) * 100
        vol_p = p.std() * np.sqrt(252) * 100
        print(f"  {name:20s} | Model SR: {sr_m:5.2f} (Ret: {ret_m:5.1f}%, Vol: {vol_m:4.1f}%) | Prior SR: {sr_p:5.2f} (Ret: {ret_p:5.1f}%, Vol: {vol_p:4.1f}%)", flush=True)

    # Print Individual Seed Results
    for name, _ in models:
        print(f"\n=======================================================", flush=True)
        print(f"Results for Individual Seed: {name}", flush=True)
        print(f"=======================================================", flush=True)
        port_seed = pd.concat(seed_portfolios[name], axis=1).mean(axis=1)
        print_period_stats(port_seed, port_prior, '1990-01-01', '2022-12-31', 'IN-SAMPLE (1990-2022)')
        print_period_stats(port_seed, port_prior, '2023-01-01', '2026-11-20', 'OUT-OF-SAMPLE TOTAL')
        print_period_stats(port_seed, port_prior, '2023-01-01', '2024-12-31', '  ↳ OOS Val (2023-2024)')
        print_period_stats(port_seed, port_prior, '2025-01-01', '2026-11-20', '  ↳ OOS Test (2025-2026)')
        print(f"  [Footnote: Full 1990-2026 Reference | Model SR: {((port_seed.mean()/(port_seed.std()+1e-9))*np.sqrt(252)):.2f}]", flush=True)

    # Print Ensemble Results
    print(f"\n=======================================================", flush=True)
    print(f"*** MULTI-SEED ENSEMBLE RESULTS ({len(models)} SEEDS COMBINED) ***", flush=True)
    print(f"=======================================================", flush=True)
    print_period_stats(port_ensemble, port_prior, '1990-01-01', '2022-12-31', 'IN-SAMPLE (1990-2022)')
    print_period_stats(port_ensemble, port_prior, '2023-01-01', '2026-11-20', 'OUT-OF-SAMPLE TOTAL')
    print_period_stats(port_ensemble, port_prior, '2023-01-01', '2024-12-31', '  ↳ OOS Val (2023-2024)')
    print_period_stats(port_ensemble, port_prior, '2025-01-01', '2026-11-20', '  ↳ OOS Test (2025-2026)')
    print(f"  [Footnote: Full 1990-2026 Reference | Model SR: {((port_ensemble.mean()/(port_ensemble.std()+1e-9))*np.sqrt(252)):.2f}]", flush=True)

    # Save PnL Series and Generate Visual Charts
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/data", exist_ok=True)

    # Compile all series into a dataframe
    df_pnl = pd.DataFrame({'Prior_TSMOM': port_prior, 'Ensemble': port_ensemble})
    for name, _ in models:
        clean_name = name.replace('.pt', '')
        df_pnl[clean_name] = pd.concat(seed_portfolios[name], axis=1).mean(axis=1)

    csv_path = "reports/data/pinnacle_portfolio_daily_pnl.csv"
    df_pnl.to_csv(csv_path)
    print(f"\nSaved daily portfolio PnL series to: {csv_path}", flush=True)

    # Plot Cumulative PnL Chart using matplotlib
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

        # Cumulative PnL curves (percentage returns)
        cum_pnl = df_pnl.cumsum() * 100

        # Plot Individual Seeds
        seed_cols = [c for c in df_pnl.columns if c.startswith('pinnacle_best_seed_')]
        for c in seed_cols:
            seed_label = c.replace('pinnacle_best_', '').replace('_', ' ').title()
            ax1.plot(cum_pnl.index, cum_pnl[c], alpha=0.35, linewidth=1.2, label=f"Model {seed_label}")

        # Plot Ensemble & Prior
        ax1.plot(cum_pnl.index, cum_pnl['Ensemble'], color='#0D47A1', linewidth=2.5, label=f'Momentum Transformer Ensemble ({len(models)} Seeds)')
        ax1.plot(cum_pnl.index, cum_pnl['Prior_TSMOM'], color='#B71C1C', linestyle='--', linewidth=1.8, label='Classical Prior Baseline (TSMOM)')

        # Visual Out-of-Sample highlighting (2023 to 2026)
        train_end = pd.to_datetime('2022-12-31')
        val_end = pd.to_datetime('2024-12-31')
        ax1.axvspan(train_end, cum_pnl.index[-1], color='#E8F5E9', alpha=0.5, label='OUT-OF-SAMPLE (2023-2026)')
        ax1.axvline(train_end, color='#2E7D32', linestyle='--', linewidth=1.5, label='In-Sample / OOS Boundary (2022)')
        ax1.axvline(val_end, color='#7B1FA2', linestyle=':', linewidth=1.2, label='OOS Val / Test Boundary (2024)')

        ax1.set_title("Wood, Roberts & Zohren (2022) Replication: In-Sample vs. Out-of-Sample PnL\n55 Continuous Futures Contracts (Equities, Fixed Income, Commodities, FX)", fontsize=13, fontweight='bold', pad=12)
        ax1.set_ylabel("Cumulative Return (%)", fontsize=11)
        ax1.legend(loc='upper left', frameon=True, framealpha=0.9)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Drawdown Subplot
        cum_wealth = (1 + df_pnl).cumprod()
        running_max = cum_wealth.cummax()
        dd_ens = (cum_wealth['Ensemble'] - running_max['Ensemble']) / running_max['Ensemble'] * 100
        dd_prior = (cum_wealth['Prior_TSMOM'] - running_max['Prior_TSMOM']) / running_max['Prior_TSMOM'] * 100

        ax2.plot(dd_ens.index, dd_ens, color='#0D47A1', linewidth=1.5, label='Ensemble Drawdown')
        ax2.plot(dd_prior.index, dd_prior, color='#B71C1C', linestyle='--', linewidth=1.2, label='Prior Baseline Drawdown')
        ax2.fill_between(dd_ens.index, dd_ens, 0, color='#0D47A1', alpha=0.15)
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.legend(loc='lower left', frameon=True)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        chart_path = "reports/figures/pinnacle_cumulative_pnl.png"
        chart_path_tagged = f"reports/figures/pinnacle_cumulative_pnl_{len(models)}seeds.png"
        fig.savefig(chart_path, dpi=300)
        fig.savefig(chart_path_tagged, dpi=300)
        plt.close(fig)
        print(f"Generated full-history PnL chart at: {chart_path} and {chart_path_tagged}", flush=True)

        # Dedicated OUT-OF-SAMPLE (2023-2026) Zoom Chart
        df_oos = df_pnl.loc['2023-01-01':]
        cum_oos = df_oos.cumsum() * 100

        fig_oos, ax_oos = plt.subplots(figsize=(12, 6))
        for c in seed_cols:
            seed_label = c.replace('pinnacle_best_', '').replace('_', ' ').title()
            ax_oos.plot(cum_oos.index, cum_oos[c], alpha=0.35, linewidth=1.2, label=f"Model {seed_label}")
        ax_oos.plot(cum_oos.index, cum_oos['Ensemble'], color='#0D47A1', linewidth=2.5, label=f'Ensemble {len(models)} Seeds (OOS)')
        ax_oos.plot(cum_oos.index, cum_oos['Prior_TSMOM'], color='#B71C1C', linestyle='--', linewidth=1.8, label='Prior TSMOM (OOS)')
        ax_oos.axvline(val_end, color='#7B1FA2', linestyle=':', linewidth=1.5, label='Validation / Test Boundary')
        ax_oos.set_title(f"OUT-OF-SAMPLE PERFORMANCE (2023 - 2026) [{len(models)}-Seed Ensemble]\n55 Futures Contracts, Transaction Costs Deducted", fontsize=12, fontweight='bold', pad=10)
        ax_oos.set_ylabel("Cumulative Out-of-Sample Return (%)", fontsize=11)
        ax_oos.set_xlabel("Date", fontsize=11)
        ax_oos.legend(loc='upper left', frameon=True)
        ax_oos.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        oos_chart_path = "reports/figures/pinnacle_out_of_sample_pnl.png"
        oos_chart_path_tagged = f"reports/figures/pinnacle_out_of_sample_pnl_{len(models)}seeds.png"
        fig_oos.savefig(oos_chart_path, dpi=300)
        fig_oos.savefig(oos_chart_path_tagged, dpi=300)
        plt.close(fig_oos)
        print(f"Generated dedicated Out-of-Sample chart at: {oos_chart_path} and {oos_chart_path_tagged}", flush=True)

    except Exception as e:
        print(f"Warning: Could not plot chart ({e})", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Pinnacle Momentum Transformer Portfolio")
    parser.add_argument("--data", type=str, default="data/processed/pinnacle_futures_1990_2026.parquet")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed IDs to evaluate (e.g. '1,2,4,5')")
    args = parser.parse_args()

    if args.seeds:
        seed_list = [s.strip() for s in args.seeds.split(',')]
        weights = [f"weights/pinnacle_best_seed_{s}.pt" for s in seed_list if os.path.exists(f"weights/pinnacle_best_seed_{s}.pt")]
    else:
        weights = sorted(glob.glob("weights/pinnacle_best_seed_*.pt"))

    evaluate_models(args.data, weights)
