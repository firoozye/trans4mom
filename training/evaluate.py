import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.momentum_transformer import WoodMomentumTransformer
from data.datasets import EXHIBIT_11_ASSETS

def calculate_portfolio_metrics(daily_returns: pd.Series, annualization: float = 252.0):
    daily_returns = daily_returns.dropna()
    if len(daily_returns) == 0:
        return {'sharpe': 0.0, 'annual_ret': 0.0, 'annual_vol': 0.0, 'total_pnl': 0.0}
    
    mean = daily_returns.mean()
    std = daily_returns.std()
    sharpe = (mean / (std + 1e-8)) * np.sqrt(annualization)
    ann_ret = (1.0 + mean)**annualization - 1.0
    ann_vol = std * np.sqrt(annualization)
    total_pnl = (1.0 + daily_returns).prod() - 1.0
    
    return {
        'sharpe': float(sharpe),
        'annual_ret': float(ann_ret) * 100.0,
        'annual_vol': float(ann_vol) * 100.0,
        'total_pnl': float(total_pnl) * 100.0
    }

def evaluate_models(prefix: str, include_cpd: bool = False, cost_bp: float = 1.0, data_path: str = "data/processed/pinnacle_futures_1990_2026.parquet"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Out-of-Sample Evaluation (prefix: {prefix}, CPD: {include_cpd}) on {device}...", flush=True)

    df = pd.read_parquet(data_path)
    valid_symbols = set(EXHIBIT_11_ASSETS.keys())
    df_50 = df[df['symbol'].isin(valid_symbols)]
    print(f"Loaded {len(df_50):,} rows across {df_50['symbol'].nunique()} canonical contracts.", flush=True)

    weight_paths = [f"weights/{prefix}_seed_{s}.pt" for s in range(1, 6)]
    models = []
    for s, wp in enumerate(weight_paths, 1):
        if not os.path.exists(wp):
            print(f"Warning: {wp} not found!")
            continue
        model = WoodMomentumTransformer(
            d_model=64, 
            n_heads=4, 
            d_head=16, 
            num_asset_classes=5,
            include_cpd=include_cpd
        ).to(device)
        model.load_state_dict(torch.load(wp, map_location=device))
        model.eval()
        models.append((f"Seed {s}", model))

    print(f"Successfully loaded {len(models)} checkpoints.", flush=True)

    seq_len = 252
    feature_names = list(WoodMomentumTransformer.BASE_FEATURE_NAMES)
    col_mapping = {
        'norm_daily_return': 'ret_1',
        'norm_monthly_return': 'ret_21',
        'norm_quarterly_return': 'ret_63',
        'norm_half_year_return': 'ret_126',
        'norm_annual_return': 'ret_252',
        'macd_8_24': 'macd_8_24',
        'macd_16_48': 'macd_16_48',
        'macd_32_96': 'macd_32_96'
    }
    if include_cpd:
        feature_names.extend(WoodMomentumTransformer.CPD_FEATURE_NAMES)
        col_mapping['cpd_sev_21'] = 'cpd_sev_21'
        col_mapping['cpd_sev_126'] = 'cpd_sev_126'

    cost_factor = cost_bp * 0.0001 # 1 bp = 0.0001

    asset_seed_gross = {name: [] for name, _ in models}
    asset_seed_net = {name: [] for name, _ in models}
    asset_ensemble_gross = []
    asset_ensemble_net = []
    
    with torch.no_grad():
        for sym, group in df_50.groupby('symbol'):
            group = group.sort_index().ffill().bfill()
            if len(group) < seq_len:
                continue

            aid = torch.tensor([EXHIBIT_11_ASSETS[sym]], device=device, dtype=torch.long)
            group_feats = {
                feat: torch.from_numpy(group[col_mapping[feat]].values.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
                for feat in feature_names
            }

            scaled_rets = group['scaled_returns'].values.astype(np.float32) if 'scaled_returns' in group.columns else group['ret_1'].values.astype(np.float32)
            vol_scale = group['vol_scale'].values.astype(np.float32) if 'vol_scale' in group.columns else np.ones(len(group), dtype=np.float32)

            seed_positions = []
            for name, m in models:
                pos = m(group_feats, aid).squeeze(0).squeeze(-1).cpu().numpy()
                seed_positions.append(pos)

                pos_prev = np.roll(pos, 1)
                pos_prev[0] = 0.0
                turnover = np.abs(pos * vol_scale - pos_prev * vol_scale)
                
                gross_ret = pos_prev * scaled_rets
                net_ret = gross_ret - cost_factor * turnover

                asset_seed_gross[name].append(pd.Series(gross_ret, index=group.index, name=f"{sym}_{name}_gross"))
                asset_seed_net[name].append(pd.Series(net_ret, index=group.index, name=f"{sym}_{name}_net"))

            # Ensemble average position
            ens_pos = np.mean(seed_positions, axis=0)
            ens_pos_prev = np.roll(ens_pos, 1)
            ens_pos_prev[0] = 0.0
            ens_turnover = np.abs(ens_pos * vol_scale - ens_pos_prev * vol_scale)
            
            ens_gross_ret = ens_pos_prev * scaled_rets
            ens_net_ret = ens_gross_ret - cost_factor * ens_turnover

            asset_ensemble_gross.append(pd.Series(ens_gross_ret, index=group.index, name=f"{sym}_ens_gross"))
            asset_ensemble_net.append(pd.Series(ens_net_ret, index=group.index, name=f"{sym}_ens_net"))

    port_ens_gross = pd.concat(asset_ensemble_gross, axis=1).mean(axis=1)
    port_ens_net = pd.concat(asset_ensemble_net, axis=1).mean(axis=1)

    horizons = {
        'In-Sample (1990-2015, Training)': ('1990-01-01', '2015-12-31'),
        'Validation (2016-2018, Selection)': ('2016-01-01', '2018-12-31'),
        'Paper Test (2015-2020, Wood Exhibit 3)': ('2015-01-01', '2020-12-31'),
        'Live Extension (2021-2026)': ('2021-01-01', '2026-12-31'),
        'Full History (1990-2026)': ('1990-01-01', '2026-12-31')
    }

    model_title = "WOOD ET AL. (2022) + GP-CPD ENSEMBLE" if include_cpd else "WOOD ET AL. (2022) BASELINE ENSEMBLE"
    print("\n" + "="*95)
    print(f"{model_title}: MULTI-ASSET PORTFOLIO BENCHMARK")
    print("="*95)
    print(f"{'Horizon / Partition':<38} | {'Gross SR':<8} | {'Net SR':<8} | {'Ann Ret (Net)':<14} | {'Ann Vol':<9} | {'Net PnL':<10}")
    print("-"*95)

    for h_name, (start_d, end_d) in horizons.items():
        sub_gross = port_ens_gross[(port_ens_gross.index >= start_d) & (port_ens_gross.index <= end_d)]
        sub_net = port_ens_net[(port_ens_net.index >= start_d) & (port_ens_net.index <= end_d)]
        mg = calculate_portfolio_metrics(sub_gross)
        mn = calculate_portfolio_metrics(sub_net)
        print(f"{h_name:<38} | {mg['sharpe']:<8.2f} | {mn['sharpe']:<8.2f} | {mn['annual_ret']:<13.2f}% | {mn['annual_vol']:<8.2f}% | {mn['total_pnl']:<+9.2f}%")

    print("="*95)

    print("\nINDIVIDUAL SEED PERFORMANCE: PAPER TEST PERIOD (2015-2020)")
    print("-"*85)
    print(f"{'Model / Seed':<20} | {'Gross SR':<10} | {'Net SR (1bp)':<12} | {'Ann Ret (Net)':<14} | {'Ann Vol':<9}")
    print("-"*85)
    for name, _ in models:
        s_g = pd.concat(asset_seed_gross[name], axis=1).mean(axis=1)
        s_n = pd.concat(asset_seed_net[name], axis=1).mean(axis=1)
        sub_sg = s_g[(s_g.index >= '2015-01-01') & (s_g.index <= '2020-12-31')]
        sub_sn = s_n[(s_n.index >= '2015-01-01') & (s_n.index <= '2020-12-31')]
        smg = calculate_portfolio_metrics(sub_sg)
        smn = calculate_portfolio_metrics(sub_sn)
        print(f"{name:<20} | {smg['sharpe']:<10.2f} | {smn['sharpe']:<12.2f} | {smn['annual_ret']:<13.2f}% | {smn['annual_vol']:<8.2f}%")
    
    em_g = calculate_portfolio_metrics(port_ens_gross[(port_ens_gross.index >= '2015-01-01') & (port_ens_gross.index <= '2020-12-31')])
    em_n = calculate_portfolio_metrics(port_ens_net[(port_ens_net.index >= '2015-01-01') & (port_ens_net.index <= '2020-12-31')])
    print(f"{'5-Seed Ensemble':<20} | {em_g['sharpe']:<10.2f} | {em_n['sharpe']:<12.2f} | {em_n['annual_ret']:<13.2f}% | {em_n['annual_vol']:<8.2f}%")
    print("-"*85)

def main():
    parser = argparse.ArgumentParser(description="Wood et al. (2022) Performance Evaluation")
    parser.add_argument("--prefix", type=str, default="wood_pure_best", help="Weights prefix (e.g., wood_pure_best or wood_cpd_best)")
    parser.add_argument("--include_cpd", action="store_true", help="Include GP-CPD severity features")
    parser.add_argument("--cost_bp", type=float, default=1.0, help="Transaction cost in basis points (default: 1.0)")
    parser.add_argument("--data_path", type=str, default="data/processed/pinnacle_futures_1990_2026.parquet")
    args = parser.parse_args()

    evaluate_models(args.prefix, include_cpd=args.include_cpd, cost_bp=args.cost_bp, data_path=args.data_path)

if __name__ == "__main__":
    main()
