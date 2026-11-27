import os
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from models.momentum_transformer import WoodMomentumTransformer

EXHIBIT_11_ASSETS: Dict[str, int] = {
    # Commodities (25) -> Class 0
    'CC': 0, 'DA': 0, 'GI': 0, 'JO': 0, 'KC': 0, 'KW': 0, 'LB': 0, 'NR': 0, 'SB': 0,
    'ZA': 0, 'ZC': 0, 'ZF': 0, 'ZG': 0, 'ZH': 0, 'ZI': 0, 'ZK': 0, 'ZL': 0, 'ZN': 0,
    'ZO': 0, 'ZP': 0, 'ZR': 0, 'ZT': 0, 'ZU': 0, 'ZW': 0, 'ZZ': 0,
    # Equities (11) -> Class 1
    'CA': 1, 'EN': 1, 'ER': 1, 'ES': 1, 'LX': 1, 'MD': 1, 'SC': 1, 'SP': 1, 'XU': 1, 'XX': 1, 'YM': 1,
    # Fixed Income (5) -> Class 2
    'DT': 2, 'FB': 2, 'TY': 2, 'UB': 2, 'US': 2,
    # Foreign Exchange (9) -> Class 3
    'AN': 3, 'BN': 3, 'CN': 3, 'DX': 3, 'FN': 3, 'JN': 3, 'MP': 3, 'NK': 3, 'SN': 3
}

class WoodNonOverlappingDataset(Dataset):
    """
    Non-overlapping sequence dataset strictly adhering to Wood et al. (2022) Exhibit 15:
    Sequence length tau = 252, Delta train start time = 252 (stride = 252).
    Filtered strictly to the canonical 50 assets from Exhibit 11.
    """
    def __init__(self, df: pd.DataFrame, seq_len: int = 252, stride: int = 252, include_cpd: bool = False):
        self.seq_len = seq_len
        self.stride = stride
        self.include_cpd = include_cpd
        self.feature_names = list(WoodMomentumTransformer.BASE_FEATURE_NAMES)
        if include_cpd:
            self.feature_names.extend(WoodMomentumTransformer.CPD_FEATURE_NAMES)
        
        # Mapping from DataFrame feature columns to model feature names
        self.col_mapping = {
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
            self.col_mapping['cpd_sev_21'] = 'cpd_sev_21'
            self.col_mapping['cpd_sev_126'] = 'cpd_sev_126'
        
        self.data_groups = []
        self.indices = []
        
        valid_symbols = set(EXHIBIT_11_ASSETS.keys())
        df_filtered = df[df['symbol'].isin(valid_symbols)]
        
        for sym, group in df_filtered.groupby('symbol'):
            group = group.sort_index().ffill().bfill()
            if len(group) < seq_len:
                continue
                
            asset_class_id = EXHIBIT_11_ASSETS[sym]
            
            # Extract the 8 continuous normalized features
            feats = {}
            for model_feat, col_name in self.col_mapping.items():
                feats[model_feat] = group[col_name].values.astype(np.float32)
                
            scaled_returns = group['scaled_returns'].values.astype(np.float32) if 'scaled_returns' in group.columns else (group['ret_1'].values.astype(np.float32))
            vol_scale = group['vol_scale'].values.astype(np.float32) if 'vol_scale' in group.columns else np.ones(len(group), dtype=np.float32)
            spread = group['spread'].values.astype(np.float32) if 'spread' in group.columns else np.zeros(len(group), dtype=np.float32)
            
            vals = {
                'feats': feats,
                'y': scaled_returns,
                'vol_scale': vol_scale,
                'spread': spread,
                'aid': asset_class_id,
                'symbol': sym
            }
            
            group_idx = len(self.data_groups)
            self.data_groups.append(vals)
            
            # Non-overlapping striding: stride = seq_len (252)
            for i in range(0, len(group) - seq_len + 1, stride):
                self.indices.append((group_idx, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        g_idx, start = self.indices[idx]
        g = self.data_groups[g_idx]
        end = start + self.seq_len
        
        feat_dict = {
            feat: torch.from_numpy(g['feats'][feat][start:end]).unsqueeze(-1) # [T, 1]
            for feat in self.feature_names
        }
        
        return (
            feat_dict,
            torch.tensor(g['aid'], dtype=torch.long),
            torch.from_numpy(g['y'][start:end]),
            torch.from_numpy(g['vol_scale'][start:end]),
            torch.from_numpy(g['spread'][start:end])
        )

def custom_collate(batch):
    feat_keys = list(batch[0][0].keys())
    batch_feats = {
        feat: torch.stack([item[0][feat] for item in batch], dim=0) # [B, T, 1]
        for feat in feat_keys
    }
    batch_aids = torch.stack([item[1] for item in batch], dim=0)    # [B]
    batch_y = torch.stack([item[2] for item in batch], dim=0)       # [B, T]
    batch_vs = torch.stack([item[3] for item in batch], dim=0)      # [B, T]
    batch_spread = torch.stack([item[4] for item in batch], dim=0)  # [B, T]
    
    return batch_feats, batch_aids, batch_y, batch_vs, batch_spread
