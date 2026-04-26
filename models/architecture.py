import torch
import torch.nn as nn
from models.components import VariableSelectionNetwork, GatedResidualNetwork
from typing import Optional, List

class EnsembleMomentumTransformer(nn.Module):
    """
    Ensemble Momentum Transformer (Wood et al., 2022).
    Refined with Static Asset-Type Context for academic reproduction.
    """
    def __init__(
        self,
        input_dim: int,
        num_vars: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int = 1,
        dropout: float = 0.3,
        output_dim: int = 1,
        num_static_vars: int = 5 # 4 asset classes + 1 unknown
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Static Context Embedding
        # Maps asset_id to a dense representation for the VSN
        self.static_embedding = nn.Embedding(num_static_vars, hidden_dim)
        
        # 2. Variable Selection Network with Context
        self.vsn = VariableSelectionNetwork(
            input_dim, 
            num_vars, 
            hidden_dim, 
            dropout, 
            context_dim=hidden_dim
        )
        
        # 3. Sequential Processing (Multiple layers as per paper search grid)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.post_lstm_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # 4. Attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 5. Modulation Head
        self.modulation_head = nn.Sequential(
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, asset_id: Optional[torch.Tensor] = None, return_attention: bool = False) -> torch.Tensor:
        """
        x: (batch, time, num_assets, num_features)
        asset_id: (batch, num_assets) or (batch,) - static context
        """
        # --- 1. Calculate the TSMOM Prior ---
        tsmom_prior = torch.mean(x, dim=-1) # (B, T, N_assets)
        
        # --- 2. Static Context ---
        static_context = None
        if asset_id is not None:
            # We take the embedding of the first asset or mean if multiple. 
            # In our implementation, we feed assets as 'variables' in VSN.
            # For simplicity, we use the first asset's ID or broadcast.
            static_context = self.static_embedding(asset_id) # (B, N_assets, hidden_dim)
            if static_context.dim() == 3:
                static_context = static_context.mean(dim=1) # (B, hidden_dim)
        
        # --- 3. Transformer Feature Extraction ---
        vsn_out = self.vsn(x, context=static_context) # (B, T, hidden_dim)
        
        lstm_out, _ = self.lstm(vsn_out)
        gated_out = self.post_lstm_grn(lstm_out)
        
        seq_len = gated_out.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=gated_out.device), diagonal=1).bool()
        
        attn_out, attn_weights = self.attention(
            gated_out, gated_out, gated_out, 
            attn_mask=causal_mask
        )
        
        # --- 4. Modulation ---
        alpha = self.modulation_head(attn_out) # (B, T, N_assets)
        
        # Final position calculation
        positions = torch.tanh(tsmom_prior * alpha)
        
        if return_attention:
            return positions, attn_weights
        return positions

if __name__ == "__main__":
    # Test for 15 assets, 5 features
    batch, time, assets, feats = 8, 252, 15, 10
    model = EnsembleMomentumTransformer(input_dim=feats, num_vars=assets, hidden_dim=64, num_heads=4, output_dim=assets)
    x = torch.randn(batch, time, assets, feats)
    asset_ids = torch.zeros(batch, assets, dtype=torch.long)
    out = model(x, asset_id=asset_ids)
    print(f"Paper-Spec Model Output Shape: {out.shape}") # (8, 252, 15)
