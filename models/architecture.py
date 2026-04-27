import torch
import torch.nn as nn
from models.components import VariableSelectionNetwork, GatedResidualNetwork
from typing import Optional, List

class EnsembleMomentumTransformer(nn.Module):
    """
    Decoder-Only Temporal Fusion Transformer (Wood et al., 2022).
    Implements multi-prior modulation: tanh(A*range + B*mean + C*interaction)
    """
    def __init__(
        self,
        input_dim: int, # Features per variable (usually 1)
        num_vars: int,  # Number of variables (MACD, returns, range, mean)
        hidden_dim: int = 160,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_static_vars: int = 5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Static Context Embedding
        self.static_embedding = nn.Embedding(num_static_vars, hidden_dim)
        
        # 2. Variable Selection Network (Shared-GRN as per paper p.18)
        self.vsn = VariableSelectionNetwork(
            input_dim, 
            num_vars, 
            hidden_dim, 
            dropout, 
            context_dim=hidden_dim
        )
        
        # 3. Sequential Processing (LSTM layers for local temporal context)
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 4. Gated Residual Network post-LSTM
        self.post_lstm_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # 5. Interpretable Multi-Head Attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 6. BROADER PRIOR MODULATION HEAD
        # Outputs 3 coefficients: A (Range), B (Mean), C (Interaction)
        self.modulation_head = nn.Sequential(
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 3) # A, B, C
        )

    def forward(
        self, 
        x: torch.Tensor, 
        asset_id: Optional[torch.Tensor] = None,
        prior_mom: Optional[torch.Tensor] = None,
        prior_trend: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        x: (batch, time, num_vars, input_dim) - All signals
        prior_mom: (batch, time, 1) - e.g. 252-day return
        prior_trend: (batch, time, 1) - e.g. average MACD
        """
        # --- 1. Static Context ---
        static_context = None
        if asset_id is not None:
            static_context = self.static_embedding(asset_id) 
            if static_context.dim() == 3:
                static_context = static_context.mean(dim=1) 
        
        # --- 2. Transformer Feature Extraction ---
        # VSN decides which MACD/returns are important
        vsn_out = self.vsn(x, context=static_context) # (B, T, hidden_dim)
        
        # LSTM processes local dependencies
        lstm_out, _ = self.lstm(vsn_out)
        gated_out = self.post_lstm_grn(lstm_out)
        
        # Causal Attention for long-term dependencies
        seq_len = gated_out.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=gated_out.device), diagonal=1).bool()
        
        attn_out, attn_weights = self.attention(
            gated_out, gated_out, gated_out, 
            attn_mask=causal_mask
        )
        
        # --- 3. Broader Prior Modulation ---
        # Get learned coefficients A, B, C
        coeffs = self.modulation_head(attn_out) # (B, T, 3)
        A = coeffs[:, :, 0:1]
        B = coeffs[:, :, 1:2]
        C = coeffs[:, :, 2:3]
        
        # If priors not passed, extract them from x (indices depend on prepare_hpc_data)
        # For robustness, we assume they are passed for this specific strategy
        if prior_mom is None or prior_trend is None:
            # Fallback to mean of input if not provided (not ideal, but prevents crash)
            pm = torch.mean(x, dim=-2) if x.dim()==4 else torch.mean(x, dim=-1, keepdim=True)
            pt = pm
        else:
            pm = prior_mom
            pt = prior_trend
            
        # Implementation of tanh(A*range + B*mean + C*interaction)
        # Here we map user's 'range' and 'mean' to our Mom and Trend priors
        # Position = tanh( A * prior_mom + B * prior_trend + C * (prior_mom * prior_trend) )
        interaction = pm * pt
        
        logits = (A * pm) + (B * pt) + (C * interaction)
        positions = torch.tanh(logits)
        
        if return_attention:
            return positions, attn_weights
        return positions

if __name__ == "__main__":
    # Test for 10 features (5 ret + 3 macd + 1 range + 1 mean)
    batch, time, feats = 8, 64, 10
    model = EnsembleMomentumTransformer(input_dim=1, num_vars=feats, hidden_dim=64, num_heads=4)
    x = torch.randn(batch, time, feats, 1)
    pm = torch.randn(batch, time, 1)
    pt = torch.randn(batch, time, 1)
    out = model(x, prior_mom=pm, prior_trend=pt)
    print(f"Modulated Model Output Shape: {out.shape}") # (8, 64, 1)
