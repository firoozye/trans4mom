import torch
import torch.nn as nn
from models.components import VariableSelectionNetwork, GatedResidualNetwork
from typing import Optional, List

class EnsembleMomentumTransformer(nn.Module):
    """
    Decoder-Only Temporal Fusion Transformer (Wood et al., 2022).
    Implements Residual Prior Modulation: Position = tanh( (A*p1 + B*p2 + C*int) * (1 + alpha) )
    """
    def __init__(
        self,
        input_dim: int,
        num_vars: int,
        hidden_dim: int = 160,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_static_vars: int = 5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Static and Temporal Embeddings
        self.static_embedding = nn.Embedding(num_static_vars, hidden_dim)
        self.month_embedding = nn.Embedding(13, hidden_dim) # 1-12
        self.day_embedding = nn.Embedding(7, hidden_dim)   # 0-6
        
        # 2. VSN (Shared-GRN)
        self.vsn = VariableSelectionNetwork(input_dim, num_vars, hidden_dim, dropout, context_dim=hidden_dim)
        
        # 3. LSTM + GRN
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.post_lstm_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # 4. Multi-Head Attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 5. Transformer Signal Head (alpha)
        self.alpha_head = nn.Sequential(
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
        # 6. Learned Prior Coefficients
        self.prior_weights = nn.Parameter(torch.ones(3)) 

    def forward(
        self, 
        x: torch.Tensor, 
        asset_id: Optional[torch.Tensor] = None,
        month: Optional[torch.Tensor] = None,
        day: Optional[torch.Tensor] = None,
        prior_mom: Optional[torch.Tensor] = None, 
        prior_trend: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        # --- 1. Transformer Logic ---
        # Contextual Information
        static_context = self.static_embedding(asset_id) if asset_id is not None else None
        if static_context is not None and static_context.dim() == 3:
            static_context = static_context.mean(dim=1)
            
        # Add temporal context to static context
        if month is not None and day is not None:
            # month/day are (B, T) or (B,)
            m_emb = self.month_embedding(month)
            d_emb = self.day_embedding(day)
            if m_emb.dim() == 3: m_emb = m_emb.mean(dim=1)
            if d_emb.dim() == 3: d_emb = d_emb.mean(dim=1)
            
            if static_context is not None:
                static_context = static_context + m_emb + d_emb
            else:
                static_context = m_emb + d_emb

        vsn_out = self.vsn(x, context=static_context)
        lstm_out, _ = self.lstm(vsn_out)
        gated_out = self.post_lstm_grn(lstm_out)
        
        seq_len = gated_out.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=gated_out.device), diagonal=1).bool()
        
        attn_out, attn_weights = self.attention(gated_out, gated_out, gated_out, attn_mask=causal_mask)
        
        # --- 2. Calculate Adjustment (alpha) ---
        alpha = self.alpha_head(attn_out) # (B, T, 1)
        
        # --- 3. Construct the Robust Composite Prior ---
        # LearnedPrior = A*range + B*mean + C*(range*mean)
        pm = prior_mom if prior_mom is not None else torch.zeros_like(alpha)
        pt = prior_trend if prior_trend is not None else torch.zeros_like(alpha)
        interact = pm * pt
        
        A, B, C = self.prior_weights[0], self.prior_weights[1], self.prior_weights[2]
        composite_prior = (A * pm) + (B * pt) + (C * interact)
        
        # --- 4. Final Modulation (Residual style) ---
        # Position = tanh( Prior * (1 + alpha) )
        positions = torch.tanh(composite_prior * (1.0 + alpha))
        
        if return_attention:
            return positions, attn_weights
        return positions
