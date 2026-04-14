import torch
import torch.nn as nn
from models.components import VariableSelectionNetwork, GatedResidualNetwork
from typing import Optional, List

class MomentumTransformer(nn.Module):
    """
    Momentum Transformer Architecture.
    Combines VSN for feature selection, LSTM for sequential dependency,
    and Multi-head Attention for global context.
    """
    def __init__(
        self,
        input_dim: int,
        num_vars: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        output_dim: int = 1 # Single asset weight by default
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Variable Selection Network
        self.vsn = VariableSelectionNetwork(input_dim, num_vars, hidden_dim, dropout)
        
        # 2. LSTM Layer (Temporal Processing)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 3. Static Gating / Skip connection (GRN)
        self.post_lstm_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # 4. Multi-head Attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 5. Output Layer (Position Sizing)
        self.output_layer = nn.Sequential(
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh() # Position in [-1, 1]
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> torch.Tensor:
        """
        x: (batch, time, num_vars, input_dim)
        """
        # Feature Selection
        vsn_out = self.vsn(x) # (batch, time, hidden_dim)
        
        # Sequential Processing
        lstm_out, _ = self.lstm(vsn_out) # (batch, time, hidden_dim)
        
        # Gating
        gated_out = self.post_lstm_grn(lstm_out) # (batch, time, hidden_dim)
        
        # Attention (Self-Attention)
        # For time series, we MUST use causal masking in attention to prevent look-ahead bias
        seq_len = gated_out.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=gated_out.device), diagonal=1).bool()
        
        attn_out, attn_weights = self.attention(
            gated_out, gated_out, gated_out, 
            key_padding_mask=mask,
            attn_mask=causal_mask
        )
        
        # Final output
        positions = self.output_layer(attn_out) # (batch, time, output_dim)
        
        if return_attention:
            return positions, attn_weights
        return positions

if __name__ == "__main__":
    # Toy test
    batch, time, num_vars, input_dim = 8, 126, 5, 1
    model = MomentumTransformer(input_dim, num_vars, 64, 4)
    x = torch.randn(batch, time, num_vars, input_dim)
    out = model(x)
    print(f"Model Output Shape: {out.shape}") # (8, 126, 1)
