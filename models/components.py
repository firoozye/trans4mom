import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) activation.
    GLU(x) = sigma(W1*x + b1) * (W2*x + b2)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2 * input_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        # Split into gate and value
        gate, val = x.chunk(2, dim=-1)
        return torch.sigmoid(gate) * val

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) for feature processing.
    Includes GLU, LayerNorm, and skip connection.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_fc = None
            
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(output_dim)
        
        # Skip connection adjustment if input/output dims differ
        if input_dim != output_dim:
            self.skip_fc = nn.Linear(input_dim, output_dim)
        else:
            self.skip_fc = nn.Identity()
        
        # Gating projection if hidden_dim != output_dim
        if hidden_dim != output_dim:
            self.project_fc = nn.Linear(hidden_dim, output_dim)
        else:
            self.project_fc = nn.Identity()
            
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.context_fc, self.skip_fc, self.project_fc]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # First layer
        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            h = h + self.context_fc(context)
        
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        
        # Gating
        h = self.glu(h)
        
        # Skip connection and LayerNorm
        h = self.project_fc(h)
            
        return self.ln(h + self.skip_fc(x))

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) for feature weighting.
    Optimized for memory by sharing the feature GRN across assets.
    """
    def __init__(
        self,
        input_dim: int,
        num_vars: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        self.num_vars = num_vars
        
        # Shared GRN for all features/assets
        # This drastically reduces parameters and memory for large asset sets
        self.feature_grn = GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
        
        # Global GRN for weighting
        # Flattened features as input
        self.weight_grn = GatedResidualNetwork(
            input_dim * num_vars,
            hidden_dim,
            num_vars,
            dropout,
            context_dim
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (batch, time, num_vars, input_dim)
        num_vars = self.num_vars
        
        if x.dim() == 4:
            batch, time, _, input_dim = x.shape
            x_flat = x.view(batch, time, -1)
        else:
            # Fallback for (batch, time, flattened_features)
            x_flat = x
            batch, time, _ = x.shape
            input_dim = x_flat.shape[-1] // num_vars
            x = x.view(batch, time, num_vars, input_dim)
        
        hidden_dim = self.feature_grn.output_dim
        
        # Prepare context for broadcasting (batch, 1, context_dim) if present
        if context is not None:
            if context.dim() == 2:
                context_vsn = context.unsqueeze(1) 
            else:
                context_vsn = context
        else:
            context_vsn = None

        # 1. Compute variable importance weights
        # weight_grn handles (batch, time, input) + (batch, 1, context) broadcasting
        weights = self.weight_grn(x_flat, context=context_vsn) # (batch, time, num_vars)
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1) # (batch, time, num_vars, 1)
        
        # 2. Process all features using the shared GRN
        x_all = x.reshape(-1, input_dim)
        processed_all = self.feature_grn(x_all) # (batch * time * num_vars, hidden_dim)
        
        # Reshape back: (batch, time, num_vars, hidden_dim)
        processed_features = processed_all.view(batch, time, num_vars, hidden_dim)
        
        # 3. Weighted sum of processed features
        return (weights * processed_features).sum(dim=2) # (batch, time, hidden_dim)

if __name__ == "__main__":
    # Toy example
    batch, time, num_vars, input_dim = 16, 64, 5, 1
    hidden_dim = 32
    
    x = torch.randn(batch, time, num_vars, input_dim)
    vsn = VariableSelectionNetwork(input_dim, num_vars, hidden_dim)
    output = vsn(x)
    print(f"VSN Output Shape: {output.shape}") # Should be (16, 64, 32)
