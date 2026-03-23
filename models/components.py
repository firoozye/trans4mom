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
        # Note: if output_dim != hidden_dim, we need a final projection
        if h.shape[-1] != self.output_dim:
            h = nn.Linear(h.shape[-1], self.output_dim)(h)
            
        return self.ln(h + self.skip_fc(x))

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) for feature weighting.
    Determines which features are most relevant at each timestep.
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
        
        # Individual GRNs for each feature
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_vars)
        ])
        
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
        # x shape: (batch, time, num_vars * input_dim) or (batch, time, num_vars, input_dim)
        # Let's assume (batch, time, num_vars, input_dim)
        
        # Flatten for weight calculation
        batch, time, num_vars, input_dim = x.shape
        x_flat = x.view(batch, time, -1)
        
        # Compute weights
        weights = self.weight_grn(x_flat, context) # (batch, time, num_vars)
        weights = torch.softmax(weights, dim=-1).unsqueeze(-1) # (batch, time, num_vars, 1)
        
        # Process individual features
        processed_features = []
        for i in range(self.num_vars):
            processed_features.append(self.feature_grns[i](x[:, :, i, :]))
            
        processed_features = torch.stack(processed_features, dim=2) # (batch, time, num_vars, hidden_dim)
        
        # Weighted sum of processed features
        return (weights * processed_features).sum(dim=2) # (batch, time, hidden_dim)

if __name__ == "__main__":
    # Toy example
    batch, time, num_vars, input_dim = 16, 64, 5, 1
    hidden_dim = 32
    
    x = torch.randn(batch, time, num_vars, input_dim)
    vsn = VariableSelectionNetwork(input_dim, num_vars, hidden_dim)
    output = vsn(x)
    print(f"VSN Output Shape: {output.shape}") # Should be (16, 64, 32)
