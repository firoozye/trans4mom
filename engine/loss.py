import torch
import torch.nn as nn
from typing import Optional

class SharpeLoss(nn.Module):
    """
    Optimises for the Negative Sharpe Ratio.
    Includes a transaction cost penalty for turnover.
    """
    def __init__(self, trans_cost: float = 0.0, annualization: float = 252.0, smoothing_beta: float = 0.01, bias_beta: float = 0.01):
        super().__init__()
        self.trans_cost = trans_cost
        self.smoothing_beta = smoothing_beta
        self.bias_beta = bias_beta
        self.annualization = torch.sqrt(torch.tensor(annualization))

    def forward(
        self,
        positions: torch.Tensor,
        returns: torch.Tensor,
        spreads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        positions: (batch, time, num_assets) - Action at time t
        returns: (batch, time, num_assets) - Returns at time t
        """
        # CRITICAL FIX: To avoid lookahead bias, positions taken at time t 
        # must be multiplied by returns earned from t to t+1.
        # If 'returns' at index i is the move from i-1 to i, then positions[i] 
        # earns returns[i+1].
        
        # Earned returns: pos[t] * returns[t+1]
        active_pos = positions[:, :-1, :]
        active_rets = returns[:, 1:, :]
        port_returns = active_pos * active_rets
        
        # Transaction costs calculation (based on change in position)
        # diff[t] = pos[t] - pos[t-1]
        shifted_positions = torch.cat([torch.zeros_like(positions[:, :1, :]), positions[:, :-1, :]], dim=1)
        diff = positions - shifted_positions
        
        if spreads is not None:
            costs = self.trans_cost + (spreads / 2.0)
            trading_costs = (torch.abs(diff) * costs).sum(dim=-1, keepdim=True)
        else:
            trading_costs = self.trans_cost * torch.abs(diff).sum(dim=-1, keepdim=True)
            
        # Align costs with returns (index 1 onwards)
        trading_costs = trading_costs[:, 1:, :]
        
        # 1. Smoothing Penalty (Quadratic Turnover)
        smooth_penalty = self.smoothing_beta * torch.pow(diff, 2).sum(dim=-1, keepdim=True)
        smooth_penalty = smooth_penalty[:, 1:, :]
        
        # 2. Bias Penalty (Encourage Market Neutrality / Shorting)
        bias_penalty = self.bias_beta * torch.abs(torch.mean(positions[:, :-1, :], dim=1, keepdim=True))
        
        # Net returns
        net_returns = port_returns.sum(dim=-1, keepdim=True) - trading_costs - smooth_penalty - bias_penalty
        
        # Calculate mean and std
        mean_ret = torch.mean(net_returns, dim=1)
        std_ret = torch.std(net_returns, dim=1, unbiased=False) + 1e-6
        
        # Negative Sharpe Ratio per batch item
        sharpe = (mean_ret / std_ret) * self.annualization.to(returns.device)
        
        return -torch.mean(sharpe)

if __name__ == "__main__":
    # Toy example
    batch, time, num_assets = 32, 252, 10
    pos = torch.tanh(torch.randn(batch, time, num_assets)) # Random positions
    rets = torch.randn(batch, time, num_assets) # Random returns
    
    loss_fn = SharpeLoss(trans_cost=0.001)
    loss = loss_fn(pos, rets)
    print(f"Negative Sharpe Loss: {loss.item()}")
