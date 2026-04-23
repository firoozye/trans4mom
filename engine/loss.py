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
        # ... (previous code for shifted_positions and port_returns)
        shifted_positions = torch.cat([torch.zeros_like(positions[:, :1, :]), positions[:, :-1, :]], dim=1)
        port_returns = shifted_positions * returns
        
        # Transaction costs calculation
        diff = positions - shifted_positions
        if spreads is not None:
            costs = self.trans_cost + (spreads / 2.0)
            trading_costs = (torch.abs(diff) * costs).sum(dim=-1, keepdim=True)
        else:
            trading_costs = self.trans_cost * torch.abs(diff).sum(dim=-1, keepdim=True)
        
        # 1. Smoothing Penalty (Quadratic Turnover)
        smooth_penalty = self.smoothing_beta * torch.pow(diff, 2).sum(dim=-1, keepdim=True)
        
        # 2. Bias Penalty (Encourage Market Neutrality / Shorting)
        # Penalizes the absolute mean position over the sequence
        bias_penalty = self.bias_beta * torch.abs(torch.mean(positions, dim=1, keepdim=True))
        
        # Net returns including smoothing and bias penalties
        # We subtract these from returns to lower the Sharpe if the model is 'cheating' or 'lumpy'
        net_returns = port_returns.sum(dim=-1, keepdim=True) - trading_costs - smooth_penalty - bias_penalty
        
        # Calculate mean and std
        mean_ret = torch.mean(net_returns, dim=1)
        std_ret = torch.std(net_returns, dim=1, unbiased=False) + 1e-6
        
        # Negative Sharpe Ratio per batch item
        sharpe = (mean_ret / std_ret) * self.annualization.to(returns.device)
        
        # Return negative average Sharpe as loss
        return -torch.mean(sharpe)

if __name__ == "__main__":
    # Toy example
    batch, time, num_assets = 32, 252, 10
    pos = torch.tanh(torch.randn(batch, time, num_assets)) # Random positions
    rets = torch.randn(batch, time, num_assets) # Random returns
    
    loss_fn = SharpeLoss(trans_cost=0.001)
    loss = loss_fn(pos, rets)
    print(f"Negative Sharpe Loss: {loss.item()}")
