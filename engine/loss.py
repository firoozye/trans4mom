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
        spreads: Optional[torch.Tensor] = None,
        vol_scales: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        positions: (batch, time, 1) - Action at time t
        returns: (batch, time, 1) - Returns at time t
        """
        # Earned returns: pos[t] * returns[t+1]
        active_pos = positions[:, :-1, :]
        active_rets = returns[:, 1:, :]
        port_returns = (active_pos * active_rets).sum(dim=-1, keepdim=True)
        
        # Scale positions to actual traded sizing for physical transaction cost calculation
        if vol_scales is not None:
            physical_pos = positions * vol_scales
        else:
            physical_pos = positions

        # Linear Transaction costs: cost * |pos[t] - pos[t-1]|
        shifted_positions = torch.cat([torch.zeros_like(physical_pos[:, :1, :]), physical_pos[:, :-1, :]], dim=1)
        pos_diff = torch.abs(physical_pos - shifted_positions)
        
        if spreads is not None:
            # spreads are per-asset, average them for the portfolio
            costs = self.trans_cost + (spreads / 2.0)
            trading_costs = (pos_diff * costs).sum(dim=-1, keepdim=True)
        else:
            trading_costs = self.trans_cost * pos_diff.sum(dim=-1, keepdim=True)
            
        # Align costs with returns (index 1 onwards)
        trading_costs = trading_costs[:, 1:, :]
        
        # Net returns
        net_returns = port_returns - trading_costs
        
        # POOLED SHARPE (V7 FIX): Calculate mean and std across the entire batch
        # This is much more stable and represents the total strategy performance
        mean_ret = torch.mean(net_returns)
        std_ret = torch.std(net_returns, unbiased=False) + 1e-6
        
        # Negative Sharpe Ratio for the whole batch
        sharpe = (mean_ret / std_ret) * self.annualization.to(returns.device)
        
        # We minimize Negative Sharpe
        return -sharpe

if __name__ == "__main__":
    # Toy example
    batch, time, num_assets = 32, 252, 10
    pos = torch.tanh(torch.randn(batch, time, num_assets)) # Random positions
    rets = torch.randn(batch, time, num_assets) # Random returns
    
    loss_fn = SharpeLoss(trans_cost=0.001)
    loss = loss_fn(pos, rets)
    print(f"Negative Sharpe Loss: {loss.item()}")
