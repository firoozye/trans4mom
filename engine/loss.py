import torch
import torch.nn as nn
from typing import Optional

class SharpeLoss(nn.Module):
    """
    Optimises for the Negative Sharpe Ratio.
    Includes a transaction cost penalty for turnover.
    """
    def __init__(self, trans_cost: float = 0.0, annualization: float = 252.0):
        super().__init__()
        self.trans_cost = trans_cost
        self.annualization = torch.sqrt(torch.tensor(annualization))

    def forward(
        self,
        positions: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        positions: (batch, time, num_assets) - Predicted weights w_t in [-1, 1]
        returns: (batch, time, num_assets) - Scaled returns r_tilde_t
        """
        # w_{t-1} * r_t
        # We shift positions to align w_{t-1} with r_t
        # Note: If positions at time t are for next period, then w_t * r_{t+1}
        # Standard approach: position at end of t-1 is w_{t-1}, applied to return at t.
        
        # Shift positions forward by 1 (pad with 0 at start)
        shifted_positions = torch.cat([torch.zeros_like(positions[:, :1, :]), positions[:, :-1, :]], dim=1)
        
        # Portfolio returns
        port_returns = shifted_positions * returns
        
        # Transaction costs: C * |w_t - w_{t-1}|
        # We sum across assets for turnover
        turnover = torch.abs(positions - shifted_positions).sum(dim=-1, keepdim=True)
        
        # Net returns
        net_returns = port_returns.sum(dim=-1, keepdim=True) - self.trans_cost * turnover
        
        # Calculate mean and std over the time dimension (dim=1)
        # We calculate per batch item first to maintain signal variance
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
