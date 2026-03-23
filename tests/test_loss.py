import torch
import pytest
from engine.loss import SharpeLoss

def test_sharpe_loss_direction():
    """Verify that higher returns lead to lower loss (more negative)."""
    batch, time, num_assets = 1, 100, 1
    loss_fn = SharpeLoss(trans_cost=0.0)
    
    # Positive returns
    pos1 = torch.ones(batch, time, num_assets)
    rets1 = torch.ones(batch, time, num_assets) * 0.01 # Always positive returns
    loss1 = loss_fn(pos1, rets1)
    
    # Zero returns
    rets2 = torch.zeros(batch, time, num_assets)
    loss2 = loss_fn(pos1, rets2)
    
    assert loss1 < loss2, f"Loss for positive returns ({loss1}) should be less than for zero returns ({loss2})"

def test_trans_cost_penalty():
    """Verify that transaction costs increase the loss."""
    batch, time, num_assets = 1, 100, 1
    # Constant positions (no turnover)
    pos_static = torch.ones(batch, time, num_assets)
    # Changing positions (high turnover)
    pos_dynamic = torch.zeros(batch, time, num_assets)
    pos_dynamic[:, ::2, :] = 1.0 # 0, 1, 0, 1...
    
    rets = torch.ones(batch, time, num_assets) * 0.01
    
    # Compare with vs without cost
    loss_no_cost = SharpeLoss(trans_cost=0.0)(pos_dynamic, rets)
    loss_with_cost = SharpeLoss(trans_cost=0.1)(pos_dynamic, rets)
    
    # More cost should mean higher loss (less negative or more positive)
    assert loss_with_cost > loss_no_cost

def test_sharpe_loss_annualization():
    """Check if annualization factor is applied."""
    loss_fn = SharpeLoss(annualization=252.0)
    # Sharpe = (mean / std) * sqrt(252)
    # If mean=0.01, std=0.01, annualization=sqrt(252) approx 15.87
    # Loss should be -15.87
    
    batch, time = 1, 1000
    rets = torch.randn(batch, time, 1) * 0.01 + 0.01 # mean 0.01, std 0.01
    pos = torch.ones(batch, time, 1)
    
    loss = loss_fn(pos, rets)
    # Expected approx -15.87
    assert loss < -10.0
