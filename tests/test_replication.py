import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.momentum_transformer import WoodMomentumTransformer, MomentumTransformer
from engine.loss import SharpeLoss
from data.datasets import EXHIBIT_11_ASSETS, custom_collate

def test_parameter_counts():
    m_base = WoodMomentumTransformer(include_cpd=False)
    base_params = sum(p.numel() for p in m_base.parameters())
    assert base_params == 380462, f"Expected 380,462 params, got {base_params:,}"

    m_cpd = WoodMomentumTransformer(include_cpd=True)
    cpd_params = sum(p.numel() for p in m_cpd.parameters())
    assert cpd_params == 425016, f"Expected 425,016 params, got {cpd_params:,}"

def test_forward_pass_baseline():
    m = WoodMomentumTransformer(include_cpd=False)
    b, t = 2, 252
    feats = {k: torch.randn(b, t, 1) for k in m.feature_names}
    aids = torch.tensor([0, 1], dtype=torch.long)
    pos = m(feats, aids)
    assert pos.shape == (b, t, 1)
    assert (pos >= -1.0).all() and (pos <= 1.0).all()

def test_forward_pass_cpd():
    m = WoodMomentumTransformer(include_cpd=True)
    b, t = 2, 252
    feats = {k: torch.randn(b, t, 1) for k in m.feature_names}
    aids = torch.tensor([2, 3], dtype=torch.long)
    pos = m(feats, aids)
    assert pos.shape == (b, t, 1)
    assert (pos >= -1.0).all() and (pos <= 1.0).all()

def test_loss_function():
    loss_fn = SharpeLoss(trans_cost=0.0001, annualization=252.0)
    b, t = 2, 252
    pos = torch.ones(b, t, 1)
    y = torch.randn(b, t, 1)
    loss = loss_fn(pos, y)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

def test_exhibit_11_universe():
    assert len(EXHIBIT_11_ASSETS) == 50, f"Expected 50 canonical assets, got {len(EXHIBIT_11_ASSETS)}"
    # Verify categories: 0=CM, 1=EQ, 2=FI, 3=FX
    classes = set(EXHIBIT_11_ASSETS.values())
    assert classes == {0, 1, 2, 3}

if __name__ == "__main__":
    test_parameter_counts()
    test_forward_pass_baseline()
    test_forward_pass_cpd()
    test_loss_function()
    test_exhibit_11_universe()
    print("ALL REPLICATION TESTS PASSED SUCCESSFULLY!")
