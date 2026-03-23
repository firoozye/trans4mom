import torch
import pytest
from models.architecture import MomentumTransformer

def test_transformer_shapes():
    """Verify output shapes of the full architecture."""
    batch, time, num_vars, input_dim = 16, 126, 5, 1
    hidden_dim, num_heads = 32, 4
    
    model = MomentumTransformer(input_dim, num_vars, hidden_dim, num_heads)
    x = torch.randn(batch, time, num_vars, input_dim)
    
    out = model(x)
    assert out.shape == (batch, time, 1)
    # Output values should be in [-1, 1] due to Tanh
    assert (out >= -1.0).all() and (out <= 1.0).all()

def test_transformer_multi_asset_shapes():
    """Verify shapes for multi-asset outputs."""
    batch, time, num_vars, input_dim = 8, 50, 5, 1
    hidden_dim, num_heads, out_dim = 16, 2, 10 # 10 assets
    
    model = MomentumTransformer(input_dim, num_vars, hidden_dim, num_heads, output_dim=out_dim)
    x = torch.randn(batch, time, num_vars, input_dim)
    
    out = model(x)
    assert out.shape == (batch, time, out_dim)

def test_transformer_gradients():
    """Verify that gradients flow from output to input."""
    batch, time, num_vars, input_dim = 2, 10, 5, 1
    model = MomentumTransformer(input_dim, num_vars, 16, 2)
    x = torch.randn(batch, time, num_vars, input_dim, requires_grad=True)
    
    out = model(x)
    loss = out.mean()
    loss.backward()
    
    assert x.grad is not None
    # Check if a specific layer has gradients
    assert model.lstm.weight_ih_l0.grad is not None
