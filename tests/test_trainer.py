import torch
import pytest
import os
from engine.trainer import Trainer, run_training_job
from models.architecture import MomentumTransformer
from engine.loss import SharpeLoss

def test_trainer_single_step():
    """Verify that a single training step reduces loss or at least runs."""
    device = torch.device("cpu")
    model = MomentumTransformer(1, 5, 16, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = SharpeLoss()
    trainer = Trainer(model, optimizer, loss_fn, device)
    
    # Toy data
    x = torch.randn(4, 10, 5, 1)
    y = torch.randn(4, 10, 1)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=2
    )
    
    initial_loss = trainer.train_epoch(train_loader)
    second_loss = trainer.train_epoch(train_loader)
    
    # In a single step with random data it might not always decrease, 
    # but it should return a valid float.
    assert isinstance(initial_loss, float)
    assert isinstance(second_loss, float)

def test_run_training_job_local():
    """Verify the high-level entry point works locally."""
    hparams = {
        'input_dim': 1,
        'num_vars': 2,
        'hidden_dim': 8,
        'num_heads': 2,
        'num_assets': 1,
        'lr': 1e-3,
        'trans_cost': 0.0,
        'batch_size': 2,
        'epochs': 1
    }
    x = torch.randn(4, 5, 2, 1)
    y = torch.randn(4, 5, 1)
    
    # Should run without error
    run_training_job(x, y, hparams=hparams)
    assert os.path.exists("weights/last_model.pt")
