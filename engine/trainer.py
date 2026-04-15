import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from models.architecture import MomentumTransformer
from engine.loss import SharpeLoss
from tqdm import tqdm
from typing import Optional, Dict

class Trainer:
    """
    Unified Trainer for Momentum Transformer.
    Supports Local (CPU/GPU) and HPC (DDP) training.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        is_distributed: bool = False
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.is_distributed = is_distributed
        
        if is_distributed:
            self.model = DDP(self.model, device_ids=[device.index] if device.index is not None else None)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            # x: (batch, time, num_vars, input_dim)
            # y: (batch, time, num_assets) - scaled returns for Sharpe
            
            output_pos = self.model(x)
            loss = self.loss_fn(output_pos, y)
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                output_pos = self.model(x)
                loss = self.loss_fn(output_pos, y)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def save_checkpoint(self, path: str):
        # Handle DDP model saving
        state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
        torch.save(state_dict, path)

def run_training_job(
    train_data: torch.Tensor,
    train_returns: torch.Tensor,
    val_data: Optional[torch.Tensor] = None,
    val_returns: Optional[torch.Tensor] = None,
    hparams: Dict = None
):
    """Entry point for a training job (local or HPC)."""
    # 1. Setup device and DDP
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.distributed.init_process_group(backend="nccl")
        is_dist = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_dist = False

    # 2. Build Model
    model = MomentumTransformer(
        input_dim=hparams['input_dim'],
        num_vars=hparams['num_vars'],
        hidden_dim=hparams['hidden_dim'],
        num_heads=hparams['num_heads'],
        output_dim=hparams['num_assets']
    )
    
    # 3. Setup Engine
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'])
    loss_fn = SharpeLoss(
        trans_cost=hparams['trans_cost'], 
        annualization=hparams.get('annualization', 252.0)
    )
    trainer = Trainer(model, optimizer, loss_fn, device, is_dist)
    
    # 4. DataLoaders
    train_set = TensorDataset(train_data, train_returns)
    train_loader = DataLoader(train_set, batch_size=hparams['batch_size'], shuffle=True)
    
    # 5. Training Loop
    best_val_loss = float('inf')
    os.makedirs("weights", exist_ok=True)
    
    # Create timestamp for this run
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"model_{run_id}.pt"
    
    for epoch in range(hparams['epochs']):
        train_loss = trainer.train_epoch(train_loader)
        
        # Only log/save on rank 0 if distributed
        if local_rank in [-1, 0]:
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}")
            trainer.save_checkpoint(os.path.join("weights", model_name))
            # Also save a 'last_model.pt' for the visualization scripts
            trainer.save_checkpoint(os.path.join("weights", "last_model.pt"))
            
    if is_dist:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Create weights dir if not exists
    os.makedirs("weights", exist_ok=True)
    
    # Simple toy run for local verification
    toy_hparams = {
        'input_dim': 1,
        'num_vars': 5,
        'hidden_dim': 32,
        'num_heads': 4,
        'num_assets': 1,
        'lr': 1e-3,
        'trans_cost': 0.001,
        'batch_size': 8,
        'epochs': 2
    }
    
    # Random toy tensors
    x = torch.randn(16, 64, 5, 1)
    y = torch.randn(16, 64, 1)
    
    print("Starting toy local training...")
    run_training_job(x, y, hparams=toy_hparams)
    print("Toy training complete.")
