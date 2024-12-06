import importlib

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import torch
import torch.nn as nn
import wandb
class SegmentationTrainer:
    def __init__(self, model: nn.Module, loss: str, optimizer: DictConfig):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = model.to(self.device)
        self._loss_function = nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor([0.05 if x == 0 else 1.0 for x in range(256)]).to(self.device))
        self._optimizer = instantiate(optimizer, params=model.parameters())

    def train_model_on_task(self, loader: DataLoader):
        self._model.train()
        for epoch in range(4):
            progress = tqdm.tqdm(loader, desc="Training", leave=True)
            total_loss = 0
            num_batches = 0
            for data, label, task_idx in progress:
                num_batches += 1

                data = data.to(self.device)
                label = label.to(self.device)
                predicted_mask = self._model(data)

                # Crossentropy expects long labels
                label = label.long()
                loss = self._loss_function(predicted_mask, label)
                loss.backward()
                self._optimizer.step()

                # Print average loss metric
                total_loss += loss.item()
            
                progress.set_postfix(avg_loss=loss.item()/num_batches)

                # Log the average loss to wandb
            wandb.log({"avg_loss": total_loss / num_batches})