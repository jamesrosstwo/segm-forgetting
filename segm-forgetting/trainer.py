import importlib

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import torch

class SegmentationTrainer:
    def __init__(self, model: nn.Module, loss: str, optimizer: DictConfig):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = model.to(self.device)
        module_name, func_name = loss.rsplit(".", 1)
        module = importlib.import_module(module_name)
        self._loss_function = getattr(module, func_name)()
        self._optimizer = instantiate(optimizer, params=model.parameters())

    def train_model_on_task(self, loader: DataLoader):
        self._model.train()
        progress = tqdm.tqdm(loader, desc="Training", leave=True)
        total_loss = 0
        num_batches = 0
        for data, label, task_idx in progress:
            num_batches += 1
            try:
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
                
                progress.set_postfix(avg_loss=loss.item())
            except OSError as e:
                pass
            