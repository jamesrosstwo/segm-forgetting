import importlib
import math
from typing import List, Set

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import torch
import torch.nn as nn
import wandb
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore

class SegmentationTrainer:
    def __init__(self, model: nn.Module, loss: str, optimizer: DictConfig, n_base_epochs: int=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = model.to(self.device)
        self._loss_function = nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor([0.01 if x == 0 else 1.0 for x in range(21)]).to(self.device))
        self._optimizer = instantiate(optimizer, params=model.parameters())
        self._seen_classes = set()
        self._n_base_classes = None
        self._n_base_epochs = n_base_epochs

    def train_model_on_task(self, loader: DataLoader, task_classes: List[int]):
        n_novel_classes = len(task_classes) - len(self._seen_classes)
        self._seen_classes = set(task_classes).union(self._seen_classes)
        if self._n_base_classes is None:
            self._n_base_classes = n_novel_classes
        n_epochs = math.ceil(n_novel_classes / self._n_base_classes * self._n_base_epochs)
        self._model.train()
        miou_class = MeanIoU(num_classes=21, per_class=True, input_format="index").to(self.device)
        wandb.log({"test": 1})
        for epoch in range(10):
            progress = tqdm.tqdm(loader, desc=f"Training epoch {epoch}/{n_epochs} on {self.device}", leave=True)
            total_loss = 0
            num_batches = 0
            # Mask out class predictions for classes not in this task by setting output logits to -inf
            class_mask = torch.full((256,), -1e10, device=self.device)
            class_mask[task_classes] = 0

            for data, label, task_idx in progress:
                num_batches += 1

                data = data.to(self.device)
                label = label.to(self.device)
                predicted_mask = self._model(data)
                masked_logits = predicted_mask + class_mask.view(1, -1, 1, 1)
                # Crossentropy expects long labels
                label = label.long()
                loss = self._loss_function(masked_logits, label)
                loss.backward()
                self._optimizer.step()
                label[label == 255] = 0
                miou_class.update(preds_max, label)
                
                # Print average loss metric
                total_loss += loss.item()
            
                progress.set_postfix(avg_loss=loss.item()/num_batches)

                # Log the average loss to wandb
            wandb.log({"avg_loss": total_loss / num_batches})
            for i in range(21):
                wandb.log({f"miou_class_{i}": miou_class.compute()[i]})
