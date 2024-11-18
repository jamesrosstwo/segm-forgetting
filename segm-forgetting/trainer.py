import importlib

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader


class SegmentationTrainer:
    def __init__(self, model: nn.Module, loss: str, optimizer: DictConfig):
        self._model = model
        module_name, func_name = loss.rsplit(".", 1)
        module = importlib.import_module(module_name)
        self._loss_function = getattr(module, func_name)
        self._optimizer = instantiate(optimizer, params=model.parameters())

    def train_model_on_task(self, loader: DataLoader):
        for data, label, task_idx in loader:
            predicted_mask = self._model(data)
            loss = self._loss_function(predicted_mask, label)
            loss.backward()
            self._optimizer.step()
