import importlib

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import torch
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
import matplotlib.pyplot as plt

class SegmentationEvaluator:
    def __init__(self, model: nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = model.to(self.device)
        
    def set_model(self, model: nn.Module):
        self._model = model.to(self.device)
        self._model.eval()

    def evaluate(self, loader: DataLoader):
        self._model.eval()
        progress = tqdm.tqdm(loader, desc="Evaluating", leave=True)

        num_batches = 0
        miou_class = MeanIoU(num_classes=256, per_class=True, input_format="one-hot").to(self.device)
        miou_mean = MeanIoU(num_classes=256, per_class=False, input_format="one-hot").to(self.device)
        dice_class = GeneralizedDiceScore(num_classes=256, per_class=True, input_format="one-hot").to(self.device)
        dice_mean = GeneralizedDiceScore(num_classes=256, per_class=False, input_format="one-hot").to(self.device)
        acc = 0
        for data, label, task_idx in progress:
            num_batches += 1
            data = data.to(self.device)
            label = label.to(self.device)
            predicted_mask = self._model(data)
            preds_max = torch.argmax(predicted_mask, dim=1)
            preds_onehot = torch.nn.functional.one_hot(preds_max, num_classes=256).permute(0, 3, 1, 2)

            # Crossentropy expects long labels
            label = label.long()
            label_onehot = torch.nn.functional.one_hot(label, num_classes=256).permute(0, 3, 1, 2)
            unique_values = torch.unique(preds_max)
            # print(f"Unique values in preds_max: {unique_values}")
            # print(f"Unique values in label: {torch.unique(label)}")
            acc += (preds_max == label).float().mean()
            miou_class.update(preds_onehot, label_onehot)
            miou_mean.update(preds_onehot, label_onehot)
            dice_class.update(preds_onehot, label_onehot)
            dice_mean.update(preds_onehot, label_onehot)

        return acc / num_batches, miou_mean.compute(), miou_class.compute(), dice_mean.compute(), dice_class.compute()