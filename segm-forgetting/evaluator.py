import importlib

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import torch
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore

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
        miou_class = MeanIoU(num_classes=256, per_class=True, input_format="index").to(self.device)
        miou_mean = MeanIoU(num_classes=256, per_class=False, input_format="index").to(self.device)
        dice_class = GeneralizedDiceScore(num_classes=256, per_class=True, input_format="index").to(self.device)
        dice_mean = GeneralizedDiceScore(num_classes=256, per_class=False, input_format="index").to(self.device)
        acc = 0
        for data, label, task_idx in progress:
            num_batches += 1
            try:
                data = data.to(self.device)
                label = label.to(self.device)
                predicted_mask = self._model(data)

                # Crossentropy expects long labels
                label = label.long()
                
                preds_max = torch.argmax(predicted_mask, dim=1)
                
                acc += (preds_max == label).float().mean()
                miou_class.update(preds_max, label)
                miou_mean.update(preds_max, label)
                dice_class.update(preds_max, label)
                dice_mean.update(preds_max, label)
                
            except OSError as e:
                pass

        return acc / num_batches, miou_mean.compute(), miou_class.compute(), dice_mean.compute(), dice_class.compute()