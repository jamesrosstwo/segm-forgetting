from pathlib import Path
from typing import Generator, List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import torch
from torchmetrics.segmentation import MeanIoU, GeneralizedDiceScore
from tqdm import tqdm
from continuum import SegmentationClassIncremental
from continuum.transforms.segmentation import Resize, ToTensor, Normalize
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt

from file import ROOT_PATH
from util import construct_dataset, construct_scenario, task_id_to_checkpoint_path, construct_model, construct_loader, \
    N_CLASSES


class SegmentationEvaluator:
    def __init__(self, model: nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = model.to(self.device)

    def set_model(self, model: nn.Module):
        self._model = model.to(self.device)
        self._model.eval()

    def _evaluate_single(self, loader: DataLoader):
        self._model.eval()
        progress = tqdm(loader, desc="Evaluating", leave=True)

        num_batches = 0
        miou_class = MeanIoU(num_classes=N_CLASSES, per_class=True, input_format="one-hot").to(self.device)
        miou_mean = MeanIoU(num_classes=N_CLASSES, per_class=False, input_format="one-hot").to(self.device)
        dice_class = GeneralizedDiceScore(num_classes=N_CLASSES, per_class=True, input_format="one-hot").to(self.device)
        dice_mean = GeneralizedDiceScore(num_classes=N_CLASSES, per_class=False, input_format="one-hot").to(self.device)
        acc = 0

        for data, label, task_idx in progress:
            num_batches += 1
            data = data.to(self.device)
            label = label.to(self.device)
            predicted_mask = self._model(data)
            preds_max = torch.argmax(predicted_mask, dim=1)
            preds_onehot = torch.nn.functional.one_hot(preds_max, num_classes=N_CLASSES).permute(0, 3, 1, 2)

            label = label.long()
            label_onehot = torch.nn.functional.one_hot(label, num_classes=N_CLASSES).permute(0, 3, 1, 2)
            unique_values = torch.unique(preds_max)
            # print(f"Unique values in preds_max: {unique_values}")
            # print(f"Unique values in label: {torch.unique(label)}")
            acc += (preds_max == label).float().mean()

            miou_class.update(preds_onehot, label_onehot)
            miou_mean.update(preds_onehot, label_onehot)
            dice_class.update(preds_onehot, label_onehot)
            dice_mean.update(preds_onehot, label_onehot)

        acc = acc.item() / num_batches
        miou_mean = miou_mean.compute().item()
        miou_class = miou_class.compute().tolist()
        dice_mean = dice_mean.compute().item()
        dice_class = dice_class.compute().tolist()

        return acc, miou_mean, miou_class, dice_mean, dice_class


    def evaluate_scenario(self, scenario) -> Generator[pd.DataFrame, None, None]:
        for eval_task_id, eval_taskset in enumerate(tqdm(scenario, desc="Evaluating across all tasks")):
            self._model.eval()
            acc, miou_mean, miou_class, dice_mean, dice_class = self._evaluate_single(
                construct_loader(eval_taskset, batch_size=4, shuffle=False)
            )

            task_metrics = {
                "task_id": eval_task_id,
                "accuracy": acc,
                "miou_mean": miou_mean,
                "dice_mean": dice_mean,
            }

            for i, val in enumerate(miou_class):
                task_metrics[f"miou_class_{i}"] = val
            for i, val in enumerate(dice_class):
                task_metrics[f"dice_class_{i}"] = val

            yield pd.DataFrame([task_metrics])


class ExperimentEvaluator:
    def __init__(
            self,
            dataset: DictConfig,
            model: DictConfig,
            checkpoints_path: str,
            use_training_set: bool = False,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._checkpoints_path = ROOT_PATH / checkpoints_path
        self._dataset = construct_dataset(dataset, train=use_training_set)
        self._model_cfg = model
        self._n_tasks = 5
        self._eval_path = self._checkpoints_path / "eval"
        self._eval_path.mkdir(exist_ok=True)

    @property
    def checkpoint_paths(self) -> List[Path]:
        return [task_id_to_checkpoint_path(self._checkpoints_path, i) for i in range(self._n_tasks)]

    def _metrics_path_from_idx(self, idx):
        return self._eval_path / f"metrics_model_{idx}.csv"

    def evaluate_tasks(self):
        for checkpoint_path, model_idx in zip(self.checkpoint_paths, range(self._n_tasks)):
            segm_model = construct_model(self._model_cfg)
            segm_model.load_state_dict(torch.load(checkpoint_path))
            scenario, task_classes = construct_scenario(self._dataset)
            evaluator = SegmentationEvaluator(segm_model)
            metrics_out_path = self._metrics_path_from_idx(model_idx)

            all_metrics = []
            for eval_task_idx, metrics_df in enumerate(evaluator.evaluate_scenario(scenario)):
                all_metrics.append(metrics_df)
            combined_metrics = pd.concat(all_metrics, ignore_index=True)
            combined_metrics.to_csv(metrics_out_path, index=False)

    def cl_metrics(self, metrics_df):
        task_ids = metrics_df['task_id'].unique()

        num_tasks = len(task_ids)
        accuracies = np.zeros((num_tasks, num_tasks))

        for i, task_id in enumerate(task_ids):
            accuracies[i, :i + 1] = metrics_df[metrics_df['task_id'] == task_id]['accuracy'].values[:i + 1]

        forgetting = []
        for t in range(num_tasks - 1):
            best_acc = np.max(accuracies[:num_tasks, t])
            last_acc = accuracies[-1, t]
            forgetting.append(best_acc - last_acc)

        bwt_values = []
        for t in range(1, num_tasks):
            bwt_values.append(np.mean(accuracies[t - 1, :t] - accuracies[t, :t]))

        forgetting = np.mean(forgetting)
        bwt = np.mean(bwt_values)
        return forgetting, bwt

    def analyze(self):
        for model_idx in range(self._n_tasks):
            df = pd.read_csv(self._metrics_path_from_idx(model_idx))
            forgetting, bwt = self.cl_metrics(df)

    def run_gradcam(self):
        segm_model = construct_model(self._model_cfg).to(self.device)
        segm_model.load_state_dict(torch.load('../experiments/none_resnet18_2024-12-05_18-10-10/model_task_0.pth'))
        segm_model.train()
        for layer in segm_model.encoder.children():
            for param in layer.parameters():
                param.requires_grad = True
        

        
        scenario = SegmentationClassIncremental(
            self._dataset,
            nb_classes=20,
            initial_increment=15, increment=1,
            mode="overlap",
            transformations=[Resize((512, 512)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]
        )
        for eval_task_id, eval_taskset in enumerate(scenario):
            data_loader = DataLoader(eval_taskset, batch_size=1, shuffle=False)
            for data, label, task_idx in data_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                if 15 in label:
                    plt.imshow(label[0].squeeze(0).detach().cpu().numpy())
                    plt.show()
                else:
                    continue
                #target_layer = segm_model.encoder.layer1[-1]
                target_layer = segm_model.decoder.blocks[-1].conv2[0]
                cam_extractor = SmoothGradCAMpp(segm_model, target_layer=target_layer)
                # with SmoothGradCAMpp(segm_model) as cam_extractor:
                # Preprocess your data and feed it to the model
                out = segm_model(data)
                # Retrieve the CAM by passing the class index and the model output
                activation_map = cam_extractor(15, out)
                plt.imshow(activation_map[0].squeeze(0).detach().cpu().numpy())
                plt.axis('off')
                plt.tight_layout()
                plt.show()

                # predicted_mask = segm_model(data)
                # print(predicted_mask.shape)
        # for data, label, task_idx in data_loader:
        #     data = data.to(self.device)
        #     label = label.to(self.device)
        #     predicted_mask = segm_model(data)
        #     print(predicted_mask.shape)
        print("good")

@hydra.main(version_base=None, config_path="../config", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    cfg = dict(cfg)
    cfg.pop('evaluator', None)
    cfg.pop('trainer', None)
    cfg.pop('dataset_val', None)
    cfg.pop('wandb_name', None)
    cfg.pop('experiment_name', None)
    cfg.pop('debug', None)
    cfg['checkpoints_path'] = ''
    eval = ExperimentEvaluator(**cfg)
    eval.run_gradcam()
    # eval.evaluate_tasks()
    # eval.analyze()


if __name__ == "__main__":
    main()
