from datetime import datetime
from pathlib import Path
from typing import Tuple

from continuum import SegmentationClassIncremental
from continuum.transforms.segmentation import Resize, ToTensor, Normalize

import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from file import EXPERIMENTS_PATH, check_create_dir, DATA_PATH
from trainer import SegmentationTrainer
from evaluator import SegmentationEvaluator

import pandas as pd
import os

from util import get_date_string, construct_dataset, construct_model, construct_evaluator, construct_trainer, \
    construct_scenario, task_id_to_checkpoint_path


class Experiment:
    def __init__(
            self,
            experiment_name: str,
            wandb_name: str,
            dataset: DictConfig,
            model: DictConfig,
            trainer: DictConfig,
            should_cache_results: bool = True,
            debug: bool = False
    ):
        self._experiment_name = experiment_name
        self._wandb_project_name = wandb_name
        self._data_conf = dataset
        self._model_conf = model
        self._exp_instance_name = f"{self.name}_{self._model_conf.encoder_name}_{get_date_string()}"
        self._should_cache_results = should_cache_results
        self._out_path, self._out_cache_dir = self._make_output_dirs()
        self._debug_mode = debug

        self._init_wandb()
        self._dataset = construct_dataset(self._data_conf, train=True)
        self._segm_model = construct_model(self._model_conf)
        self._trainer: SegmentationTrainer = construct_trainer(trainer, self._segm_model)
        self._freeze_encoder_weights()

    def _make_output_dirs(self):
        out_path = EXPERIMENTS_PATH / self._exp_instance_name
        out_dir = check_create_dir(str(out_path))
        latest_path = EXPERIMENTS_PATH / "latest"
        latest_path.unlink(missing_ok=True)
        latest_path.symlink_to(out_dir, target_is_directory=True)

        base_cache_dir = check_create_dir(str(EXPERIMENTS_PATH / "cache"))
        out_cache_dir = check_create_dir(str(base_cache_dir / self.name))
        return out_dir, out_cache_dir

    def _freeze_encoder_weights(self):
        for layer in self._segm_model.encoder.children():
            for param in layer.parameters():
                param.requires_grad = False
        return

    def _cache_path(self, exp_resource_path: Path, target_is_directory: bool) -> Tuple[Path, bool]:
        """
        Converts `exp_resource_path` to a symlink, which points to the path of the same name relative to the
        shared outputs directory for this experiment type. Allows module authors to cache results between experiments.
        :param target_is_directory: If sharing a directory, set this to true.
        :param exp_resource_path: Path relative the module output directory.
        :return: The resolved symlink path, and whether the specified path already exists within the cache.
        """

        _error_st = f"Should not attempt to cache paths while cache_results is False for module {self.name}."
        assert self._should_cache_results, _error_st
        relative_resource_path = exp_resource_path.relative_to(self._out_path)
        shared_resource_path = self._out_cache_dir / relative_resource_path
        shared_resource_path.parent.mkdir(exist_ok=True, parents=True)
        exp_resource_path.parent.mkdir(exist_ok=True, parents=True)
        exp_resource_path.symlink_to(shared_resource_path, target_is_directory=target_is_directory)
        out_path = exp_resource_path.resolve()
        return out_path, out_path.exists()

    def _init_wandb(self):
        run_id = wandb.util.generate_id()
        print("Creating new wandb run id: {}".format(run_id))
        wandb.init(
            id=run_id,
            dir=str(self._out_path),
            project=self._wandb_project_name,
            name=f"{self._exp_instance_name}",
            config={
                "dataset": OmegaConf.to_container(self._data_conf),
                "model": OmegaConf.to_container(self._model_conf)
            }
        )

    @property
    def name(self):
        return self._experiment_name

    def run(self):
        scenario, tasks_classes = construct_scenario(self._dataset)
        saved_models = []
        for task_id, taskset in enumerate(scenario):
            task_classes = tasks_classes[task_id]
            # Load data for this task
            loader = DataLoader(taskset, batch_size=4, shuffle=False)
            # Train the model on this task
            self._trainer.train_model_on_task(loader, task_classes)
            # Save the model
            save_path = task_id_to_checkpoint_path(self._out_path, task_id)
            saved_models.append(save_path)
            torch.save(self._segm_model.state_dict(), save_path)
