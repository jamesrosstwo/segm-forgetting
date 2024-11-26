from datetime import datetime
from pathlib import Path
from typing import Tuple

from continuum import SegmentationClassIncremental
from continuum.transforms.segmentation import Resize, ToTensor

import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from file import EXPERIMENTS_PATH, check_create_dir, DATA_PATH
from trainer import SegmentationTrainer


def get_date_string() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


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
        self._trainer_conf = trainer
        self._exp_instance_name = f"{self.name}_{get_date_string()}"
        self._should_cache_results = should_cache_results
        self._model_conf = model
        self._out_path, self._out_cache_dir = self._make_output_dirs()
        self._debug_mode = debug

        self._init_wandb()
        self._dataset = self._construct_dataset(self._data_conf)
        self._segm_model = self._construct_model(self._model_conf)
        self._trainer: SegmentationTrainer = self._construct_trainer(self._trainer_conf)

    def _make_output_dirs(self):
        out_path = EXPERIMENTS_PATH / self._exp_instance_name
        out_dir = check_create_dir(str(out_path))
        latest_path = EXPERIMENTS_PATH / "latest"
        latest_path.unlink(missing_ok=True)
        latest_path.symlink_to(out_dir, target_is_directory=True)

        base_cache_dir = check_create_dir(str(EXPERIMENTS_PATH / "cache"))
        out_cache_dir = check_create_dir(str(base_cache_dir / self.name))
        return out_dir, out_cache_dir

    def _construct_trainer(self, trainer_conf: DictConfig) -> SegmentationTrainer:
        return instantiate(self._trainer_conf, _recursive_=False, model=self._segm_model)

    def _construct_dataset(self, data_conf: DictConfig):
        return instantiate(data_conf, data_path=str(DATA_PATH / data_conf._target_))

    def _construct_model(self, model_conf: DictConfig):
        return instantiate(model_conf)

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
            dir=self._out_path,
            project=self._wandb_project_name,
            name=self._exp_instance_name,
            config={
                "dataset": OmegaConf.to_container(self._data_conf),
                "model": OmegaConf.to_container(self._model_conf)
            }
        )

    @property
    def name(self):
        return self._experiment_name

    def run(self):
        scenario = SegmentationClassIncremental(
            self._dataset,
            nb_classes=20,
            initial_increment=15, increment=1,
            mode="overlap",
            transformations=[Resize((512, 512)), ToTensor()]
        )

        for task_id, taskset in enumerate(scenario):
            loader = DataLoader(taskset)
            self._trainer.train_model_on_task(loader)
            # TODO: run metrics
