from datetime import datetime
from pathlib import Path

from continuum import SegmentationClassIncremental
from continuum.transforms.segmentation import Resize, ToTensor, Normalize
from hydra.utils import instantiate
from omegaconf import DictConfig

from file import DATA_PATH


def get_date_string() -> str:
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def task_id_to_checkpoint_path(base_out_path: Path, task_id: int) -> Path:
    return base_out_path / f"model_task_{task_id}.pth"


def construct_trainer(trainer_conf: DictConfig, segm, **kwargs):
    return instantiate(trainer_conf, _recursive_=False, model=segm, **kwargs)


def construct_dataset(data_conf: DictConfig, **kwargs):
    return instantiate(data_conf, data_path=str(DATA_PATH / data_conf._target_), **kwargs)


def construct_evaluator(evaluator_conf: DictConfig, segm, **kwargs):
    return instantiate(evaluator_conf, _recursive_=False, model=segm, **kwargs)


def construct_model(model_conf: DictConfig, **kwargs):
    return instantiate(model_conf, **kwargs)


def construct_scenario(dataset):
    scenario = SegmentationClassIncremental(
        dataset,
        nb_classes=20,
        initial_increment=15, increment=1,
        mode="sequential",
        transformations=[
            Resize((512, 512)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    task_classes = [[0] + scenario.class_order[:15 + i] for i in range(6)]
    return scenario, task_classes
