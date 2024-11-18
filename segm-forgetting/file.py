import os
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent.resolve().absolute()
DATA_PATH = ROOT_PATH / "data"
EXPERIMENTS_PATH = ROOT_PATH / "experiments"
SRC_PATH = ROOT_PATH / "segm-forgetting"

DATA_PATH.mkdir(exist_ok=True)
EXPERIMENTS_PATH.mkdir(exist_ok=True)


def resolve_to_project_root(path: str):
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT_PATH / p


def check_create_dir(path: str) -> Path:
    p = resolve_to_project_root(path)
    if not p.exists():
        print("Creating directory {}".format(p))
        p.mkdir(parents=True, exist_ok=True)
    return p
