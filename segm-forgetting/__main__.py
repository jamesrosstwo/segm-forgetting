import hydra
from omegaconf import DictConfig

from experiment import Experiment


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run_experiment(cfg: DictConfig):
    experiment = Experiment(**cfg)
    return experiment.run()


if __name__ == "__main__":
    run_experiment()