import hydra
from omegaconf import DictConfig

from experiment import Experiment

@hydra.main(version_base=None, config_path="../config", config_name="config")
def run_experiment(cfg: DictConfig):
    if isinstance(cfg, list):
        results = []
        for config in cfg:
            experiment = Experiment(**config)
            results.append(experiment.run())
        return results

    experiment = Experiment(**cfg)
    return experiment.run()

if __name__ == "__main__":
    run_experiment()
