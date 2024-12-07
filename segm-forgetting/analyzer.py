from pathlib import Path

import numpy as np
import pandas as pd

from evaluator import metrics_file_from_idx
from util import N_TASKS


class CLAnalyzer:
    def __init__(self, experiment_path: Path):
        self._eval_path = experiment_path / 'eval'
        assert self._eval_path.exists() and self._eval_path.is_dir()
        self._accuracies_mx = np.zeros((N_TASKS, N_TASKS)) # each row is a model's performance on all tasks
        self._miou_mx = np.zeros((N_TASKS, N_TASKS)) # Each column is a single task performance on subsequent models
        self._forgetting_mx = np.zeros(N_TASKS)
        self._bwt_mx = np.zeros(N_TASKS)


    def _cl_metrics(self, metrics_df):
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

    def full_analysis(self):

        for model_idx in range(N_TASKS):
            metrics_file_path = self._eval_path / metrics_file_from_idx(model_idx)
            metrics_df = pd.read_csv(metrics_file_path)



        forgetting, bwt = self.cl_metrics()
        return {
            "forgetting": forgetting,
            "bwt": bwt
        }