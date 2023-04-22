"""hyperparameter tuner for sysidexpr models"""
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Tuple

import autokoopman.core.trajectory as atraj
import numpy as np

from sysidexpr.model import Metric


class HyperparamTuner:
    """hyperparameter tuner for sysidexpr models"""

    def __init__(
        self,
        training_data: atraj.TrajectoriesData,
        validation_data: atraj.TrajectoriesData,
        hyperparameters: Generator[Tuple[Any]],
        experiment_runner: Callable[[Any], Tuple[Tuple[Metric, float], Dict[str, Any]]],
    ) -> None:
        """hyperparameter tuner for sysidexpr models
        :param training_data: training data
        :param validation_data: validation data
        :param hyperparameters: generator of hyperparameters
        :param experiment_runner: function that takes a hyperparameter and returns a metric and score
        """
        self.training_data = training_data
        self.validation_data = validation_data
        self.hyperparameters = hyperparameters
        self.experiment_runner = experiment_runner

        self.metrics = []
        self.scores = []
        self.hyperparams = []

        self.best_metric = None
        self.best_score = None
        self.best_hyperparams = None

    def tune(self):
        """run the tuner"""
        for hyperparam in self.hyperparameters:
            (metric, score), hyperparam = self.experiment_runner(hyperparam)
            self.metrics.append(metric)
            self.scores.append(score)
            self.hyperparams.append(hyperparam)

        if self.metrics[0].lower_better:
            best_idx = np.argmin(self.scores)
        else:
            best_idx = np.argmax(self.scores)

        best_metric = self.metrics[best_idx]
        best_score = self.scores[best_idx]
        best_hyperparams = self.hyperparams[best_idx]

        self.best_metric = best_metric
        self.best_score = best_score
        self.best_hyperparams = best_hyperparams

        return best_metric, best_score, best_hyperparams
