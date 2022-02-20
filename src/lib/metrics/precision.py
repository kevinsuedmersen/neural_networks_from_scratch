import logging

import numpy as np
import numpy.typing as npt

from src.lib.metrics import Metric

logger = logging.getLogger(__name__)


class Precision(Metric):
    def __init__(self, name: str, threshold: float = 0.5, average_type: str = "micro"):
        super().__init__(name)
        self.threshold = threshold
        self.average_type = average_type

        self.all_true_positives = 0
        self.all_false_positives = 0

    def update_state(self, ytrue: npt.NDArray, ypred: npt.NDArray):
        """Update number of true and false positives"""
        n_true_positives, n_false_positives, _, _ = self._count(
            ytrue,
            ypred,
            self.threshold
        )
        self.all_true_positives += n_true_positives
        self.all_false_positives += n_false_positives

    def result(self):
        """Compute precision and return NaN if there are no predictions of the positive class"""
        if (self.all_true_positives + self.all_false_positives) == 0:
            return np.nan
        else:
            return self.all_true_positives / (self.all_true_positives + self.all_false_positives)

    def reset_state(self):
        """Resets the counters of all true and all false positives"""
        self.all_true_positives = 0
        self.all_false_positives = 0
