import logging

import numpy.typing as npt

from src.lib.metrics import Metric

logger = logging.getLogger(__name__)


class Accuracy(Metric):
    def __init__(self, name: str, threshold: float = 0.5):
        super().__init__(name)
        self.threshold = threshold

        # Internal counters / state trackers
        self.n_correct_preds = 0
        self.n_all_preds = 0

    def update_state(self, ytrue: npt.NDArray, ypred: npt.NDArray):
        """Count and updates number of correct predictions and number of all predictions"""
        n_true_positives, n_false_positives, n_true_negatives, n_false_negatives = self._count(
            ytrue,
            ypred,
            self.threshold
        )
        self.n_correct_preds += (n_true_positives + n_true_negatives)
        self.n_all_preds += (n_true_positives + n_false_positives + n_true_negatives + n_false_negatives)

    def result(self):
        """Returns the percentage of correct predictions from all predictions"""
        accuracy_value = self.n_correct_preds / self.n_all_preds * 100

        return accuracy_value

    def reset_state(self):
        """Resets internal counters"""
        self.n_correct_preds = 0
        self.n_all_preds = 0
