import logging

import numpy.typing as npt

from src.lib.metrics import Metric

logger = logging.getLogger(__name__)


class Accuracy(Metric):
    def __init__(self, name: str, threshold: float = 0.5, average_type: str = "micro"):
        super().__init__(name)
        self.threshold = threshold
        self.average_type = average_type

        # Internal counters / state trackers
        self.n_correct_preds = 0
        self.n_all_preds = 0

    def _validate_input(self):
        supported_average_types = ["micro"]
        if self.average_type not in supported_average_types:
            raise ValueError(
                f"Unsupported average type: {self.average_type}. "
                f"Supported average types are: {supported_average_types}"
            )

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
        accuracy_value = self.n_correct_preds / self.n_all_preds

        return accuracy_value

    def reset_state(self):
        """Resets internal counters"""
        self.n_correct_preds = 0
        self.n_all_preds = 0
