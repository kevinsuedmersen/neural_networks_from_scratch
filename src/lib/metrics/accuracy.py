import logging

import numpy as np
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
        binarized_ypred = self._binarize(ypred, self.threshold)

        # Align types to auxilarate equality comparisons
        binarized_ypred = binarized_ypred.astype(np.float32)
        ytrue = ytrue.astype(np.float32)

        # Count correct predictions for each class in the whole batch
        _n_correct_preds = np.sum(ytrue == binarized_ypred)
        self.n_correct_preds += _n_correct_preds

        # Count number all predictions for each class in the whole batch
        _n_all_preds = ytrue.size
        self.n_all_preds += _n_all_preds

    def result(self):
        """Returns the percentage of correct predictions from all predictions"""
        percentage_correct_preds = self.n_correct_preds / self.n_all_preds * 100

        return percentage_correct_preds

    def reset_state(self):
        """Resets internal counters"""
        self.n_correct_preds = 0
        self.n_all_preds = 0
