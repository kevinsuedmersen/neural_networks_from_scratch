from abc import abstractmethod
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from src.lib.metrics import Metric


class ScoreMetric(Metric):
    """Metrics reflecting a score, like accuracy"""
    def __init__(self, name: str, threshold: Union[float, None], average_type: str = "micro"):
        super().__init__(name)
        self.name = name
        self.threshold = threshold
        self.average_type = average_type

        # Init counters
        self.all_true_positives = 0
        self.all_false_positives = 0
        self.all_true_negatives = 0
        self.all_false_negatives = 0

    def _validate_input(self):
        supported_average_types = ["micro"]
        if self.average_type not in supported_average_types:
            raise ValueError(
                f"Unsupported average type: {self.average_type}. "
                f"Supported average types are: {supported_average_types}"
            )

    @staticmethod
    def _binarize(ypred: npt.NDArray, threshold: Union[float, None]) -> npt.NDArray:
        """Binarizes predictions into 0 or 1.
        If threshold is not None, `ypred` becomes 1 if `ypred >= threshold` and 0 otherwise.
        If threshold is None, `ypred` becomes 1 where it is largest and 0 everywhere else.
        """
        binarized_ypred = np.zeros(ypred.shape)

        if threshold is not None:
            binarized_ypred[ypred >= threshold] = 1
        else:
            max_values = np.max(ypred, axis=1, keepdims=True)
            max_idxs = (ypred == max_values)
            binarized_ypred[max_idxs] = 1

        return binarized_ypred

    def _count(
            self,
            ytrue: npt.NDArray,
            ypred: npt.NDArray,
            threshold: Union[float, None]
    ) -> Tuple[int, int, int, int]:
        """Counts true positives, false positives, true negatives, false negatives"""
        binarized_ypred = self._binarize(ypred, threshold)

        # Count TPs, FPs, TNs and FNs
        n_true_positives = np.sum((binarized_ypred == 1) & (ytrue == 1)).item()
        n_false_positives = np.sum((binarized_ypred == 1) & (ytrue == 0)).item()
        n_true_negatives = np.sum((binarized_ypred == 0) & (ytrue == 0)).item()
        n_false_negatives = np.sum((binarized_ypred == 0) & (ytrue == 1)).item()

        # Verify we haven't left out any prediction
        assert (
            (n_true_positives + n_false_positives + n_true_negatives + n_false_negatives) ==
            ytrue.size ==
            binarized_ypred.size
        )

        return n_true_positives, n_false_positives, n_true_negatives, n_false_negatives

    def update_state(self, ytrue: npt.NDArray, ypred: npt.NDArray):
        """Updates all counters"""
        n_true_positives, n_false_positives, n_true_negatives, n_false_negatives = self._count(
            ytrue,
            ypred,
            self.threshold
        )
        self.all_true_positives += n_true_positives
        self.all_false_positives += n_false_positives
        self.all_true_negatives += n_true_negatives
        self.all_false_negatives += n_false_negatives

    @abstractmethod
    def result(self) -> float:
        """Evaluates the current state and returns its result"""
        pass

    def reset_state(self):
        """Sets all counters back to 0"""
        self.all_true_positives = 0
        self.all_false_positives = 0
        self.all_true_negatives = 0
        self.all_false_negatives = 0
