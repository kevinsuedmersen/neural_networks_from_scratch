import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class Metric(ABC):
    def __init__(self, name: str, threshold: float = 0.5, average_type: str = "micro"):
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
    def _binarize(ypred: npt.NDArray, threshold: float) -> npt.NDArray:
        """Binarizes predictions into 0 or 1. `ypred` becomes 1 if `ypred >= threshold` and 0
        otherwise
        """
        binarized_ypred = ypred.copy()
        binarized_ypred[ypred >= threshold] = 1
        binarized_ypred[ypred < threshold] = 0

        return binarized_ypred

    def _count(
            self,
            ytrue: npt.NDArray,
            ypred: npt.NDArray,
            threshold: float
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
