import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class Metric(ABC):
    def __init__(self, name: str):
        self.name = name

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

        n_true_positives = np.sum((binarized_ypred == 1) & (ytrue == 1)).item()
        n_false_positives = np.sum((binarized_ypred == 1) & (ytrue == 0)).item()
        n_true_negatives = np.sum((binarized_ypred == 0) & (ytrue == 0)).item()
        n_false_negatives = np.sum((binarized_ypred == 0) & (ytrue == 1)).item()

        # Verify we left out no predictions
        assert (
            (n_true_positives + n_false_positives + n_true_negatives + n_false_negatives) ==
            ytrue.size ==
            binarized_ypred.size
        )

        return n_true_positives, n_false_positives, n_true_negatives, n_false_negatives

    @abstractmethod
    def update_state(self, ytrue: npt.NDArray, ypred: npt.NDArray):
        """Updates the internal state of the metric"""
        pass

    @abstractmethod
    def result(self):
        """Evaluates the current state and returns its result"""
        pass

    @abstractmethod
    def reset_state(self):
        """Resets the internal state of the metric to its initial state"""
        pass
