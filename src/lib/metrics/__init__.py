import logging
from abc import ABC, abstractmethod

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
