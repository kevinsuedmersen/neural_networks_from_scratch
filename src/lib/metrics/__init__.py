import logging
from abc import ABC, abstractmethod

import numpy.typing as npt

logger = logging.getLogger(__name__)


class Metric(ABC):
    @abstractmethod
    def update_state(self, ytrue: npt.NDArray, ypred: npt.NDArray):
        pass

    @abstractmethod
    def result(self) -> float:
        """Evaluates the current state and returns its result"""
        pass

    @abstractmethod
    def reset_state(self):
        """Resets internal state"""
        pass
