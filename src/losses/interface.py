import logging
from abc import ABC, abstractmethod

import numpy.typing as npt

logger = logging.getLogger(__name__)


class Loss(ABC):
    @abstractmethod
    def compute_loss(
            self,
            ytrue: npt.NDArray,
            ypred: npt.NDArray
    ) -> npt.NDArray:
        pass

    @abstractmethod
    def compute_cost(self, losses: npt.NDArray) -> float:
        pass

    @abstractmethod
    def init_error(self, *args, **kwargs) -> npt.NDArray:
        pass
