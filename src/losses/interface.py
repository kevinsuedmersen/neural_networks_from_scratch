import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy.typing as npt

from src.types import BatchSize, NNeuronsOut

logger = logging.getLogger(__name__)


class Loss(ABC):
    @abstractmethod
    def compute_loss(
            self,
            ytrue: npt.NDArray[Tuple[BatchSize, NNeuronsOut, 1]],
            ypred: npt.NDArray[Tuple[BatchSize, NNeuronsOut, 1]]
    ) -> npt.NDArray[Tuple[BatchSize, 1, 1]]:
        pass

    @abstractmethod
    def compute_cost(self,losses: npt.NDArray[Tuple[BatchSize, 1, 1]]) -> float:
        pass

    @abstractmethod
    def init_error(self, *args, **kwargs) -> npt.NDArray:
        pass
