import logging
from abc import ABC, abstractmethod

import numpy.typing as npt

logger = logging.getLogger(__name__)


class Loss(ABC):
    @abstractmethod
    def compute_losses(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_cost(self, losses: npt.NDArray):
        pass

    @abstractmethod
    def init_error(self, *args, **kwargs):
        pass
