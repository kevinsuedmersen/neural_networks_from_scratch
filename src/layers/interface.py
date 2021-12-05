import logging
from abc import ABC, abstractmethod

import numpy.typing as npt

logger = logging.getLogger(__name__)


class Layer(ABC):
    def init_parameters(self):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> npt.NDArray:
        pass

    @abstractmethod
    def backward(self, *args, **kwargs) -> npt.NDArray:
        pass
