import logging
from abc import ABC, abstractmethod
from typing import Tuple, Generator

import numpy.typing as npt

logger = logging.getLogger(__name__)


class DataGenerator(ABC):
    @abstractmethod
    def train(self, *args, **kwargs) -> Tuple[Generator[Tuple[npt.NDArray, npt.NDArray], None, None], int]:
        pass

    @abstractmethod
    def val(self, *args, **kwargs) -> Tuple[Generator[Tuple[npt.NDArray, npt.NDArray], None, None], int]:
        pass

    @abstractmethod
    def test(self, *args, **kwargs) -> Tuple[Generator[Tuple[npt.NDArray, npt.NDArray], None, None], int]:
        pass

    @abstractmethod
    def get_n_classes(self) -> int:
        """Returns the number of different classes. Returns 1 for regressions"""
        pass
