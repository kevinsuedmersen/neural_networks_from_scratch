import logging
from abc import ABC, abstractmethod
from typing import Generator

import numpy.typing as npt

logger = logging.getLogger(__name__)


class DataGenerator(ABC):
    @abstractmethod
    def train(self) -> Generator[npt.NDArray, None, None]:
        pass

    @abstractmethod
    def val(self) -> Generator[npt.NDArray, None, None]:
        pass

    @abstractmethod
    def test(self) -> Generator[npt.NDArray, None, None]:
        pass
