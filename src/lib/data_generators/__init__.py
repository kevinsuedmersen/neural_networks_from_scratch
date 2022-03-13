import logging
from abc import ABC, abstractmethod
from typing import Tuple, Generator

import numpy.typing as npt

logger = logging.getLogger(__name__)


class DataGenerator(ABC):
    def __init__(self):
        self.n_samples_train = None
        self.n_samples_val = None
        self.n_samples_test = None
        self.n_classes = None
        self.loop_forever = None

    @abstractmethod
    def train(self, *args, **kwargs) -> Tuple[Generator[Tuple[npt.NDArray, npt.NDArray], None, None], int]:
        pass

    @abstractmethod
    def val(self, *args, **kwargs) -> Tuple[Generator[Tuple[npt.NDArray, npt.NDArray], None, None], int]:
        pass

    @abstractmethod
    def test(self, *args, **kwargs) -> Tuple[Generator[Tuple[npt.NDArray, npt.NDArray], None, None], int]:
        pass
