from abc import ABC, abstractmethod
from typing import Tuple, Generator

import numpy.typing as npt

from src.types import NFeatures, NNeuronsOut, NSamples


class Model(ABC):
    @abstractmethod
    def train(
            self,
            data_gen_train: Generator,
            data_gen_val: Generator,
            epochs: int,
            batch_size: int,
            **kwargs
    ):
        pass

    @abstractmethod
    def predict(
            self,
            x: npt.NDArray[Tuple[NSamples, NFeatures]],
            **kwargs
    ) -> npt.NDArray[Tuple[NSamples, NNeuronsOut]]:
        pass

    @abstractmethod
    def evaluate(self, data_gen: Generator, **kwargs):
        pass
