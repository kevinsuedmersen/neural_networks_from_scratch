from abc import ABC, abstractmethod
from typing import Generator

import numpy.typing as npt


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
    def train_step(
            self,
            x_train: npt.NDArray,
            ytrue_train: npt.NDArray
    ):
        pass

    @abstractmethod
    def val_step(
            self,
            x_val: npt.NDArray,
            ytrue_val: npt.NDArray
    ):
        pass

    @abstractmethod
    def predict(
            self,
            x: npt.NDArray,
            **kwargs
    ) -> npt.NDArray:
        pass

    @abstractmethod
    def evaluate(self, data_gen: Generator, **kwargs):
        pass
