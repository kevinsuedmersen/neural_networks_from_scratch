from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import numpy.typing as npt


class Model(ABC):
    @abstractmethod
    def _forward_pass(self, **kwargs):
        pass

    @abstractmethod
    def _compute_cost(self, **kwargs):
        pass

    @abstractmethod
    def _backward_pass(self, **kwargs):
        pass

    @abstractmethod
    def _update_params(self, **kwargs):
        pass

    @abstractmethod
    def train(
            self,
            train_data_gen: Iterable[Tuple[npt.NDArray, npt.NDArray]],
            val_data_gen: Iterable,
            **kwargs
    ):
        pass

    @abstractmethod
    def predict(self, data_gen: Iterable[Tuple[npt.NDArray, npt.NDArray]], **kwargs):
        pass

    @abstractmethod
    def evaluate(self, data_gen: Iterable[Tuple[npt.NDArray, npt.NDArray]], **kwargs):
        pass
