from abc import ABC, abstractmethod
from typing import Tuple

import numpy.typing as npt

from src.data_gen.interface import DataGenerator
from src.types import BatchSize, NFeatures, NNeuronsOut, NSamples


class Model(ABC):
    @abstractmethod
    def _train_step(
            self,
            x_train: npt.NDArray[Tuple[BatchSize, NFeatures]],
            ytrue_train: npt.NDArray[Tuple[BatchSize, NNeuronsOut]]
    ) -> npt.NDArray[Tuple[BatchSize, NNeuronsOut]]:
        pass

    @abstractmethod
    def _val_step(
            self,
            x_val: npt.NDArray[BatchSize, NFeatures],
            ytrue_val: npt.NDArray[BatchSize, NNeuronsOut]
    ) -> npt.NDArray[Tuple[BatchSize, NNeuronsOut]]:
        pass

    @abstractmethod
    def _compute_cost(
            self,
            ytrue_batch: npt.NDArray[Tuple[BatchSize, NNeuronsOut, 1]],
            ypred_batch: npt.NDArray[Tuple[BatchSize, NNeuronsOut, 1]],
            batch_idx: int
    ):
        pass

    @abstractmethod
    def _backward_pass(self, *args, **kwargs):
        pass

    @abstractmethod
    def _update_params(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, data_gen: DataGenerator, epochs: int):
        pass

    @abstractmethod
    def predict(
            self,
            x: npt.NDArray[NSamples, NFeatures]
    ) -> npt.NDArray[NSamples, NNeuronsOut]:
        pass

    @abstractmethod
    def evaluate(
            self,
            ytrue: npt.NDArray[NSamples, NNeuronsOut],
            ypred: npt.NDArray[NSamples, NNeuronsOut]):
        pass
