from abc import ABC, abstractmethod

from src.data_gen.interface import DataGenerator


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
    def train(self, data_gen: DataGenerator, epochs: int):
        pass

    @abstractmethod
    def predict(self, data_gen: DataGenerator, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, data_gen: DataGenerator, **kwargs):
        pass
