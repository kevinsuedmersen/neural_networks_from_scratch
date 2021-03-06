from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self):
        self.trained = False

    @abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def val_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass
