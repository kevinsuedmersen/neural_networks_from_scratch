from abc import ABC, abstractmethod
from typing import Union

from src.config_parser.classification import ImageClassificationConfigParser


class MLJob(ABC):
    def __init__(self, cp: Union[ImageClassificationConfigParser]):
        self.cp = cp

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def benchmark_performance(self):
        pass
