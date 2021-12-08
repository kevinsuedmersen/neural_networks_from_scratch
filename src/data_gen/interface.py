import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataGenerator(ABC):
    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def val(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass
