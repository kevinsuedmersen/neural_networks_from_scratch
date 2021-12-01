import logging
from abc import ABC, abstractmethod
from typing import Generator

logger = logging.getLogger(__name__)


class DataGenerator(ABC):
    @abstractmethod
    def train(self) -> Generator:
        pass

    @abstractmethod
    def val(self) -> Generator:
        pass

    @abstractmethod
    def test(self) -> Generator:
        pass
