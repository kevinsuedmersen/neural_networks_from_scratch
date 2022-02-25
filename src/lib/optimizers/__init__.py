import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        logger.info(f"Learning rate: {learning_rate}")

    @abstractmethod
    def update_parameters(self, *args, **kwargs):
        pass
