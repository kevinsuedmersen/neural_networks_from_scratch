import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Layer(ABC):
    @abstractmethod
    def init_parameters(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_weight_grads(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_bias_grads(self, *args, **kwargs):
        pass
