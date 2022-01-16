import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Layer(ABC):
    @abstractmethod
    def init_parameters(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward_propagate(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward_propagate(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_weight_gradients(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute_bias_gradients(self, *args, **kwargs):
        pass
