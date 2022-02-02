import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Layer(ABC):
    def __init__(self):
        self.weights = None
        self.biases = None
        self.output_shape = None
        self.dendritic_potentials = None
        self.activations = None
        self.errors = None
        self.weight_gradients = None
        self.bias_gradients = None
        self.layer_idx = None

    @abstractmethod
    def init_parameters(self, *args, **kwargs):
        """Initializes weights and biases of the current layer"""
        pass

    @abstractmethod
    def forward_propagate(self, *args, **kwargs):
        """Computes the activations of the current layer"""
        pass

    @abstractmethod
    def backward_propagate(self, *args, **kwargs):
        """Computes the error of the current layer"""
        pass

    @abstractmethod
    def compute_weight_gradients(self, *args, **kwargs):
        """Computes the weight gradients of the current layer"""
        pass

    @abstractmethod
    def compute_bias_gradients(self, *args, **kwargs):
        """Computes the bias gradients of the current layer"""
        pass
