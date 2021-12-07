import logging

import numpy as np
import numpy.typing as npt

from src.layers.activation_functions import get_activation_function
from src.layers.interface import Layer

logger = logging.getLogger(__name__)


class DenseLayer(Layer):
    def __init__(
            self,
            units: int,
            activation_function_name: str
    ):
        self.units = units
        self.activation_function_name = activation_function_name

        self.activation_function = get_activation_function(activation_function_name)
        self.weights = None
        self.biases = None
        self.output_shape = None
        self.dendritic_potentials = None
        self.activations = None

    def _init_weights(self, units_prev: int):
        self.weights = np.random.randn(1, self.units, units_prev) * 0.01  # batch_size=1 for broadcasting

    def _init_biases(self):
        self.biases = np.random.randn(1, self.units, 1)  # batch_size=1 for broadcasting

    def init_parameters(self, units_prev: int):
        self._init_weights(units_prev)
        self._init_biases()
        self.output_shape = (None, self.units)

    def forward(self, activations_prev: npt.NDArray) -> npt.NDArray:
        """Computes the activations of the current layer"""
        self.dendritic_potentials = np.matmul(self.weights, activations_prev) + self.biases
        self.activations = self.activation_function(self.dendritic_potentials)

        return self.activations

    def backward(self, error_next: npt.NDArray) -> npt.NDArray:
        pass

    def update_params(self):
        pass
