import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt

from src.activation_functions import get_activation_function
from src.layers.interface import Layer

logger = logging.getLogger(__name__)


class DenseLayer(Layer):
    def __init__(
            self,
            n_neurons: int,
            activation_function_name: str,
            layer_idx: int = None
    ):
        self.n_neurons = n_neurons
        self.activation_function_name = activation_function_name
        self.layer_idx = layer_idx

        self.activation_function_forward, self.activation_function_backward = get_activation_function(
            activation_function_name
        )
        self.weights = None
        self.biases = None
        self.output_shape = None
        self.dendritic_potentials = None
        self.activations = None
        self.error = None

    def _init_weights(self, units_prev: int):
        self.weights = np.random.randn(1, self.n_neurons, units_prev) * 0.01  # batch_size=1 for broadcasting

    def _init_biases(self):
        self.biases = np.random.randn(1, self.n_neurons, 1)  # batch_size=1 for broadcasting

    def init_parameters(self, n_neurons_prev: int):
        self._init_weights(n_neurons_prev)
        self._init_biases()
        self.output_shape = (None, self.n_neurons)

    def forward(self, activations_prev: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Computes the activations of the current layer"""
        self.dendritic_potentials = np.matmul(self.weights, activations_prev) + self.biases
        self.activations = self.activation_function_forward(self.dendritic_potentials)

        return self.activations, self.dendritic_potentials

    def backward(self, error_next: npt.NDArray, weights_next: npt.NDArray) -> npt.NDArray:
        """Computes the backward pass of the current layer
        :param error_next: If the current layer is l, error_next is the error in layer l + 1
        :param weights_next: If the current layer is l, weights_next are the weights in layer l + 1
        :return: Returns the error of the current layer, i.e. layer l
        """
        # Compute the jacobians of the current layer batch_size times at once
        # jacobians.shape = (batch_size, n_neurons, n_neurons)
        jacobians = self.activation_function_backward(self.dendritic_potentials, self.activations)

        # Compute the error of the current layer
        # self.error.shape = (batch_size, n_neurons, 1)
        weights_next_t = np.transpose(weights_next, axes=[0, 2, 1])
        j_w = np.matmul(jacobians, weights_next_t)
        self.error = np.matmul(j_w, error_next)

        # TODO: Test that the shape of the errors is (batch_size, n_neurons, 1)

        return self.error

    def compute_weight_grads(self, *args, **kwargs):
        pass

    def compute_bias_grads(self, *args, **kwargs):
        pass

    def update_params(self):
        pass
