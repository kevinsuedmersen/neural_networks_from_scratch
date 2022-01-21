import logging
from typing import Tuple

import numpy as np
import numpy.typing as npt

from src.activation_functions import get_activation_function
from src.layers import Layer

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
        self.errors = None
        self.weight_gradients = None
        self.bias_gradients = None

    def _init_weights(self, n_neurons_prev: int):
        """Init weight matrices"""
        self.weights = np.random.randn(1, self.n_neurons, n_neurons_prev) * 0.01  # batch_size=1 for broadcasting

    def _init_biases(self):
        """Init biases"""
        self.biases = np.random.randn(1, self.n_neurons, 1)  # batch_size=1 for broadcasting

    def _init_weight_gradients(self):
        """Init the weight gradients. Necessary for testing gradient computation"""
        self.weight_gradients = np.zeros(self.weights.shape)

    def _init_bias_gradients(self):
        """Init the bias gradients. Necessary for testing gradient computation"""
        self.bias_gradients = np.zeros(self.biases.shape)

    def init_parameters(self, n_neurons_prev: int):
        self._init_weights(n_neurons_prev)
        self._init_biases()
        self._init_weight_gradients()
        self._init_bias_gradients()
        self.output_shape = (None, self.n_neurons)

    def forward_propagate(self, activations_prev: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Computes the activations of the current layer"""
        self.dendritic_potentials = np.matmul(self.weights, activations_prev) + self.biases
        self.activations = self.activation_function_forward(self.dendritic_potentials)

        return self.activations, self.dendritic_potentials

    def backward_propagate(self, error_next: npt.NDArray, weights_next: npt.NDArray) -> npt.NDArray:
        """Computes the errors of the current layer
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
        self.errors = np.matmul(j_w, error_next)

        return self.errors

    def compute_weight_gradients(self, activations_prev: npt.NDArray, *args, **kwargs):
        """Computes the weight gradients of the current layer
        :param activations_prev: If the current layer is l, these activations belong to layer l - 1
        """
        # Derivatives of the loss function w.r.t. each weight in each batch-element in the current layer
        # shape=(batch_size, n_neurons, n_neurons_prev)
        activations_prev_t = np.transpose(activations_prev, axes=[0, 2, 1])
        derivative_loss_wrt_weights = np.matmul(self.errors, activations_prev_t)

        # Derivative of the cost function w.r.t. each weight in the current layer
        # shape=(1, n_neurons, n_neurons_prev), the same as the layer's weights
        self.weight_gradients = np.mean(derivative_loss_wrt_weights, axis=0, keepdims=True)

    def compute_bias_gradients(self, *args, **kwargs):
        """Computes the bias gradients of the current layer"""
        # Bias gradients equal the errors averaged over all batch elements
        # shape=(1, n_neurons, 1), same the layers' biases
        self.bias_gradients = np.mean(self.errors, axis=0, keepdims=True)

    def update_parameters(self):
        logger.warning("the update_parameters method is not implemented yet and currently does nothing")
