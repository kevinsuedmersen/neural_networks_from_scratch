import numpy as np
import pytest

from src.layers.dense import DenseLayer
from tests.test_config import TestConfig


class TestLayer(TestConfig):
    n_neurons_prev = 8
    n_neurons_next = 32

    @pytest.fixture
    def weights(self):
        """Weights of the current layer"""
        return np.random.randn(1, self.n_neurons, self.n_neurons_prev) * 0.01

    @pytest.fixture
    def biases(self):
        """Biases of the current layer"""
        return np.random.randn(1, self.n_neurons, 1)

    @pytest.fixture
    def weights_next(self):
        """If the current layer's index is l, weights_next correspond to layer with index l + 1"""
        return np.random.randn(1, self.n_neurons_next, self.n_neurons) * 0.01

    @pytest.fixture
    def error(self):
        """Error of the current layer"""
        return np.random.randn(self.batch_size, self.n_neurons, 1)

    @pytest.fixture
    def error_next(self):
        """If the current layer's index is l, error_next corresponds to layer with index l + 1"""
        return np.random.randn(self.batch_size, self.n_neurons_next, 1)


class TestDenseLayer(TestLayer):
    @pytest.fixture
    def activations_prev(self):
        return np.random.randn(self.batch_size, self.n_neurons_prev, 1)

    def _init_layer(self, activation_function_name, activations_prev):
        dense_layer = DenseLayer(self.n_neurons, activation_function_name)
        dense_layer.init_parameters(self.n_neurons_prev)

        return dense_layer

    def test_forward_propagate(self, activations_prev):
        """Tests that the shape of the dendritic_potentials and activations is as expected for all
        activation functions
        """
        for activation_function_name in self.activation_function_names:
            dense_layer = self._init_layer(activation_function_name, activations_prev)
            activations, dendritic_potentials = dense_layer.forward_propagate(activations_prev)
            assert activations.shape == dendritic_potentials.shape == (self.batch_size, self.n_neurons, 1)

    def test_backward_propagate(self, error_next, weights_next, activations_prev):
        """Tests that the shape of the returned errors is as expected for all activation functions"""
        for activation_function_name in self.activation_function_names:
            dense_layer = self._init_layer(activation_function_name, activations_prev)
            _ = dense_layer.forward_propagate(activations_prev)
            error = dense_layer.backward_propagate(error_next, weights_next)
            assert error.shape == (self.batch_size, self.n_neurons, 1)
