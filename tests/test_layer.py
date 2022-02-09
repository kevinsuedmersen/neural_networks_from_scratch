import numpy as np
import pytest

from src.lib.layers.dense import DenseLayer
from tests.test_config import TestConfig


class TestLayer(TestConfig):
    activation_function_name = "tanh"

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

    @pytest.fixture
    def layer(self):
        dense_layer = DenseLayer(self.n_neurons, self.activation_function_name)
        dense_layer.init_parameters(self.n_neurons_prev)

        return dense_layer

    def test_forward_propagate(self, layer, activations_prev):
        """Tests that the shape of the dendritic_potentials and activations is as expected for all
        activation functions
        """
        activations, dendritic_potentials = layer.forward_propagate(activations_prev)
        assert activations.shape == dendritic_potentials.shape == (self.batch_size, self.n_neurons, 1)

    @pytest.fixture
    def forward_propagated_layer(self, layer, activations_prev):
        activations, dendritic_potentials = layer.forward_propagate(activations_prev)

        return layer

    def test_backward_propagate(self, forward_propagated_layer, error_next, weights_next, activations_prev):
        """Tests that the shape of the returned errors is as expected for all activation functions"""
        error = forward_propagated_layer.backward_propagate(error_next, weights_next)
        assert error.shape == (self.batch_size, self.n_neurons, 1)

    @pytest.fixture
    def backward_propagated_layer(self, forward_propagated_layer, error_next, weights_next):
        error = forward_propagated_layer.backward_propagate(error_next, weights_next)

        return forward_propagated_layer

    def test_compute_weight_gradients(self, backward_propagated_layer, activations_prev):
        """Test that the shape of the computed weight gradients is as expected"""
        backward_propagated_layer.compute_weight_gradients(activations_prev)
        assert backward_propagated_layer.weight_gradients.shape == (1, self.n_neurons, self.n_neurons_prev)

    def test_compute_bias_gradients(self, backward_propagated_layer):
        backward_propagated_layer.compute_bias_gradients()
        assert backward_propagated_layer.bias_gradients.shape == (1, self.n_neurons, 1)
