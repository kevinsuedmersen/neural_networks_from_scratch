import numpy as np
import numpy.typing as npt
import pytest

from src.lib.activation_functions import softmax_forward, relu_forward, linear_backward, \
    linear_forward, \
    relu_backward, sigmoid_forward, sigmoid_backward, tanh_forward, tanh_backward
from src.lib.activation_functions.softmax import softmax_backward
from tests.test_config import TestConfig
from tests.utils import assert_euclidean_distance


class TestLinearActivationFunction(TestConfig):
    def test_linear_forward(self, dendritic_potentials):
        """Tests that inputs equal outputs"""
        activations = linear_forward(dendritic_potentials)
        np.testing.assert_array_equal(activations, dendritic_potentials)

    @pytest.fixture
    def activations(self, dendritic_potentials):
        activations = linear_forward(dendritic_potentials)

        return activations

    def test_linear_backward(self, dendritic_potentials, activations):
        """Tests that we have batch_size identity matrices"""
        jacobians = linear_backward(dendritic_potentials, activations)
        assert jacobians.shape == (self.batch_size, self.n_neurons, self.n_neurons)
        for jacobian in jacobians:
            identity_matrix = np.eye(self.n_neurons)
            np.testing.assert_array_equal(jacobian, identity_matrix)


class TestReluActivationFunction(TestConfig):
    def test_relu_forward(self, dendritic_potentials):
        """Tests the expected shape of activations and that the activations are not smaller than 0"""
        activations = relu_forward(dendritic_potentials)
        assert activations.shape == (self.batch_size, self.n_neurons, 1)
        assert np.all(activations >= 0)

    @pytest.fixture
    def activations(self, dendritic_potentials):
        activations = relu_forward(dendritic_potentials)

        return activations

    def test_relu_backward(self, dendritic_potentials, activations):
        """Verifies the shape of the jacobians and that its elements are 1 where
        dendritic_potentials > 0 and 0 elsewhere
        """
        jacobians = relu_backward(dendritic_potentials, activations)
        assert jacobians.shape == (self.batch_size, self.n_neurons, self.n_neurons)
        indices_1 = np.squeeze(dendritic_potentials > 0)
        indices_0 = np.squeeze(dendritic_potentials <= 0)
        diagonal_elements = jacobians[:, np.arange(self.n_neurons), np.arange(self.n_neurons)]
        assert np.all(diagonal_elements[indices_1] == 1)
        assert np.all(diagonal_elements[indices_0] == 0)


class TestSigmoidActivationFunction(TestConfig):
    def test_sigmoid_forward(self, dendritic_potentials):
        """Tests that:
        - The shape of the activations make sense
        - All activations are between 0 and 1
        - Our optimized implementation yields the same results as f(z) = 1 / (1 + e^(-z))
        """
        actual_activations = sigmoid_forward(dendritic_potentials)
        assert actual_activations.shape == (self.batch_size, self.n_neurons, 1)
        assert np.all((actual_activations >= 0) & (actual_activations <= 1))

        expected_activations = 1 / (1 + np.exp(-dendritic_potentials))
        assert_euclidean_distance(actual_activations, expected_activations)

    @pytest.fixture
    def activations(self, dendritic_potentials):
        return sigmoid_forward(dendritic_potentials)

    def test_sigmoid_backward(self, dendritic_potentials, activations):
        """Tests that the jacobians of the sigmoid function have the expected shapes and that
        its diagonal elements are between 0 and 0.25 and that all other values are 0
        """
        jacobians = sigmoid_backward(dendritic_potentials, activations)
        assert jacobians.shape == (self.batch_size, self.n_neurons, self.n_neurons)

        # Test that its diagonal elements are between [0, 0.25]
        diagonal_elements = jacobians[:, np.arange(self.n_neurons), np.arange(self.n_neurons)]
        assert np.all((diagonal_elements >= 0) & (diagonal_elements <= 0.25))

        # Test that all off diagonal elements equal 0
        jacobians[:, np.arange(self.n_neurons), np.arange(self.n_neurons)] = 0
        assert np.all(jacobians == 0)


class TestTanhActivationFunction(TestConfig):
    def test_tanh_forward(self, dendritic_potentials):
        """Tests that the shape of the activations make sense and that all activations are between
        -1 and 1
        """
        activations = tanh_forward(dendritic_potentials)
        assert activations.shape == (self.batch_size, self.n_neurons, 1)
        assert np.all((activations >= -1) & (activations <= 1))

    @pytest.fixture
    def activations(self, dendritic_potentials):
        return tanh_forward(dendritic_potentials)

    def test_tanh_backward(self, dendritic_potentials, activations):
        """Tests that the jacobians of the tanh function have the expected shapes and that
        its diagonal elements are between 0 and 1 and that all other values are 0
        """
        jacobians = tanh_backward(dendritic_potentials, activations)
        assert jacobians.shape == (self.batch_size, self.n_neurons, self.n_neurons)

        # Test that its diagonal elements are between [0, 1]
        diagonal_elements = jacobians[:, np.arange(self.n_neurons), np.arange(self.n_neurons)]
        assert np.all((diagonal_elements >= 0) & (diagonal_elements <= 1))

        # Test that all off diagonal elements equal 0
        jacobians[:, np.arange(self.n_neurons), np.arange(self.n_neurons)] = 0
        assert np.all(jacobians == 0)


class TestSoftmaxActivationFunction(TestConfig):
    def test_softmax_forward(self, dendritic_potentials):
        """Tests expected shape of activations and whether the softmax_forward activation function
        makes sure that the sum of neurons in the current layer sums up to 1
        """
        activations = softmax_forward(dendritic_potentials)

        # Test that the shape makes sense
        assert activations.shape == (self.batch_size, self.n_neurons, 1)

        # Test that the sum of neurons in the current layer equals 1
        actual_activation_sums = np.sum(activations, axis=1, keepdims=True)
        assert actual_activation_sums.shape == (self.batch_size, 1, 1)
        expected_activation_sums = np.ones(actual_activation_sums.shape)
        np.testing.assert_almost_equal(actual_activation_sums, expected_activation_sums)

    @pytest.fixture
    def activations(self, dendritic_potentials: npt.NDArray) -> npt.NDArray:
        activations = softmax_forward(dendritic_potentials)

        return activations

    def test_softmax_backward(self, dendritic_potentials: npt.NDArray, activations: npt.NDArray):
        jacobians = softmax_backward(dendritic_potentials, activations)
        diagonal_elements = jacobians[:, np.arange(self.n_neurons), np.arange(self.n_neurons)]

        # Test that the dimensions make sense
        batch_size = activations.shape[0]
        n_neurons = activations.shape[1]
        assert jacobians.shape == (batch_size, n_neurons, n_neurons)
        assert diagonal_elements.shape == (batch_size, n_neurons)

        # Test that the diagonal elements have been placed on the diagonals of all batch_size jacobians
        for batch_idx in range(batch_size):
            for neuron_idx in range(n_neurons):
                assert jacobians[batch_idx, neuron_idx, neuron_idx] == diagonal_elements[batch_idx, neuron_idx]
