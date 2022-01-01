import numpy as np
import numpy.typing as npt
import pytest

from src.activation_functions import softmax_forward, relu_forward
from src.activation_functions.softmax import softmax_jacobian
from tests.test_config import TestConfig


class TestActivationFunction(TestConfig):
    batch_size = 32
    n_neurons = 16


class TestSoftmaxActivationFunction(TestActivationFunction):
    @pytest.fixture
    def dendritic_potentials(self) -> npt.NDArray:
        """Random dendritic potentials of layer l"""
        dendritic_potentials_ = np.random.randn(self.batch_size, self.n_neurons, 1)

        return dendritic_potentials_

    def test_softmax_forward(self, dendritic_potentials):
        """Tests expected shape of activations and
         whether the softmax_forward activation function makes sure that the sum of neurons
        in the current layer sums up to 1
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

    def test_softmax_jacobian(self, dendritic_potentials: npt.NDArray, activations: npt.NDArray):
        jacobian = softmax_jacobian(dendritic_potentials, activations)


class TestReluActivationFunction(TestActivationFunction):
    def test_relu_forward(self, dendritic_potentials):
        """Tests expected shape of activations"""
        activations = relu_forward(dendritic_potentials)

        # Test that the shape makes sense
        assert activations.shape == (self.batch_size, self.n_neurons, 1)
