import numpy as np
import pytest

from src.lib.layers.dense import DenseLayer
from src.lib.layers.input import InputLayer
from src.lib.losses.categorical_crossentropy import CategoricalCrossEntropyLoss
from src.lib.metrics.metrics import Accuracy
from src.lib.models.sequential import SequentialModel
from src.lib.optimizers.stochastic_gradient_descent import StochasticGradientDescentOptimizer
from tests.test_config import TestConfig


def get_random_scalar():
    return np.random.rand(1).item()


class TestForwardAndBackwardPropManaually(TestConfig):
    """Testing forward and backpropagation by computing the gradients manually
    with the chain rule and then asserting that the program returns the same gradients.
    The network to be tested has the following nr of neurons in the following layers:

    - Input layer: 3 neurons
    - Hidden layer: 2 neurons with sigmoid activation function
    - Output layer: 2 neurons with softmax activation function

    Notation:
    - z_l_j represents the dendritic potential of the j-th neuron in layer l
    - a_l_j represents the activation of the j-th neuron in layer l
    - b_l_j represents the bias of the j-th neuron in layer l
    - w_l_jk represents the weight that connects neuron j in layer l with neuron k in layer l-1
    """
    @pytest.fixture
    def input_data(self):
        """Some fixed input data"""
        a_0_1 = get_random_scalar()
        a_0_2 = get_random_scalar()
        a_0_3 = get_random_scalar()

        return a_0_1, a_0_2, a_0_3

    @pytest.fixture
    def weights_1(self):
        """Weights in layer 1"""
        # Weights connecting to neuron 1 in layer 1
        w_1_11 = get_random_scalar()
        w_1_12 = get_random_scalar()
        w_1_13 = get_random_scalar()

        # Weights connecting to neuron 2 in layer 1
        w_1_21 = get_random_scalar()
        w_1_22 = get_random_scalar()
        w_1_23 = get_random_scalar()

        return w_1_11, w_1_12, w_1_13, w_1_21, w_1_22, w_1_23

    @pytest.fixture
    def biases_1(self):
        """Biases in layer 1"""
        b_1_1 = get_random_scalar()
        b_1_2 = get_random_scalar()

        return b_1_1, b_1_2

    @pytest.fixture
    def weights_2(self):
        """Weights in layer 2"""
        # Weights connecting to neuron 1 in layer 2
        w_2_11 = get_random_scalar()
        w_2_12 = get_random_scalar()

        # Weights connecting to neuron 2 in layer 2
        w_2_21 = get_random_scalar()
        w_2_22 = get_random_scalar()

        return w_2_11, w_2_12, w_2_21, w_2_22

    @pytest.fixture
    def biases_2(self):
        """Biases in layer 2"""
        b_2_1 = get_random_scalar()
        b_2_2 = get_random_scalar()

        return b_2_1, b_2_2

    @pytest.fixture
    def dendritic_potentials_1(self, input_data, weights_1, biases_1):
        """Dendritic potentials in layer 1"""
        a_0_1, a_0_2, a_0_3 = input_data
        w_1_11, w_1_12, w_1_13, w_1_21, w_1_22, w_1_23 = weights_1
        b_1_1, b_1_2 = biases_1

        # Compute dendritic potentials in layer 1
        z_1_1 = a_0_1*w_1_11 + a_0_2*w_1_12 + a_0_3*w_1_13 + b_1_1
        z_1_2 = a_0_1*w_1_21 + a_0_2*w_1_22 + a_0_3*w_1_23 + b_1_2

        return z_1_1, z_1_2

    @staticmethod
    def _sigmoid_forward(z):
        return 1 / (1 + np.exp(-z))

    @pytest.fixture
    def activations_1(self, dendritic_potentials_1):
        """Activations in layer 1"""
        z_1_1, z_1_2 = dendritic_potentials_1

        # Compute activations in layer 1
        a_1_1 = self._sigmoid_forward(z_1_1)
        a_1_2 = self._sigmoid_forward(z_1_2)

        return a_1_1, a_1_2

    @pytest.fixture
    def fixed_model(self, weights_1, biases_1, weights_2, biases_2):
        """A simple MLP model with fixed weights and biases and the same architecture used in this
        test
        """
        _simple_model = SequentialModel(
            loss=CategoricalCrossEntropyLoss("softmax", "multi_class_classification"),
            metrics_train=[Accuracy("acc_train")],
            metrics_val=[Accuracy("acc_val")],
            optimizer=StochasticGradientDescentOptimizer()
        )
        _simple_model.add_layer(InputLayer(input_shape=(None, 3)))
        _simple_model.add_layer(DenseLayer(2, "tanh"))
        _simple_model.add_layer(DenseLayer(2, "tanh"))

        # Put weights and biases into matrices of the correct shape
        _weights_1 = np.asarray(weights_1).reshape((1, 2, 3))  # (batch_size, n_neurons, n_neurons_prev)
        _biases_1 = np.asarray(biases_1).reshape((1, 2, 1))  # (batch_size, n_neurons, 1)
        _weights_2 = np.asarray(weights_2).reshape((1, 2, 2))  # (batch_size, n_neurons, n_neurons_prev)
        _biases_2 = np.asarray(biases_2).reshape((1, 2, 1))  # (batch_size, n_neurons, 1)

        # Replace weights and biases in the model
        _simple_model.layers[1].weights = _weights_1
        _simple_model.layers[1].biases = _biases_1
        _simple_model.layers[2].weights = _weights_2
        _simple_model.layers[2].biases = _biases_2

        return _simple_model

    def test_forward_propagation_layer_1(
            self,
            input_data,
            dendritic_potentials_1,
            activations_1,
            fixed_model
    ):
        """Tests that the forward propagation to layer 1 has been done correctly"""
        a_0_1, a_0_2, a_0_3 = input_data
        z_1_1_expected, z_1_2_expected = dendritic_potentials_1
        a_1_1_expected, a_1_2_expected = activations_1

        # Use the model to compute the forward pass
        x_train = np.array([a_0_1, a_0_2, a_0_3]).reshape((1, 3, 1))
        activations, dendritic_potentials = fixed_model._forward_pass(x_train)
        np.testing.assert_array_equal(dendritic_potentials, fixed_model.layers[2].dendritic_potentials)
        np.testing.assert_array_equal(activations, fixed_model.layers[2].activations)

        # Extract results from layer 1
        z_1_1_actual, z_1_2_actual = fixed_model.layers[1].dendritic_potentials
        a_1_1_actual, a_1_2_actual = fixed_model.layers[1].activations

        # Verfiy resutls of layer 1
        assert z_1_1_expected == z_1_1_actual
        assert z_1_2_expected == z_1_2_actual
        assert a_1_1_expected == a_1_1_actual
        assert a_1_2_expected == a_1_2_actual
