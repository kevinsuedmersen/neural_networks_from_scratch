import copy
from functools import reduce

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
    - z_l represents the vector of all dendritic potentials in layer l
    - a_l_j represents the activation of the j-th neuron in layer l
    - a_l represents the vector of all activations in layer l
    - b_l_j represents the bias of the j-th neuron in layer l
    - b_l represents the vector of all biases in layer l
    - w_l_jk represents the weight that connects neuron j in layer l with neuron k in layer l-1
    - W_l represents the matrix of all weights in layer l
    - dx__dy represents the derivative of x with respect to (w.r.t.) y
    """
    @pytest.fixture
    def a_0(self):
        """Some fixed input data"""
        a_0_1 = get_random_scalar()
        a_0_2 = get_random_scalar()
        a_0_3 = get_random_scalar()

        return a_0_1, a_0_2, a_0_3

    @pytest.fixture
    def W_1(self):
        """Weights in layer with index 1"""
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
    def b_1(self):
        """Biases in layer with index 1"""
        b_1_1 = get_random_scalar()
        b_1_2 = get_random_scalar()

        return b_1_1, b_1_2

    @pytest.fixture
    def W_2(self):
        """Weights in layer with index 2"""
        # Weights connecting to neuron 1 in layer 2
        w_2_11 = get_random_scalar()
        w_2_12 = get_random_scalar()

        # Weights connecting to neuron 2 in layer 2
        w_2_21 = get_random_scalar()
        w_2_22 = get_random_scalar()

        return w_2_11, w_2_12, w_2_21, w_2_22

    @pytest.fixture
    def b_2(self):
        """Biases in layer with index 2"""
        b_2_1 = get_random_scalar()
        b_2_2 = get_random_scalar()

        return b_2_1, b_2_2

    @pytest.fixture
    def z_1(self, a_0, W_1, b_1):
        """Dendritic potentials in layer with index 1"""
        a_0_1, a_0_2, a_0_3 = a_0
        w_1_11, w_1_12, w_1_13, w_1_21, w_1_22, w_1_23 = W_1
        b_1_1, b_1_2 = b_1

        # Compute dendritic potentials in layer 1
        z_1_1 = a_0_1*w_1_11 + a_0_2*w_1_12 + a_0_3*w_1_13 + b_1_1
        z_1_2 = a_0_1*w_1_21 + a_0_2*w_1_22 + a_0_3*w_1_23 + b_1_2

        return z_1_1, z_1_2

    @staticmethod
    def _sigmoid_forward(z):
        """Naive implementation of the sigmoid function if z is 1 dimensional"""
        return 1 / (1 + np.exp(-z))

    @pytest.fixture
    def a_1(self, z_1):
        """Activations in layer with index 1"""
        z_1_1, z_1_2 = z_1

        # Compute activations in layer 1
        a_1_1 = self._sigmoid_forward(z_1_1)
        a_1_2 = self._sigmoid_forward(z_1_2)

        return a_1_1, a_1_2

    @pytest.fixture
    def z_2(self, a_1, W_2, b_2):
        """Dendritic potentials of layer with index 2"""
        a_1_1, a_1_2 = a_1
        w_2_11, w_2_12, w_2_21, w_2_22 = W_2
        b_2_1, b_2_2 = b_2

        z_2_1 = a_1_1*w_2_11 + a_1_2*w_2_12 + b_2_1
        z_2_2 = a_1_1*w_2_21 + a_1_2*w_2_22 + b_2_2

        return z_2_1, z_2_2

    @staticmethod
    def _softmax_forward(z):
        """Naive implementation of the softmax function if z is 1 dimensional"""
        exp = np.exp(z)
        exp_sum = exp.sum()

        return exp / exp_sum

    @pytest.fixture
    def a_2(self, z_2):
        """Activations of layer with index 2, i.e. predictions"""
        z_2 = np.asarray(z_2)
        a_2_1, a_2_2 = self._softmax_forward(z_2)

        # Just making sure the softmax function works
        assert a_2_1 + a_2_2 == 1

        return a_2_1, a_2_2

    @pytest.fixture
    def untrained_model(self, W_1, b_1, W_2, b_2):
        """A simple MLP model with fixed weights and biases and the same architecture used in this
        test
        """
        _simple_model = SequentialModel(
            loss=CategoricalCrossEntropyLoss("softmax", "multi_class_classification"),
            metrics_train=[Accuracy("acc_train")],
            metrics_val=[Accuracy("acc_val")],
            optimizer=StochasticGradientDescentOptimizer()
        )
        _simple_model.add_layer(InputLayer(input_shape=(None, 3, 1)))  # TODO: Why do I need to provide input_shape?
        _simple_model.add_layer(DenseLayer(2, "sigmoid"))
        _simple_model.add_layer(DenseLayer(2, "softmax"))

        # Put weights and biases into matrices of the correct shape
        _weights_1 = np.asarray(W_1).reshape((1, 2, 3))  # (batch_size, n_neurons, n_neurons_prev)
        _biases_1 = np.asarray(b_1).reshape((1, 2, 1))  # (batch_size, n_neurons, 1)
        _weights_2 = np.asarray(W_2).reshape((1, 2, 2))  # (batch_size, n_neurons, n_neurons_prev)
        _biases_2 = np.asarray(b_2).reshape((1, 2, 1))  # (batch_size, n_neurons, 1)

        # Replace weights and biases in the model
        _simple_model.layers[1].weights = _weights_1
        _simple_model.layers[1].biases = _biases_1
        _simple_model.layers[2].weights = _weights_2
        _simple_model.layers[2].biases = _biases_2

        return _simple_model

    @pytest.fixture
    def y(self):
        """Ground truth labels"""
        y_1 = 1
        y_2 = 0

        return y_1, y_2

    @pytest.fixture
    def trained_model(self, untrained_model, a_0, y):
        """Model trained on 1 input example"""
        # Unpack fixtures
        a_0_1, a_0_2, a_0_3 = a_0
        y1, y2 = y

        # Put data into correct shapes
        x_train = np.array([a_0_1, a_0_2, a_0_3]).reshape((1, 3, 1))
        ytrue_train = np.array([y1, y2]).reshape((1, 2, 1))

        # Run forward and backward pass
        trained_model = copy.deepcopy(untrained_model)
        trained_model.train_step(x_train, ytrue_train)

        return trained_model

    def test_forward_propagation(self, a_0, z_1, a_1, z_2, a_2, trained_model):
        """Tests that the forward propagation to layer 1 has been done correctly"""
        # Unpack fixtures
        z_1_1_expected, z_1_2_expected = z_1
        a_1_1_expected, a_1_2_expected = a_1
        z_2_1_expected, z_2_2_expected = z_2
        a_2_1_expected, a_2_2_expected = a_2

        # TODO: Write separate test that the activations have been correctly forward propagated through the model
        # np.testing.assert_array_equal(dendritic_potentials, untrained_model.layers[2].dendritic_potentials)
        # np.testing.assert_array_equal(activations, untrained_model.layers[2].activations)

        # Extract results from layer 1
        z_1_1_actual, z_1_2_actual = np.squeeze(trained_model.layers[1].dendritic_potentials)
        a_1_1_actual, a_1_2_actual = np.squeeze(trained_model.layers[1].activations)

        # Verfiy resutls of layer 1
        assert z_1_1_expected == z_1_1_actual
        assert z_1_2_expected == z_1_2_actual
        assert a_1_1_expected == a_1_1_actual
        assert a_1_2_expected == a_1_2_actual

        # TODO: Put below code into separate test method
        # Extract results of layer 2
        z_2_1_actual, z_2_2_actual = np.squeeze(trained_model.layers[2].dendritic_potentials)
        a_2_1_actual, a_2_2_actual = np.squeeze(trained_model.layers[2].activations)

        # Verify results of layer 2
        assert z_2_1_expected == z_2_1_actual
        assert z_2_2_expected == z_2_2_actual
        assert a_2_1_expected == a_2_1_actual
        assert a_2_2_expected == a_2_2_actual

    @pytest.fixture
    def dL__da_2(self, y, a_2):
        """Gradient of the loss w.r.t. the activations in layer 2, given that the loss function
        is the categorical cross entropy loss function
        """
        # Unpack fixtures
        y_1, y_2 = y
        a_2_1, a_2_2 = a_2

        dL__da_2 = np.array([-y_1 / a_2_1, y_2 / a_2_2]).reshape(1, 2)

        return dL__da_2

    @pytest.fixture
    def da_2__dz_2(self, a_2):
        """Jacobian of the activations in layer 2 w.r.t. the dendritic potentials in layer 2, given
        that the activation function in layer 2 is the softmax function
        """
        # Unpack fixtures
        a_2_1, a_2_2 = a_2

        da_2__dz_2 = np.array([
            [a_2_1 * (1 - a_2_1), -a_2_1 * a_2_2],
            [-a_2_2 * a_2_1, a_2_2 * (1 - a_2_2)]
        ]).reshape(2, 2)

        return da_2__dz_2

    @pytest.fixture
    def dL__dW_2(
            self,
            dL__da_2,
            da_2__dz_2,
            a_1
    ):
        """Manually computed weight gradients of layer with index 2, i.e.
        dL/dW_2 = dL/a_2 * da_2/dz_2 * dz_2/dW_2, assuming that ytrue = [1, 0]
        """
        # Unpack fixtures
        da_2__dz_2 = da_2__dz_2
        dL__da_2 = dL__da_2
        a_1_1, a_1_2 = a_1

        # Jacobian of the dendritic potentials in layer 2 w.r.t. the weights in layer 2
        dz_2__dW_2 = np.array([
            [a_1_1, a_1_2, 0, 0],
            [0, 0, a_1_1, a_1_2]
        ]).reshape(2, 4)

        # Put it all together
        tmp = np.matmul(dL__da_2, da_2__dz_2)
        dL__dW_2 = np.matmul(tmp, dz_2__dW_2)
        dL__dW_2 = dL__dW_2.reshape(2, 2)

        return dL__dW_2

    @pytest.fixture
    def dL__dW_1(self, dL__da_2, da_2__dz_2, W_1, a_1, a_0):
        """Manually computed weight gradients of layer with index 1, i.e.
        dL/dW_1 = dL/da_2 * da_2/dz_2 * dz_2/da_1 * da_1/dz_1 * dz_1/dW_1
        fff
        Note that this assumes that the weights in layer 2 are fixed for now.
        """
        # Unpack fixtures
        w_1_11, w_1_12, w_1_13, w_1_21, w_1_22, w_1_23 = W_1
        a_1_1, a_1_2 = a_1
        a_0_1, a_0_2, a_0_3 = a_0
        
        # Derivative of the dendritic potentials in layer 2 w.r.t. the activations in layer 1
        dz_2__da_1 = np.array([
            [w_1_11, w_1_12],
            [w_1_21, w_1_22]
        ])
        
        # Derivative of the activations in layer 1 w.r.t. the dendritic potentials in layer 1, given
        # that the activation function in layer 1 is the sigmoid function
        da_1__dz_1 = np.array([
            [a_1_1 * (1 - a_1_1), 0],
            [0, a_1_2 * (1 - a_1_2)]
        ])
        
        # Derivative of the dendritic potentials in layer 1 w.r.t. the weights in layer 1
        dz_1__dW_1 = np.array([
            [a_0_1, a_0_2, a_0_3, 0, 0, 0],
            [0, 0, 0, a_0_1, a_0_2, a_0_3]
        ])
        
        # Putting it all together
        derivatives = [dL__da_2, da_2__dz_2, dz_2__da_1, da_1__dz_1, dz_1__dW_1]
        dL__dW_1 = reduce(lambda left, right: np.matmul(left, right), derivatives)
        dL__dW_1 = dL__dW_1.reshape(2, 3)
        
        return dL__dW_1

    @staticmethod
    def _compute_euclidean_distance(actual, expected):
        _actual = actual.ravel()
        _expected = expected.ravel()
        assert _actual.shape == _expected.shape
        squared_distances = (_actual - _expected)**2
        euclidean_distance = np.sqrt(np.sum(squared_distances))

        return euclidean_distance

    def _assert_euclidean_distance(self, actual, expected, absolute_tolerance=1e-6):
        """Asserts that the euclidean distance is below a certain threshold"""
        euclidean_distance = self._compute_euclidean_distance(actual, expected)
        assert euclidean_distance < absolute_tolerance

    def test_backward_propagation_layer_2(self, dL__dW_2, trained_model):
        """Tests whether the gradients of layer 2 have been computed correctly"""
        dL__dW_2_actual = trained_model.layers[2].weight_gradients
        dL__dW_2_expected = dL__dW_2
        self._assert_euclidean_distance(dL__dW_2_actual, dL__dW_2_expected)

    def test_backward_propagation_layer_1(self, dL__dW_1, trained_model):
        """Tests whether the gradients of layer 1 have been computed correctly"""
        dL__dW_1_actual = trained_model.layers[1].weight_gradients
        dL__dW_1_expected = dL__dW_1
        self._assert_euclidean_distance(dL__dW_1_actual, dL__dW_1_expected)
