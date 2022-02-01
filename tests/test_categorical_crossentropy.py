import numpy as np
import pytest

from src.lib.activation_functions import softmax_forward
from src.lib.losses.categorical_crossentropy import CategoricalCrossEntropyLoss
from tests.test_config import TestConfig


class TestCategoricalCrossEntropyLoss(TestConfig):
    batch_size = 32
    n_neurons_out = 10

    @pytest.fixture
    def cce_simplified(self) -> CategoricalCrossEntropyLoss:
        cce = CategoricalCrossEntropyLoss(
            output_activation="softmax",
            task="multi_class_classification"
        )
        return cce

    @pytest.fixture
    def cce_general(self) -> CategoricalCrossEntropyLoss:
        # Choose any output_activation and task other than "softmax" and "multi_class_classification"
        # to invoke the call of the actual ``init_error`` method
        cce = CategoricalCrossEntropyLoss(
            output_activation="softmax",
            task="multi_label_classification"
        )
        return cce

    @pytest.fixture
    def ytrue_one_hot(self):
        ytrue_oh = np.zeros((self.batch_size, self.n_neurons_out, 1))
        # Set 1 element per row to 1
        for row_idx in range(ytrue_oh.shape[0]):
            # Get a random column index
            col_idx = np.random.randint(0, ytrue_oh.shape[1])
            # Set column element to 1
            ytrue_oh[row_idx, col_idx, 0] = 1

        # Make sure there is one ``1`` per batch-element
        assert np.sum(ytrue_oh) == self.batch_size

        return ytrue_oh

    @pytest.fixture
    def dendritic_potentials_out(self):
        dendritic_potentials = np.random.randn(self.batch_size, self.n_neurons_out, 1)

        return dendritic_potentials

    @pytest.fixture
    def activations_out(self, dendritic_potentials_out):
        activations = softmax_forward(dendritic_potentials_out)

        return activations

    def test_init_error(self, cce_simplified, cce_general, ytrue_one_hot, dendritic_potentials_out, activations_out):
        """Tests that the simplified method for error calculation yields (almost) the same result
        as the more general, unsimplified method
        """
        error_simplified = cce_simplified.init_error(ytrue_one_hot, dendritic_potentials_out, activations_out)
        error_general = cce_general.init_error(ytrue_one_hot, dendritic_potentials_out, activations_out)
        np.testing.assert_array_almost_equal(error_simplified, error_general)
