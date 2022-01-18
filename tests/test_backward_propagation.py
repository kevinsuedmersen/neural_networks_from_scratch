import numpy as np
import pytest

from src.jobs import MLJob
from src.model_architectures import get_model
from tests.test_config import TestConfig


class TestBackwardPropagation(TestConfig):
    """Tests that the backpropagation algorithm computes the correct gradients, i.e. the same gradients
    which are computed using a brute force method
    """
    @pytest.fixture
    def ml_job(self, config_parser):
        ml_job_ = MLJob(config_parser)

        # Train the model with the backpropagation algorithm for 1 epoch
        ml_job_.cp.n_epochs = 1
        ml_job_.train_and_evaluate()

        return ml_job_

    @pytest.fixture
    def fixed_model(self, config_parser):
        return get_model(config_parser.model_name)

    def test_weight_gradients(self, ml_job, config_parser, fixed_model):
        """Tests that the backward propagation algorithm computes the correct weight gradients"""
        # Extract all weight gradients and store them in a flattened numpy array
        # Flattened, because each layer's weight matrices have different dimensions
        weight_gradients = []
        for layer in ml_job.model.layers[1:]:
            flat_weight_gradients = layer.weight_gradients.ravel().reshape(-1, 1)
            weight_gradients.append(flat_weight_gradients)
        # shape=(n_layers, n_neurons, n_neurons_prev)
        weight_gradients_backprop = np.concatenate(weight_gradients, axis=0)
        print()

        # Forward propagate ``model`` and cache the loss value

        # Compute the weight gradients using the brute force method and store them in a numpy array
        # For each training sample, for each layer, for each weight:
        # - change the weight slightly and re-compute the loss value
        # - calculate the partial derivative as (L(slightly_changed_weight) - L(original_weights)) / slight_change

    def test_bias_gradients(self, ml_job):
        """Tests that the backward propagation algorithm computes the correct bias gradients"""
        pass
