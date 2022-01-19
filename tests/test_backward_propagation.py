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

    @staticmethod
    def _compute_gradients_backprop(ml_job, gradient_type="weight_gradients"):
        """Computes the weight or bias gradients using backpropagation"""
        # Extract all gradients and store them in a flattened numpy array
        # Flattened, because each layer's weight matrices / bias vectors have different dimensions
        gradients = []
        for layer in ml_job.model.layers[1:]:
            flat_gradients = getattr(layer, gradient_type).ravel().reshape(-1, 1)
            gradients.append(flat_gradients)

        # shape=(n_layers, n_neurons, n_neurons_prev)
        gradients_backprop = np.concatenate(gradients, axis=0)

        return gradients_backprop

    def _compute_gradients_brute_force(self, ml_job, gradient_type="weight_gradients"):
        """Computes the weight or bias gradients using a brute force method. Each derivative is
        calculated by computing the losses after slightly changing one parameter each time while
        keeping all other parameters constant, then subtracting the loss value computed with the
        constant parameters and then dividing that difference by the slight change, i.e.:
        (L(slightly_changed_parameters, all_other_parameters) - L(original_parameters)) / slight_change
        """
        pass

    @pytest.fixture
    def mean_absolute_percentage_error(self):
        pass

    def test_weight_gradients(self, ml_job, config_parser, fixed_model, mean_absolute_percentage_error):
        """Tests that the backward propagation algorithm computes the correct weight gradients"""
        weight_gradients_backprop = self._compute_gradients_backprop(ml_job, "weight_gradients")
        weight_gradients_brute_force = self._compute_gradients_brute_force(ml_job, "weight_gradients")
        np.testing.assert_allclose(weight_gradients_backprop, weight_gradients_brute_force)
        mean_absolute_percentage_error = mean_absolute_percentage_error.result(
            weight_gradients_backprop,
            weight_gradients_brute_force
        )

    def test_bias_gradients(self, ml_job, mean_absolute_percentage_error):
        """Tests that the backward propagation algorithm computes the correct bias gradients"""
        bias_gradients_backprop = self._compute_gradients_backprop(ml_job, "bias_gradients")
        bias_gradients_brute_force = self._compute_gradients_brute_force(ml_job, "bias_gradients")
        np.testing.assert_allclose(bias_gradients_backprop, bias_gradients_brute_force)
        mean_absolute_percentage_error = mean_absolute_percentage_error.result(
            bias_gradients_backprop,
            bias_gradients_brute_force
        )
