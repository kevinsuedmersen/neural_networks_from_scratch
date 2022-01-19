import copy

import numpy as np
import pytest

import src.constants as c
from src.data_generators.factory import get_data_generator
from src.model_architectures import get_model
from tests.test_config import TestConfig


class TestBackwardPropagation(TestConfig):
    """Tests that the backpropagation algorithm computes the correct gradients, i.e. the same gradients
    which are computed using a brute force method
    """
    @pytest.fixture
    def img_gen_train(self, config_parser):
        img_gen = get_data_generator(
            config_parser.data_gen_name,
            config_parser.data_dir,
            config_parser.val_size,
            config_parser.test_size,
            config_parser.batch_size,
            config_parser.img_height,
            config_parser.img_width,
            c.RANDOM_STATE
        )
        img_gen_train, n_samples_train = img_gen.train()

        return img_gen_train

    @pytest.fixture
    def untrained_model(self, config_parser):
        """Untrained model instance"""
        model = get_model(
            model_name=config_parser.model_name,
            img_height=config_parser.img_height,
            img_width=config_parser.img_width,
            n_color_channels=config_parser.n_color_channels,
            random_state=c.RANDOM_STATE
        )
        return model

    @pytest.fixture
    def trained_model_backprop(self, img_gen_train, config_parser, untrained_model):
        """Model trained on a single image for 1 epoch using backpropagation"""
        # Create a deepcopy of untrained model so that untrained_model remains untrained
        model = copy.deepcopy(untrained_model)

        # Retrieve the first image array and first ground truth labels
        x_train_0 = None
        ytrue_train_0 = None
        for x_train, ytrue_train in img_gen_train:
            x_train_0 = x_train[0]
            ytrue_train_0 = ytrue_train[0]
            # Add back batch dimension to keep dimensions compatible
            x_train_0 = x_train_0[np.newaxis, ...]
            ytrue_train_0 = ytrue_train_0[np.newaxis, ...]
            break

        # Run the forward and backward pass on that single image
        model.train_step(x_train_0, ytrue_train_0)

        return model

    @staticmethod
    def _compute_gradients_backprop(trained_model_backprop, gradient_type="weight_gradients"):
        """Computes the weight or bias gradients using backpropagation"""
        # Extract all gradients and store them in a flattened numpy array
        # Flattened, because each layer's weight matrices / bias vectors have different dimensions
        gradients = []
        for layer in trained_model_backprop.layers[1:]:
            flat_gradients = getattr(layer, gradient_type).ravel().reshape(-1, 1)
            gradients.append(flat_gradients)

        # shape=(n_layers, n_neurons, n_neurons_prev)
        gradients_backprop = np.concatenate(gradients, axis=0)

        return gradients_backprop

    def _compute_gradients_brute_force(self, gradient_type="weight_gradients"):
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

    def test_weight_gradients(self, trained_model_backprop, config_parser, untrained_model, mean_absolute_percentage_error):
        """Tests that the backward propagation algorithm computes the correct weight gradients"""
        weight_gradients_backprop = self._compute_gradients_backprop(trained_model_backprop, "weight_gradients")
        weight_gradients_brute_force = self._compute_gradients_brute_force(trained_model_backprop, "weight_gradients")
        np.testing.assert_allclose(weight_gradients_backprop, weight_gradients_brute_force)
        mean_absolute_percentage_error = mean_absolute_percentage_error.result(
            weight_gradients_backprop,
            weight_gradients_brute_force
        )

    def test_bias_gradients(self, trained_model_backprop, mean_absolute_percentage_error):
        """Tests that the backward propagation algorithm computes the correct bias gradients"""
        bias_gradients_backprop = self._compute_gradients_backprop(trained_model_backprop, "bias_gradients")
        bias_gradients_brute_force = self._compute_gradients_brute_force(trained_model_backprop, "bias_gradients")
        np.testing.assert_allclose(bias_gradients_backprop, bias_gradients_brute_force)
        mean_absolute_percentage_error = mean_absolute_percentage_error.result(
            bias_gradients_backprop,
            bias_gradients_brute_force
        )
