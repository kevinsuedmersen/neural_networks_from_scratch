import copy
import itertools
import logging

import numpy as np
import pytest

import src.constants as c
from src.lib.data_generators.factory import get_data_generator
from src.model_architectures import get_tiny_mlp_model
from src.utils import log_progress
from tests.test_config import TestConfig

logger = logging.getLogger(__name__)


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
    def single_training_tuple(self, img_gen_train):
        # Retrieve the first image array and first ground truth labels
        x_train, ytrue_train = next(img_gen_train)
        x_train_0 = x_train[0]
        ytrue_train_0 = ytrue_train[0]

        # Add back batch dimension to keep dimensions compatible
        x_train_0 = x_train_0[np.newaxis, ...]
        ytrue_train_0 = ytrue_train_0[np.newaxis, ...]

        return x_train_0, ytrue_train_0

    @staticmethod
    def _init_parameters(shape):
        return np.ones(shape) / 100

    @pytest.fixture
    def untrained_model(self, config_parser):
        """Untrained model instance and set weights and biases to a single value"""
        model = get_tiny_mlp_model(
            img_height=config_parser.img_height,
            img_width=config_parser.img_width,
            n_color_channels=config_parser.n_color_channels
        )
        for layer in model.layers[1:]:
            layer.weights = self._init_parameters(layer.weights.shape)
            layer.biases = self._init_parameters(layer.biases.shape)
        return model

    @pytest.fixture
    def trained_model_backprop(self, config_parser, untrained_model, single_training_tuple):
        """Model trained on a single image for 1 epoch using backpropagation"""
        # Create a deepcopy of untrained model so that untrained_model remains untrained
        model = copy.deepcopy(untrained_model)

        # Run the forward and backward pass on that single image
        x_train_0, ytrue_train_0 = single_training_tuple
        model.train_step(x_train_0, ytrue_train_0)

        return model

    @staticmethod
    def _compute_loss(untrained_model, x_train_0, ytrue_train_0):
        """Computes the loss with unchanged parameters"""
        activations_out, dendritic_potentials_out = untrained_model._forward_pass(x_train_0)
        loss = untrained_model.loss.compute_losses(ytrue_train_0, activations_out)

        return loss.item()

    @pytest.fixture
    def trained_model_brute_force(self, untrained_model, single_training_tuple, epsilon=1):
        """Computes the weight or bias gradients using a brute force method. Each derivative is
        calculated by computing the losses after slightly changing one parameter each time while
        keeping all other parameters constant, then subtracting the loss value computed with the
        constant parameters and then dividing that difference by the slight change, i.e.:
        (L(slightly_changed_parameters, all_other_parameters) - L(original_parameters)) / slight_change
        """
        # Compute the loss with unchange parameters once
        x_train_0, ytrue_train_0 = single_training_tuple
        loss_unchanged_parameters = self._compute_loss(untrained_model, x_train_0, ytrue_train_0)

        # Innit a model which will contain all trained/updated weight gradients
        trained_model = copy.deepcopy(untrained_model)

        # Iterate over parameter in every layer
        for l in range(1, untrained_model.n_layers):
            logger.info(
                f"Computing gradients of layer with index {l} (output layer index={untrained_model.n_layers - 1}) "
                f"using brute force method"
            )
            weights = untrained_model.layers[l].weights
            idx_generator = itertools.product(range(weights.shape[1]), range(weights.shape[2]))
            for counter, (row_idx, col_idx) in enumerate(idx_generator):
                log_progress(
                    counter=counter,
                    total=weights.size,
                    topic=f"Computing gradients of layer_index {l} using brute force",
                    frequency=10_000
                )

                # Make sure the next time we change a parameter, we keep all other parameters unchanged
                # The slightly_changed_model is only needed to compute the current loss value
                slightly_changed_model = copy.deepcopy(untrained_model)

                # Make sure that we always start with the same set of parameters
                np.testing.assert_array_equal(
                    slightly_changed_model.layers[l].weights,
                    self._init_parameters(slightly_changed_model.layers[l].weights.shape)
                )

                # Slightly change the parameter value
                slightly_changed_model.layers[l].weights[:, row_idx, col_idx] += epsilon

                # Compute loss with changed parameters
                loss_changed_parameters = self._compute_loss(slightly_changed_model, x_train_0, ytrue_train_0)

                # Compute and set partial derivative into the trained model
                partial_derivative = (loss_changed_parameters - loss_unchanged_parameters) / epsilon
                trained_model.layers[l].weight_gradients[:, row_idx, col_idx] = partial_derivative

        return trained_model

    @staticmethod
    def _mean_absolute_error(gradients_backprop, gradients_brute_force):
        """Calculates the Mean Absolute Error between the gradients computed during
        gradients computed using the brute force method and backpropagation
        """
        mean_absolute_error = np.mean(np.abs(gradients_brute_force - gradients_backprop))

        return mean_absolute_error

    def _compare_gradients(self, trained_model_backprop, trained_model_brute_force, gradient_type="weight_gradients"):
        """Iterates through each layer of both networks and compares gradients"""
        # Extract gradients of each layer from both models
        for l in range(1, trained_model_backprop.n_layers):
            gradients_backprop = getattr(trained_model_backprop.layers[l], gradient_type)
            gradients_brute_force = getattr(trained_model_brute_force.layers[l], gradient_type)

            # Test that all gradients are similar
            mae = self._mean_absolute_error(gradients_backprop, gradients_brute_force)
            logger.info(f"Mean Absolute Percentage Error of {gradient_type} in layer {l}: {mae}")
            np.testing.assert_allclose(gradients_backprop, gradients_brute_force)

    def test_weight_gradients(self, trained_model_backprop, trained_model_brute_force, config_parser, untrained_model):
        """Tests that the backward propagation algorithm computes the correct weight gradients"""
        self._compare_gradients(trained_model_backprop, trained_model_brute_force, "weight_gradients")
        self._compare_gradients(trained_model_backprop, trained_model_brute_force, "bias_gradients")
