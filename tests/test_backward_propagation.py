import copy
import itertools
import logging
import os

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
    if not os.path.exists("results"):
        os.makedirs("results")

    x_train_0_path = os.path.join("results", "x_train_0.npy")
    ytrue_train_0_path = os.path.join("results", "ytrue_train_0.npy")

    initial_costs_backprop = None
    initial_cost_brute_force = None

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

        # Save the data to disk so that later, we can verify each model always gets the same data
        np.save(self.x_train_0_path, x_train_0)
        np.save(self.ytrue_train_0_path, ytrue_train_0)

        return x_train_0, ytrue_train_0

    @staticmethod
    def _init_parameters(shape):
        np.random.seed(c.RANDOM_STATE)

        return np.random.randn(*shape)

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

    def _get_and_check_data(self, single_training_tuple):
        """Verifies that single_traiing_tuple fixture always contains the same dat"""
        x_train_0, ytrue_train_0 = single_training_tuple

        x_train_0_fix = np.load(self.x_train_0_path)
        ytrue_train_0_fix = np.load(self.ytrue_train_0_path)

        np.testing.assert_array_equal(x_train_0, x_train_0_fix)
        np.testing.assert_array_equal(ytrue_train_0, ytrue_train_0_fix)

        return x_train_0, ytrue_train_0

    def _check_parameters(self, some_untrained_model):
        """Checks that each type of learning algorithm is initialized with the same parameters"""
        for layer in some_untrained_model.layers[1:]:
            weights = self._init_parameters(layer.weights.shape)
            biases = self._init_parameters(layer.biases.shape)
            np.testing.assert_array_equal(layer.weights, weights)
            np.testing.assert_array_equal(layer.biases, biases)

    def _check_intial_settings(self, single_training_tuple, some_model):
        """Conducts a series of checks to make sure each type of learning algorithm starts with the
        same settings
        """
        x_train_0, ytrue_train_0 = self._get_and_check_data(single_training_tuple)
        self._check_parameters(some_model)

        return x_train_0, ytrue_train_0

    @staticmethod
    def _check_gradients_changed(some_trained_model, untrained_model):
        """Make sure that the gradients of some learning algorithm have actually changed after
        training
        """
        for trained_layer, untrained_layer in zip(
                some_trained_model.layers[1:],
                untrained_model.layers[1:]
        ):
            # Verify that the gradients have been initialized with zeros
            np.testing.assert_array_equal(
                untrained_layer.weight_gradients,
                np.zeros(untrained_layer.weights.shape)
            )

            # Verify that the gradients of the trained model have changed
            is_equal = np.array_equal(trained_layer.weight_gradients, untrained_layer.weight_gradients)
            assert not is_equal

    @pytest.fixture
    def trained_model_backprop(self, config_parser, untrained_model, single_training_tuple):
        """Model trained on a single image for 1 epoch using backpropagation"""
        x_train_0, ytrue_train_0 = self._check_intial_settings(single_training_tuple, untrained_model)

        # Create a deepcopy of untrained model so that untrained_model remains untrained
        trained_model = copy.deepcopy(untrained_model)

        # Run the forward and backward pass on that single image
        trained_model.train_step(x_train_0, ytrue_train_0)
        self.initial_costs_backprop = trained_model.costs[0]

        self._check_gradients_changed(trained_model, untrained_model)

        return trained_model

    @staticmethod
    def _compute_losses(untrained_model, x_train_0, ytrue_train_0):
        """Computes the loss with unchanged parameters"""
        activations_out, dendritic_potentials_out = untrained_model._forward_pass(x_train_0)
        loss = untrained_model.loss.compute_losses(ytrue_train_0, activations_out)

        return loss

    @pytest.fixture
    def trained_model_brute_force(self, untrained_model, single_training_tuple, epsilon=1):
        """Computes the weight or bias gradients using a brute force method. Each derivative is
        calculated by computing the losses after slightly changing one parameter each time while
        keeping all other parameters constant, then subtracting the loss value computed with the
        constant parameters and then dividing that difference by the slight change, i.e.:
        (L(slightly_changed_parameters, all_other_parameters) - L(original_parameters)) / slight_change
        """
        x_train_0, ytrue_train_0 = self._check_intial_settings(single_training_tuple, untrained_model)

        # Compute the loss with unchange parameters once
        losses_unchanged_parameters = self._compute_losses(untrained_model, x_train_0, ytrue_train_0)
        self.initial_cost_brute_force = untrained_model.loss.compute_cost(losses_unchanged_parameters)

        # Make sure the initial costs of both learning algorithms are equal
        assert self.initial_costs_backprop == self.initial_cost_brute_force

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
                loss_changed_parameters = self._compute_losses(slightly_changed_model, x_train_0, ytrue_train_0)

                # Compute and set partial derivative into the trained model
                partial_derivative = (loss_changed_parameters - losses_unchanged_parameters.item()) / epsilon
                trained_model.layers[l].weight_gradients[:, row_idx, col_idx] = partial_derivative

        self._check_gradients_changed(trained_model, untrained_model)

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

    def test_weight_gradients(self, trained_model_backprop, trained_model_brute_force, config_parser):
        """Tests that the backward propagation algorithm computes the correct weight gradients"""
        self._compare_gradients(trained_model_backprop, trained_model_brute_force, "weight_gradients")
        self._compare_gradients(trained_model_backprop, trained_model_brute_force, "bias_gradients")
