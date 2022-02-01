import copy
import itertools
import logging
import os
import pickle

import numpy as np
import pytest

import src.constants as c
from src.lib.data_generators.factory import get_data_generator
from src.model_architectures import get_tiny_mlp_model
from src.utils import log_progress
from tests.test_config import TestConfig

logger = logging.getLogger(__name__)


class TestGradientComputation(TestConfig):
    """Tests that the backpropagation algorithm computes the correct gradients, i.e. the same gradients
    which are computed using a brute force method
    """
    if not os.path.exists("results"):
        os.makedirs("results")

    x_train_0_path = os.path.join("fixtures", "x_train_0.npy")
    ytrue_train_0_path = os.path.join("fixtures", "ytrue_train_0.npy")
    untrained_model_path = os.path.join("fixtures", "untrained_model.pickle")

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

    def _create_new_img_data(self, img_gen_train):
        """Creates new image data using the image data generator"""
        # Retrieve the first image array and first ground truth labels
        x_train, ytrue_train = next(img_gen_train)
        x_train_0 = x_train[0]
        ytrue_train_0 = ytrue_train[0]

        # Add back batch dimension to keep dimensions compatible
        x_train_0 = x_train_0[np.newaxis, ...]
        ytrue_train_0 = ytrue_train_0[np.newaxis, ...]

        # Save to disk for subsequent tests
        np.save(self.x_train_0_path, x_train_0)
        np.save(self.ytrue_train_0_path, ytrue_train_0)

        return x_train_0, ytrue_train_0

    def _load_img_data_from_disk(self):
        """Loads existing image data training tuple from disk"""
        x_train_0 = np.load(self.x_train_0_path)
        ytrue_train_0 = np.load(self.ytrue_train_0_path)

        return x_train_0, ytrue_train_0

    @pytest.fixture
    def single_training_tuple(self, img_gen_train):
        """Returns a single training tuple of image data"""
        if (not os.path.exists(self.x_train_0_path)) or (not os.path.exists(self.ytrue_train_0_path)):
            x_train_0, ytrue_train_0 = self._create_new_img_data(img_gen_train)
        else:
            x_train_0, ytrue_train_0 = self._load_img_data_from_disk()

        return x_train_0, ytrue_train_0

    @pytest.fixture
    def x_train_0(self, single_training_tuple):
        x_train_0, _ = single_training_tuple

        return x_train_0

    @pytest.fixture
    def ytrue_train_0(self, single_training_tuple):
        _, ytrue_train_0 = single_training_tuple

        return ytrue_train_0

    def _create_new_untrained_model(self, config_parser):
        """Creates a new untrained model instance and saves it to disk"""
        untrained_model = get_tiny_mlp_model(
            img_height=config_parser.img_height,
            img_width=config_parser.img_width,
            n_color_channels=config_parser.n_color_channels
        )
        with open(self.untrained_model_path, "wb") as file:
            pickle.dump(untrained_model, file)

        return untrained_model

    def _load_untrained_model_from_disk(self):
        """Loads an untrained model from disk"""
        with open(self.untrained_model_path, "rb") as file:
            untrained_model = pickle.load(file)

        return untrained_model

    @pytest.fixture
    def untrained_model(self, config_parser):
        """Untrained model instance and set weights and biases to a single value"""
        if not os.path.exists(self.untrained_model_path):
            untrained_model = self._create_new_untrained_model(config_parser)
        else:
            untrained_model = self._load_untrained_model_from_disk()

        return untrained_model

    @pytest.fixture
    def initial_weights(self, untrained_model):
        return [layer.weights for layer in untrained_model.layers]

    @pytest.fixture
    def initial_biases(self, untrained_model):
        return [layer.biases for layer in untrained_model.layers]

    @pytest.fixture
    def initial_weight_gradients(self, untrained_model):
        return [layer.weight_gradients for layer in untrained_model.layers]

    @pytest.fixture
    def initial_bias_gradients(self, untrained_model):
        return [layer.bias_gradients for layer in untrained_model.layers]

    @staticmethod
    def _check_gradients_changed(some_trained_model, initial_weight_gradients):
        """Make sure that the gradients of some learning algorithm have actually changed after
        training
        """
        trained_weight_gradients = [layer.weight_gradients for layer in some_trained_model.layers]
        for _initial_weight_gradiends, _trained_weight_gradients in zip(
            initial_weight_gradients[1:],
            trained_weight_gradients[1:]
        ):
            # Verify that the gradients have been initialized with zeros
            np.testing.assert_array_equal(
                _initial_weight_gradiends,
                np.zeros(_initial_weight_gradiends.shape)
            )

            # Verify that the gradients of the trained model have changed
            is_equal = np.array_equal(_initial_weight_gradiends, _trained_weight_gradients)
            assert not is_equal

    @pytest.fixture
    def trained_model_backprop(self, untrained_model, x_train_0, ytrue_train_0, initial_weight_gradients):
        """Model trained on a single image for 1 epoch using backpropagation"""
        # Create a deepcopy of untrained model so that untrained_model remains untrained
        trained_model = copy.deepcopy(untrained_model)

        # Run the forward and backward pass on that single image
        trained_model.train_step(x_train_0, ytrue_train_0)
        self.initial_costs_backprop = trained_model.costs[0]

        self._check_gradients_changed(trained_model, initial_weight_gradients)

        return trained_model

    @staticmethod
    def _compute_losses(untrained_model, x_train_0, ytrue_train_0):
        """Simply computes losses without backpropagation"""
        activations_out, dendritic_potentials_out = untrained_model._forward_pass(x_train_0)
        losses = untrained_model.loss.compute_losses(ytrue_train_0, activations_out)

        return losses

    @pytest.fixture
    def trained_model_brute_force(
            self,
            untrained_model,
            x_train_0,
            ytrue_train_0,
            initial_weights,
            initial_weight_gradients,
            epsilon=1
    ):
        """Computes the weight or bias gradients using a brute force method. Each derivative is
        calculated by computing the losses after slightly changing one parameter each time while
        keeping all other parameters constant, then subtracting the loss value computed with the
        constant parameters and then dividing that difference by the slight change, i.e.:
        (L(slightly_changed_parameters, all_other_parameters) - L(original_parameters)) / slight_change
        """
        # Compute the loss with unchanged parameters once
        initial_losses = self._compute_losses(untrained_model, x_train_0, ytrue_train_0)
        self.initial_cost_brute_force = untrained_model.loss.compute_cost(initial_losses)

        # Make sure the initial costs of both learning algorithms are equal
        assert self.initial_costs_backprop == self.initial_cost_brute_force

        # Innit a model which will contain all trained/updated weight gradients
        trained_model = copy.deepcopy(untrained_model)

        # Iterate over every parameter in every layer
        for l in range(1, untrained_model.n_layers):
            logger.info(
                f"Computing gradients of layer with index {l} (output layer index={untrained_model.n_layers - 1}) "
                f"using brute force method"
            )
            n_rows = untrained_model.layers[l].weights.shape[1]
            n_cols = untrained_model.layers[l].weights.shape[2]
            idx_generator = itertools.product(range(n_rows), range(n_cols))
            for counter, (row_idx, col_idx) in enumerate(idx_generator):
                log_progress(
                    counter=counter,
                    total=(n_rows * n_cols),
                    topic=f"Computing gradients of layer_index {l} using brute force",
                    frequency=10_000
                )

                # Make sure the next time we change a parameter, we keep all other parameters unchanged
                # The _untrained_model is only needed to compute the current loss value
                _untrained_model = copy.deepcopy(untrained_model)

                # Make sure that we always start with the same set of parameters
                np.testing.assert_array_equal(_untrained_model.layers[l].weights, initial_weights[l])

                # Slightly change the parameter value
                _untrained_model.layers[l].weights[0, row_idx, col_idx] += epsilon

                # Compute loss with changed parameters
                current_losses = self._compute_losses(_untrained_model, x_train_0, ytrue_train_0)

                # Compute and set partial derivative
                partial_derivative = (current_losses - initial_losses.item()) / epsilon

                # Set partial_derivative into the trained_model
                trained_model.layers[l].weight_gradients[0, row_idx, col_idx] = partial_derivative

        self._check_gradients_changed(trained_model, initial_weight_gradients)

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
